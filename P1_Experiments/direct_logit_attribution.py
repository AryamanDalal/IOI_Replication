# imports
import torch
from torch import Tensor
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from dataset_verification import load_model
from process_dataset import process_prompts_dict, Run_Details, collect_processed_prompts_lists
# Imported only for type hints. transformer_lens is already imported (with offline mode
# set) as a side effect of importing dataset_verification above, so this line is safe here.
from transformer_lens import HookedTransformer

# Chart colors: positive contributions, negative contributions, and the measured-total bar.
POS, NEG, TOTAL = "#2c7fb8", "#d95f0e", "#31a354"


@dataclass(frozen=True)
class ResidualComponents:
    head_results: Tensor        # [total_heads, N, d_model] — per-head output at the final position
    mlp_results: Tensor         # [total_mlps,  N, d_model] — per-MLP output at the final position
    token_embedding: Tensor     # [1, N, d_model]
    position_embedding: Tensor  # [1, N, d_model]
    bias: Tensor                # [1, N, d_model] — accumulated attention bias b_O, still in the residual stream


class LogitAttribution:
    # -- Section 1: construction and the state the pipeline fills ---------------------------

    def __init__(self, model: HookedTransformer, processed_prompts_list: list[Run_Details]):
        """
        Decompose the measured logit[IO] - logit[S] at the final position into the contribution
        of every residual-stream component, over one flat list of already-cached prompts.

        model                  --> a loaded HookedTransformer (run_with_cache / set_use_attn_result ready)
        processed_prompts_list --> one list of cached Run_Details, as returned by
                                   process_dataset.collect_processed_prompts_lists
        """
        self.model = model
        self.processed_prompts_list = processed_prompts_list

        # Component counts fixed by the architecture: one result per (layer, head) and one per MLP.
        self.total_heads = model.cfg.n_layers * model.cfg.n_heads
        self.total_mlps = model.cfg.n_layers

        # Populated by the pipeline below; declared here so all state is visible up front.
        self.components = None                     # ResidualComponents once collected (fields [C, N, d_model])
        self.io = self.s1 = None                   # [N] IO / S token ids
        self.direction_vectors = None              # [N, d_model] IO−S residual directions
        self.head_logit_attr = None                # [total_heads, N]
        self.mlp_logit_attr = None                 # [total_mlps, N]
        self.token_embedding_logit_attr = None     # [1, N]
        self.position_embedding_logit_attr = None  # [1, N]
        self.bias_logit_attr = None                # [1, N] — attention bias b_O attribution
        self.unembedding_bias = None               # [1, N] — b_U[IO] - b_U[S]
        self.total_contribution = None             # [N] — measured logit[IO] - logit[S] (recorded in collect_components)

    # -- Section 2: organize the residual stream into named components ----------------------

    def _split_stack(self, stack: Tensor) -> tuple[Tensor, ...]:
        """
        Divide one prompt's full residual decomposition into the five component groups the
        attribution treats separately. The decomposition orders components as
        [per-head outputs, per-MLP outputs, ..., token embedding, positional embedding, bias],
        so heads/MLPs are sliced from the front and the embeddings/bias from the back.

        stack  --> [n_components, 1, d_model] decomposition for a single prompt at the final position
        return -> (head_results, mlp_results, token_embedding, position_embedding, bias);
                  head/MLP as [C, d_model] (batch squeezed), the rest kept as [1, d_model]
        """
        head_results = stack[:self.total_heads, 0, :]
        mlp_results = stack[self.total_heads:self.total_heads + self.total_mlps, 0, :]
        return head_results, mlp_results, stack[-3], stack[-2], stack[-1]

    def collect_components(self) -> None:
        """
        Run the model once per prompt and, from each prompt's LayerNorm-scaled full residual
        decomposition at the final position, collect the five component groups plus the IO / S
        token ids. The same forward pass also yields the measured logit[IO] - logit[S] at the
        final position (the faithfulness ground truth), so it is recorded here rather than via a
        second forward pass. A single pass over the prompts fills parallel lists that are stacked
        once at the end along the prompt axis (dim=1) into a ResidualComponents bundle.
        Sets self.components, self.io, self.s1, self.total_contribution. return -> None
        """
        head_results, mlp_results = [], []
        token_embedding, position_embedding, bias = [], [], []
        io, s1, total_contribution = [], [], []

        for processed_prompt in self.processed_prompts_list:
            # Full forward pass: logits for the whole sequence + the cache for the decomposition.
            logits, cache = processed_prompt.logits, processed_prompt.cache
            stack = cache.get_full_resid_decomposition(layer=-1, pos_slice=-1, expand_neurons=False, apply_ln=True)

            # Divide the stack into its named component groups (see _split_stack).
            head, mlp, token, position, resid_bias = self._split_stack(stack)
            head_results.append(head)
            mlp_results.append(mlp)
            token_embedding.append(token)
            position_embedding.append(position)
            bias.append(resid_bias)

            # IO / S token ids, and the measured logit[IO] - logit[S] read off THIS forward pass.
            io_token, s1_token = processed_prompt.io_id, processed_prompt.s1_id
            io.append(io_token)
            s1.append(s1_token)
            logit_diff = processed_prompt.logit_diff
            total_contribution.append(logit_diff)

        # Stack each list once along the processed_prompt axis (dim=1): [C, d_model] -> [C, N, d_model].
        self.components = ResidualComponents(
            head_results=torch.stack(head_results, dim=1),
            mlp_results=torch.stack(mlp_results, dim=1),
            token_embedding=torch.stack(token_embedding, dim=1),
            position_embedding=torch.stack(position_embedding, dim=1),
            bias=torch.stack(bias, dim=1),
        )
        self.io, self.s1 = torch.tensor(io), torch.tensor(s1)
        self.total_contribution = torch.stack(total_contribution)       # [N] measured logit[IO] - logit[S]
        return None

    # -- Section 3: the attribution direction (isolated; may be redefined) ------------------

    def compute_direction_vectors(self) -> None:
        """
        Per-prompt residual-stream direction the attribution projects onto: the unembedding
        direction of the IO token minus that of the S token. Isolated in its own method so the
        attribution target can be redefined without touching the rest of the pipeline.
        Sets self.direction_vectors ([N, d_model]). return -> None
        """
        self.direction_vectors = (self.model.tokens_to_residual_directions(self.io)
                                  - self.model.tokens_to_residual_directions(self.s1)
                                  ).reshape(len(self.io), -1)   # [N, d_model]
        return None

    # -- Section 4: compute and store the logit attribution per component -------------------

    def _attribute(self, component: Tensor) -> Tensor:
        """
        Direct logit attribution of one stacked component group onto the IO−S direction:
        contract the d_model axis per prompt. [C, N, d_model] · [N, d_model] -> [C, N].
        """
        return torch.einsum("c p d, p d -> c p", component, self.direction_vectors)

    def compute_logit_attribution(self) -> None:
        """
        Project every residual component onto the IO−S direction to obtain its per-prompt logit
        attribution, and separately form the unembedding-bias term. The unembedding bias is added
        AFTER the unembedding, so (unlike the attention bias b_O, which lives in the residual
        stream) it never passes through the final LayerNorm.
        Sets the five *_logit_attr attributes and self.unembedding_bias ([1, N]). return -> None
        """
        self.head_logit_attr = self._attribute(self.components.head_results)
        self.mlp_logit_attr = self._attribute(self.components.mlp_results)
        self.token_embedding_logit_attr = self._attribute(self.components.token_embedding)
        self.position_embedding_logit_attr = self._attribute(self.components.position_embedding)
        self.bias_logit_attr = self._attribute(self.components.bias)
        self.unembedding_bias = (self.model.b_U[self.io] - self.model.b_U[self.s1]).unsqueeze(0)  # [1, N]
        return None

    def check_faithfulness(self) -> None:
        """
        Confirm the decomposition is faithful: the five component attributions plus the
        unembedding bias should reconstruct the measured logit[IO] - logit[S] per prompt.
        Prints the component/total shapes and the maximum absolute reconstruction error.
        return -> None
        """
        reconstructed = (self.head_logit_attr.sum(0) + self.mlp_logit_attr.sum(0)
                         + self.token_embedding_logit_attr.sum(0) + self.position_embedding_logit_attr.sum(0)
                         + self.bias_logit_attr.sum(0) + self.unembedding_bias.sum(0))   # [N]
        print("unembedding_bias:", tuple(self.unembedding_bias.shape), " total_contribution:", tuple(self.total_contribution.shape))
        print("max |reconstructed - total_contribution| =", (reconstructed - self.total_contribution).abs().max().item())
        return None

    def report_shapes(self) -> None:
        """Diagnostic: print the prompt count and the component / direction / attribution shapes."""
        print(len(self.processed_prompts_list))
        print(self.components.head_results.shape)
        print(self.components.mlp_results.shape)
        print(self.direction_vectors.shape)
        print(self.head_logit_attr.shape)
        print(self.mlp_logit_attr.shape)
        print(self.token_embedding_logit_attr.shape)
        print(self.position_embedding_logit_attr.shape)
        print(self.bias_logit_attr.shape)
        return None

    # -- Section 5: plotting (one chart per method) -----------------------------------------

    def plot_composition(self) -> None:
        """Chart 1 — DLA composition: the 6 components sum to the total (faithfulness), ±1 std over prompts."""
        N = self.head_logit_attr.shape[1]
        cat_series = {
            "Heads (Σ144)":       self.head_logit_attr.sum(0),
            "MLPs (Σ12)":         self.mlp_logit_attr.sum(0),
            "Attn bias (b_O)":    self.bias_logit_attr.sum(0),
            "Token embed":        self.token_embedding_logit_attr.sum(0),
            "Pos embed":          self.position_embedding_logit_attr.sum(0),
            "Unembed bias (b_U)": self.unembedding_bias.sum(0),
            "TOTAL (measured)":   self.total_contribution,
        }
        names = list(cat_series)
        means = np.array([v.mean().item() for v in cat_series.values()])
        stds  = np.array([v.std().item()  for v in cat_series.values()])
        comp_sum = means[:-1].sum()   # sum of the 6 components (excludes TOTAL)
        colors = [POS if m >= 0 else NEG for m in means]; colors[-1] = TOTAL
        y = np.arange(len(names))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(y, means, xerr=stds, color=colors, error_kw=dict(ecolor="gray", capsize=3, lw=1))
        ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8); ax.margins(x=0.18)
        ax.set_xlabel("Contribution to logit[IO] - logit[S]   (bars: mean ± 1 std over prompts)")
        ax.set_title(f"DLA composition (mean over {N} prompts)\n"
                     f"Σ 6 components = {comp_sum:.4f}   measured total = {means[-1]:.4f}   Δ = {abs(comp_sum - means[-1]):.2e}")
        for yi, mn in zip(y, means):
            ax.text(mn + (0.06 if mn >= 0 else -0.06), yi, f"{mn:+.3f}", va="center",
                    ha="left" if mn >= 0 else "right", fontsize=8)
        plt.tight_layout(); plt.show()
        return None

    def plot_head_heatmap(self) -> None:
        """Chart 2 — Per-head DLA heatmap (layer × head), mean over prompts."""
        N = self.head_logit_attr.shape[1]; n_layers, n_heads = self.model.cfg.n_layers, self.model.cfg.n_heads
        grid = self.head_logit_attr.mean(1).reshape(n_layers, n_heads).numpy()
        vmax = np.abs(grid).max()

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_xticks(range(n_heads)); ax.set_yticks(range(n_layers))
        ax.set_title(f"Per-head DLA (mean over {N} prompts)\nΣ all heads = {grid.sum():.4f}")
        for l in range(n_layers):
            for h in range(n_heads):
                v = grid[l, h]
                if abs(v) > 0.1:
                    ax.text(h, l, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > vmax * 0.6 else "black")
        fig.colorbar(im, ax=ax, label="Contribution to logit diff")
        plt.tight_layout(); plt.show()
        return None

    def plot_top_heads(self) -> None:
        """Chart 3 — Top-5 positive and top-5 negative heads (Name Movers), ±1 std over prompts."""
        n_layers, n_heads = self.model.cfg.n_layers, self.model.cfg.n_heads
        grid_mean = self.head_logit_attr.mean(1).reshape(n_layers, n_heads)
        grid_std  = self.head_logit_attr.std(1).reshape(n_layers, n_heads)
        fm, fs = grid_mean.flatten(), grid_std.flatten()
        order = torch.argsort(fm, descending=True)
        idx = torch.cat([order[:5], order[-5:]])
        labels = [f"{i.item() // n_heads}.{i.item() % n_heads}" for i in idx]
        m3, e3 = fm[idx].numpy(), fs[idx].numpy()

        fig, ax = plt.subplots(figsize=(8, 5)); y = np.arange(len(m3))
        ax.barh(y, m3, xerr=e3, color=[POS if v >= 0 else NEG for v in m3],
                error_kw=dict(ecolor="gray", capsize=3, lw=1))
        ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8); ax.margins(x=0.15)
        ax.set_xlabel("Mean DLA contribution (± 1 std over prompts)")
        ax.set_title("Top-5 positive and top-5 negative heads (Name Movers)")
        for yi, mn in zip(y, m3):
            ax.text(mn + (0.05 if mn >= 0 else -0.05), yi, f"{mn:+.2f}", va="center",
                    ha="left" if mn >= 0 else "right", fontsize=7)
        plt.tight_layout(); plt.show()
        return None

    def plot_mlp(self) -> None:
        """Chart 4 — Per-MLP DLA, ±1 std over prompts."""
        N = self.mlp_logit_attr.shape[1]; n_layers = self.model.cfg.n_layers
        m4, e4 = self.mlp_logit_attr.mean(1).numpy(), self.mlp_logit_attr.std(1).numpy()
        x = np.arange(n_layers)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x, m4, yerr=e4, color=[POS if v >= 0 else NEG for v in m4],
               error_kw=dict(ecolor="gray", capsize=2, lw=1))
        ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x)
        ax.set_xlabel("MLP layer"); ax.set_ylabel("Mean DLA (± 1 std)")
        ax.set_title(f"Per-MLP DLA (mean over {N} prompts)\nΣ all MLPs = {m4.sum():.4f}")
        plt.tight_layout(); plt.show()
        return None

    def plot_layerwise(self) -> None:
        """Chart 5 — Layerwise DLA: per-layer attention (Σ heads) and MLP, ±1 std over prompts."""
        N = self.mlp_logit_attr.shape[1]; n_layers, n_heads = self.model.cfg.n_layers, self.model.cfg.n_heads
        heads_layer = self.head_logit_attr.reshape(n_layers, n_heads, -1).sum(1)     # [n_layers, N]
        hm, he = heads_layer.mean(1).numpy(), heads_layer.std(1).numpy()
        mm, me = self.mlp_logit_attr.mean(1).numpy(), self.mlp_logit_attr.std(1).numpy()
        x = np.arange(n_layers); w = 0.4

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w/2, hm, w, yerr=he, label="Attention (Σ heads)", color=POS,
               error_kw=dict(ecolor="gray", capsize=2, lw=1))
        ax.bar(x + w/2, mm, w, yerr=me, label="MLP", color="#756bb1",
               error_kw=dict(ecolor="gray", capsize=2, lw=1))
        ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x)
        ax.set_xlabel("Layer"); ax.set_ylabel("Mean DLA (± 1 std)")
        const = (self.bias_logit_attr.mean() + self.token_embedding_logit_attr.mean()
                 + self.position_embedding_logit_attr.mean() + self.unembedding_bias.mean()).item()
        layer_sum = hm.sum() + mm.sum()
        ax.set_title(f"Layerwise DLA (mean over {N} prompts)\n"
                     f"Σ layers (heads+MLP) = {layer_sum:.3f}  +  constants (b_O, embed, pos, b_U) = {const:.3f}"
                     f"  =  total {self.total_contribution.mean():.3f}")
        ax.legend()
        plt.tight_layout(); plt.show()
        return None

    # -- Section 6: pipeline entry point ----------------------------------------------------

    def run(self) -> None:
        """
        Drive the pipeline end to end in dependency order: collect the components, form the
        IO−S direction, project one onto the other, print the diagnostics, then draw the charts.
        return -> None
        """
        self.collect_components()
        self.compute_direction_vectors()
        self.compute_logit_attribution()
        self.report_shapes()
        self.check_faithfulness()

        # The five charts.
        self.plot_composition()
        self.plot_head_heatmap()
        self.plot_top_heads()
        self.plot_mlp()
        self.plot_layerwise()
        return None


def main() -> None:
    """
    Load the model, cache a forward pass per prompt, then run the DLA pipeline for one
    configuration and render the charts. Change the orderings / sizes / prompt_types below to
    attribute a different slice of the dataset.
    """
    model = load_model()
    processed_prompts_dict = process_prompts_dict(model)

    orderings, sizes, prompt_types = ["IO_S1_S2", "S1_IO_S2"], ["small"], ["clean"]
    processed_prompts_lists: list[list[Run_Details]] = collect_processed_prompts_lists(
        processed_prompts_dict=processed_prompts_dict,
        orderings=orderings,
        sizes=sizes,
        prompt_types=prompt_types
    )
    # One prompt_type and one size were requested, so the collection holds a single list.
    processed_prompts_list: list[Run_Details] = processed_prompts_lists[0]

    attribution = LogitAttribution(model, processed_prompts_list)
    attribution.run()
    attribution.plot_head_heatmap()
    return None


if __name__ == "__main__":
    main()
