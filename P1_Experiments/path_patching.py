# imports
from dataclasses import dataclass

from dataset_verification import load_model, make_cache_filter
from process_dataset import process_prompts_dict, Run_Details, collect_processed_prompts_lists
# transformer_lens is already imported (with offline mode set) as a side effect of importing
# dataset_verification above, so these lines are safe here. HookedTransformer is for type hints only.
from transformer_lens import utils, HookedTransformer


@dataclass(frozen=False)
class Path_Patching_Pairs:
    """
    The two runs a path patch is defined against, held index-aligned: entry i of freeze and
    entry i of patch are the same template and name triplet under two different variants.
    Left mutable so a caller can swap either side in place without rebuilding the pair.
    """
    freeze: list[Run_Details]  # the run whose prompt is re-run, and whose activations are held fixed
    patch: list[Run_Details]   # the run activations are read out of and spliced in from


class Patch_Dict_Verification:
    """
    Normalize and validate the patch specification before any forward pass happens.

    The specification is a dict keyed by what is being patched. "z" and "mlp_out" name whole
    nodes to overwrite outright; the path keys ("q_input", "k_input", "v_input", "mlp_in") each
    carry a (senders, receivers) pair naming the edges to patch. A node is a (layer, head)
    tuple, where head == n_heads is the sentinel for "this layer's MLP rather than a head".

    Verifying up front means a malformed specification fails immediately with a pointed
    assertion, rather than silently patching the wrong activation deep inside a hook.
    """

    VALID_KEYS = ["z", "mlp_out", "q_input", "k_input", "v_input", "mlp_in", "z_input"]
    QKV_KEYS = ["q_input", "k_input", "v_input"]
    PATH_KEYS = QKV_KEYS + ["mlp_in"]

    # -- Section 1: construction and node predicates ----------------------------------------

    def __init__(self, model: HookedTransformer, patch_dict: dict, effect: str):
        """
        model      --> a loaded HookedTransformer; supplies cfg.n_layers and cfg.n_heads
        patch_dict --> the raw specification, keyed by VALID_KEYS (copied, not mutated in place)
        effect     --> "direct_effect" or "total_effect"; only total_effect constrains node ordering
        """
        assert isinstance(patch_dict, dict)
        assert effect == "direct_effect" or effect == "total_effect", "Invalid effect"
        assert len(patch_dict.keys()) > 0
        self.model, self.patch_dict, self.effect = model, dict(patch_dict), effect

        # Whole nodes to overwrite, accumulated per key by _collect_nodes.
        self.z_list, self.mlp_out_list = [], []
        # Endpoints of every path edge, kept sorted by position.
        self.sender_list, self.receiver_list = [], []
        # Extremes used by the total_effect ordering check; seeded so any real node beats them.
        self.max_sender, self.min_receiver = (0, 0), (0, 0)
        self.min_z, self.min_mlp_out = (model.cfg.n_layers, 0), (model.cfg.n_layers, 0)

    def _is_mlp(self, node: tuple) -> bool:
        """A node is the layer's MLP, rather than one of its heads, when head == n_heads."""
        return node[1] == self.model.cfg.n_heads

    def _pos(self, node: tuple) -> tuple:
        """
        Sort key placing a node in computation order. Within a layer every head runs before the
        MLP, so heads collapse to 0 and the MLP keeps n_heads; comparing the resulting tuples
        answers "does this node run before that one?".

        node   --> (layer, head)
        return -> (layer, 0) for a head, (layer, n_heads) for an MLP
        """
        return (node[0], self.model.cfg.n_heads if self._is_mlp(node) else 0)

    # -- Section 2: shape and range checks on the head sets ---------------------------------

    def _verify_head_set(self, head_set) -> set:
        """Attention heads only: a non-empty set of (layer, head) tuples, both in range."""
        assert isinstance(head_set, set) and len(head_set) > 0
        assert all(isinstance(tup, tuple) and len(tup) == 2 for tup in head_set)
        assert all(isinstance(tup[0], int) and isinstance(tup[1], int) for tup in head_set)
        assert all(0 <= layer < self.model.cfg.n_layers and 0 <= head < self.model.cfg.n_heads
                   for layer, head in head_set)
        return head_set

    def _verify_mlp_head_set(self, head_set) -> set:
        """
        Heads or MLPs. Rewrites a None head to the n_heads sentinel first, so callers may write
        (layer, None) for an MLP and everything downstream sees one uniform integer encoding.
        Note the range check admits head == n_heads, which _verify_head_set rejects.
        """
        assert isinstance(head_set, set) and len(head_set) > 0
        assert all(isinstance(tup, tuple) and len(tup) == 2 for tup in head_set)
        head_set = {(layer, self.model.cfg.n_heads if head is None else head)
                    for layer, head in head_set}
        assert all(isinstance(layer, int) and isinstance(head, int) for layer, head in head_set)
        assert all(0 <= layer < self.model.cfg.n_layers and 0 <= head <= self.model.cfg.n_heads
                   for layer, head in head_set)
        return head_set

    def _verify_mlp_only_head_set(self, head_set) -> set:
        """MLPs only — used where a head would be meaningless, such as the "mlp_out" key."""
        head_set = self._verify_mlp_head_set(head_set)
        assert all(self._is_mlp(tup) for tup in head_set)
        return head_set

    def _verify_tuple_head_set(self, tuple_head_set) -> tuple:
        """
        A path key's (senders, receivers) pair. Senders may be heads or MLPs, since either can
        write to the residual stream; receivers must be heads, because these keys name a head's
        Q/K/V input.
        """
        assert isinstance(tuple_head_set, tuple) and len(tuple_head_set) == 2
        return (self._verify_mlp_head_set(tuple_head_set[0]),
                self._verify_head_set(tuple_head_set[1]))

    def _verify_tuple_mlp_head_set(self, mlp_tuple_head_set) -> tuple:
        """The "mlp_in" variant of the above: senders unrestricted, receivers must be MLPs."""
        assert isinstance(mlp_tuple_head_set, tuple) and len(mlp_tuple_head_set) == 2
        return (self._verify_mlp_head_set(mlp_tuple_head_set[0]),
                self._verify_mlp_only_head_set(mlp_tuple_head_set[1]))

    # -- Section 3: normalization and node collection ---------------------------------------

    def _expand_z_input(self) -> None:
        """
        Desugar the "z_input" convenience key, which patches one edge set into all three of Q, K
        and V at once, into the three explicit QKV keys. Any edge given both via z_input and via
        an explicit key would be patched twice, so the two are asserted disjoint before merging.
        Rewrites self.patch_dict and deletes "z_input". return -> None
        """
        if "z_input" not in self.patch_dict:
            return None
        z_senders, z_receivers = self._verify_tuple_head_set(self.patch_dict["z_input"])

        for key in self.QKV_KEYS:
            if key in self.patch_dict:
                self.patch_dict[key] = self._verify_tuple_head_set(self.patch_dict[key])
                senders, receivers = self.patch_dict[key]
                assert len(set.intersection(z_senders, senders)) == 0
                assert len(set.intersection(z_receivers, receivers)) == 0

        for key in self.QKV_KEYS:
            if key in self.patch_dict:
                senders, receivers = self.patch_dict[key]
                self.patch_dict[key] = (senders | z_senders, receivers | z_receivers)
            else:
                self.patch_dict[key] = (set(z_senders), set(z_receivers))

        del self.patch_dict["z_input"]
        return None

    def _collect_nodes(self) -> None:
        """
        Run each key through the verifier its shape demands, write the normalized value back,
        and accumulate the flat node lists the later checks read. The running minima for z and
        mlp_out are tracked here so the ordering check does not have to rescan the lists.
        Sets self.z_list, self.mlp_out_list, self.sender_list, self.receiver_list and the
        min/max extremes. return -> None
        """
        for key, value in list(self.patch_dict.items()):
            if key == "z":
                value = self._verify_head_set(value)
                self.z_list.extend(value)
                self.min_z = min(self.min_z, min(self._pos(node) for node in self.z_list))

            elif key == "mlp_out":
                value = self._verify_mlp_only_head_set(value)
                self.mlp_out_list.extend(value)
                self.min_mlp_out = min(self.min_mlp_out,
                                       min(self._pos(node) for node in self.mlp_out_list))

            elif key == "mlp_in":
                value = self._verify_tuple_mlp_head_set(value)

            else:
                value = self._verify_tuple_head_set(value)

            self.patch_dict[key] = value

            if key in self.PATH_KEYS:
                self.sender_list.extend(list(value[0]))
                self.sender_list.sort(key=self._pos)
                self.receiver_list.extend(list(value[1]))
                self.receiver_list.sort(key=self._pos)

                self.min_receiver, self.max_sender = self.receiver_list[0], self.sender_list[-1]
        return None

    # -- Section 4: cross-key consistency checks --------------------------------------------

    def _verify_disjoint(self) -> None:
        """
        No node may be claimed by two mechanisms at once. A node that is both a path endpoint and
        a whole-node overwrite, or both a sender and a receiver, would have its activation written
        twice with the second write silently deciding the result. return -> None
        """
        mlp_path_nodes = {node for node in self.sender_list + self.receiver_list
                          if self._is_mlp(node)}
        assert len(set.intersection(mlp_path_nodes, set(self.mlp_out_list))) == 0
        assert len(set.intersection(set(self.receiver_list), set(self.z_list))) == 0
        assert len(set.intersection(set(self.sender_list), set(self.z_list))) == 0
        assert len(set.intersection(set(self.sender_list), set(self.receiver_list))) == 0
        return None

    def _verify_ordering(self) -> None:
        """
        For a total effect, every sender must run strictly before every receiver and before any
        overwritten node — otherwise the patched signal would be injected after the point that was
        meant to observe it, and the measured effect would not be the intended one. A direct
        effect imposes no such constraint, since it freezes everything else anyway. return -> None
        """
        if self.effect == "total_effect" and len(self.sender_list) > 0:
            assert self._pos(self.max_sender) < self._pos(self.min_receiver)
            if len(self.z_list) > 0:
                assert self._pos(self.max_sender) < self.min_z
            if len(self.mlp_out_list) > 0:
                assert self._pos(self.max_sender) < self.min_mlp_out
        return None

    def _filter_pairs(self) -> None:
        """
        Replace each path key's (senders, receivers) pair with the explicit set of edges, keeping
        only those that run forwards. The asserts then demand every sender and every receiver
        still appear in at least one surviving edge: if one dropped out entirely, the caller named
        a node that cannot participate, and silently ignoring it would misreport the patch.
        Rewrites each path key to a set of (sender, receiver) tuples. return -> None
        """
        for key in self.PATH_KEYS:
            if key not in self.patch_dict:
                continue
            senders, receivers = self.patch_dict[key]
            pairs = {(sender, receiver) for sender in senders for receiver in receivers
                     if self._pos(sender) < self._pos(receiver)}
            assert {pair[0] for pair in pairs} == senders
            assert {pair[1] for pair in pairs} == receivers
            self.patch_dict[key] = pairs
        return None

    # -- Section 5: entry point -------------------------------------------------------------

    def run(self) -> tuple:
        """
        Normalize then validate the specification, in the one order the steps permit: z_input is
        desugared before anything counts nodes, and the checks run before the edges are expanded.
        return -> (normalized patch_dict, set of receiver nodes, set of whole nodes to overwrite)
        """
        assert all(key in self.VALID_KEYS for key in self.patch_dict.keys())
        self._expand_z_input()
        self._collect_nodes()
        self._verify_disjoint()
        self._verify_ordering()
        self._filter_pairs()
        return self.patch_dict, set(self.receiver_list), set.union(set(self.z_list), set(self.mlp_out_list))


class Patched_Run:
    """
    Execute a verified patch specification over every freeze/patch prompt pair.

    Each pair is re-run on the freeze prompt with a set of forward hooks installed. Two kinds of
    hook are assembled: whole-node patches, which overwrite a node's output with the value from
    one of the two cached runs, and path patches, which add a single sender's patch-minus-freeze
    delta into one receiver's input, leaving every other route into that receiver untouched.
    """

    # -- Section 1: construction ------------------------------------------------------------

    def __init__(self, path_patching_pairs: Path_Patching_Pairs, model: HookedTransformer, patch_dict: dict, effect: str):
        """
        path_patching_pairs --> index-aligned freeze and patch runs to sweep over
        model               --> a loaded HookedTransformer with split QKV inputs and mlp_in hooks enabled
        patch_dict          --> the raw specification; verified here, so callers pass it unnormalized
        effect              --> "direct_effect" or "total_effect"
        """
        self.model, self.effect = model, effect
        self.patch_dict, self.receiver_heads_set, self.z_and_mlp_heads_set = Patch_Dict_Verification(model=model, patch_dict=patch_dict, effect=effect).run()
        self.path_patching_pairs = path_patching_pairs
        self.fwd_hooks = []
        # patch_dict key -> the hook the receiver end writes into. Identity today, kept explicit
        # so the two can diverge without touching the loop that reads it.
        self.receiver_hook_keys = {"q_input": "q_input", "k_input": "k_input", "v_input": "v_input", "mlp_in": "mlp_in"}
        self.patched_run_details = []
        self.cache_filter = make_cache_filter(model)

    # -- Section 2: whole-node patches ------------------------------------------------------

    def _shared_patch(self, receiver_layer: int, receiver_head: int, cache) -> None:
        """
        Queue a hook overwriting one node's output with the same node's value from the given
        cache. Which cache is supplied is the whole distinction between splicing a node in and
        freezing it at its original value.

        receiver_layer --> layer the node lives in
        receiver_head  --> head index, or n_heads for the layer's MLP
        cache          --> the ActivationCache to read the replacement value from
        Appends to self.fwd_hooks. return -> None
        """
        key = "z" if receiver_head != self.model.cfg.n_heads else "mlp_out"
        name = utils.get_act_name(key, receiver_layer)

        def patch_hook(activation, hook):
            layer = hook.layer()
            # z carries a head axis to index into; mlp_out does not, so it takes a full slice.
            index = (slice(None), slice(None), receiver_head, slice(None)) if receiver_head != self.model.cfg.n_heads else slice(None)

            if layer == receiver_layer:
                activation[index] = cache[key, layer][index]
            return activation
        self.fwd_hooks.append((name, patch_hook))
        return None

    def _layer_patch(self, receiver_layer: int, patch_cache, freeze_cache) -> None:
        """
        Decide, for every node in one layer, which cache (if any) its output is taken from.
        Receivers are skipped because their inputs are handled by the path patches instead;
        nodes named for overwriting read from the patch cache; and under a direct effect every
        remaining node is pinned to the freeze cache, which is what confines the measured effect
        to the patched paths alone. Under a total effect the remainder is left free to respond.

        receiver_layer --> the layer to sweep
        Appends to self.fwd_hooks. return -> None
        """
        for receiver_head in range(self.model.cfg.n_heads + 1):
            pair = (receiver_layer, receiver_head)
            if pair in self.receiver_heads_set:
                continue
            elif pair in self.z_and_mlp_heads_set:
                self._shared_patch(receiver_layer, receiver_head, patch_cache)
            elif self.effect == "direct_effect":
                self._shared_patch(receiver_layer, receiver_head, freeze_cache)
        return None

    # -- Section 3: path patches ------------------------------------------------------------

    def _sender_delta(self, sender_layer: int, sender_head: int, patch_cache, freeze_cache):
        """patch − freeze contribution of one sender. [batch, seq, d_model]"""
        if sender_head == self.model.cfg.n_heads:
            return patch_cache["mlp_out", sender_layer] - freeze_cache["mlp_out", sender_layer]
        z_delta = (patch_cache["z", sender_layer][:, :, sender_head, :]
                - freeze_cache["z", sender_layer][:, :, sender_head, :])
        # [batch, seq, d_head] @ [d_head, d_model] -> the delta this head writes to the residual stream.
        return z_delta @ self.model.W_O[sender_layer, sender_head]

    def _shared_path_patch(self, receiver_key: str, sender_layer: int, sender_head: int, receiver_layer: int, receiver_head: int, patch_cache, freeze_cache) -> None:
        """
        Queue a hook adding one sender's delta into one receiver's input. Adding the delta rather
        than overwriting the input is what isolates the single edge: every other contribution to
        that input survives untouched, so the difference measured afterwards is attributable to
        this path alone. The delta is computed once now, outside the hook, since neither cache
        changes during the run.

        receiver_key --> which input hook to write into: "q_input" | "k_input" | "v_input" | "mlp_in"
        Appends to self.fwd_hooks. return -> None
        """
        name = utils.get_act_name(receiver_key, receiver_layer)
        receiver_index = (slice(None), slice(None), receiver_head, slice(None)) if receiver_head != self.model.cfg.n_heads else slice(None)
        delta = self._sender_delta(sender_layer, sender_head, patch_cache, freeze_cache)

        def patch_hook(activation, hook):
            layer = hook.layer()

            if layer == receiver_layer:
                activation[receiver_index] += delta
            return activation
        self.fwd_hooks.append((name, patch_hook))
        return None

    def _run_shared_path_patches(self, freeze_cache, patch_cache) -> None:
        """Queue one path patch per (sender, receiver) edge, across every path key present."""
        for key, receiver_key in self.receiver_hook_keys.items():
            if key not in self.patch_dict:
                continue

            head_set = self.patch_dict[key]
            for (sender, receiver) in head_set:
                sender_layer, sender_head = sender
                receiver_layer, receiver_head = receiver
                self._shared_path_patch(receiver_key=receiver_key,
                                        sender_layer=sender_layer,
                                        sender_head=sender_head,
                                        receiver_layer=receiver_layer,
                                        receiver_head=receiver_head,
                                        patch_cache=patch_cache,
                                        freeze_cache=freeze_cache
                )
        return None

    def _run_layer_patches(self, patch_cache, freeze_cache) -> None:
        """Queue the whole-node patches for every layer in the model."""
        for receiver_layer in range(self.model.cfg.n_layers):
            self._layer_patch(receiver_layer=receiver_layer,
                              patch_cache=patch_cache,
                              freeze_cache=freeze_cache
            )
        return None

    # -- Section 4: hook assembly and the patched forward pass ------------------------------

    def _get_fwd_hooks(self, patch_cache, freeze_cache) -> list:
        """
        Assemble the full hook list for one pair, from an empty list each time so hooks never
        accumulate across pairs. Layer patches are queued before path patches: a receiver's input
        must already carry its frozen or spliced value before an edge delta is added on top.
        return -> [(hook name, hook fn)] ready for model.hooks(fwd_hooks=...)
        """
        self.fwd_hooks = []
        self._run_layer_patches(patch_cache=patch_cache, freeze_cache=freeze_cache)
        self._run_shared_path_patches(patch_cache=patch_cache, freeze_cache=freeze_cache)
        return self.fwd_hooks

    def _run_single_pair(self, freeze_run_details: Run_Details, patch_run_details: Run_Details) -> None:
        """
        Re-run the freeze prompt with this pair's hooks installed and record the result in the
        same Run_Details shape the unpatched runs use, so patched and unpatched runs stay directly
        comparable. The prompt and token ids come from the freeze side, since that is the run
        being intervened on; only the activations come from the patch side.
        Appends to self.patched_run_details. return -> None
        """
        fwd_hooks = self._get_fwd_hooks(patch_cache=patch_run_details.cache, freeze_cache=freeze_run_details.cache)

        with self.model.hooks(fwd_hooks=fwd_hooks):
            logits, cache = self.model.run_with_cache(freeze_run_details.prompt.text, names_filter=self.cache_filter)
            run_details = Run_Details(prompt=freeze_run_details.prompt,
                                      cache=cache,
                                      logits=logits,
                                      io_id=freeze_run_details.io_id,
                                      s1_id=freeze_run_details.s1_id,
                                      logit_diff=logits[0, -1, freeze_run_details.io_id] - logits[0, -1, freeze_run_details.s1_id]
            )
            self.patched_run_details.append(run_details)
        return None

    def run(self) -> list[Run_Details]:
        """
        Sweep every freeze/patch pair, patching each in turn.
        return -> one patched Run_Details per pair, in the order the pairs were given
        """
        for index in range(len(self.path_patching_pairs.freeze)):
            freeze_run_details = self.path_patching_pairs.freeze[index]
            patch_run_details = self.path_patching_pairs.patch[index]
            self._run_single_pair(freeze_run_details=freeze_run_details, patch_run_details=patch_run_details)
        return self.patched_run_details


def main() -> None:
    """
    Load the model, cache a forward pass per prompt, then pair the clean and corrupt runs into
    the freeze/patch structure a path patch is defined against. Change the orderings / sizes /
    prompt_types below to pair a different slice of the dataset.
    """
    model = load_model()
    processed_prompts_dict = process_prompts_dict(model)

    orderings, sizes, prompt_types = ["IO_S1_S2", "S1_IO_S2"], ["small"], ["clean", "corrupt"]
    processed_prompts_lists: list[list[Run_Details]] = collect_processed_prompts_lists(
        processed_prompts_dict=processed_prompts_dict,
        orderings=orderings,
        sizes=sizes,
        prompt_types=prompt_types
    )

    # Two prompt_types were requested and the lists come back prompt_types-major, so the clean
    # run is frozen and the corrupt run is the one patched in.
    freeze_prompts, patch_prompts = processed_prompts_lists
    assert len(freeze_prompts) == len(patch_prompts)
    path_patching_pairs = Path_Patching_Pairs(freeze=freeze_prompts, patch=patch_prompts)
    return None


if __name__ == "__main__":
    main()
