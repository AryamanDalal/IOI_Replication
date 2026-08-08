from dataclasses import dataclass
import random

@dataclass(frozen=True)
class Prompt:
    text: str  # Prompt template filled with names (leading spaces come from the name tokens)
    io: str    # Name token playing the IO role, with a leading space
    s1: str    # Name token for the first subject occurrence, with a leading space
    s2: str    # Name token for the second subject occurrence, with a leading space

@dataclass(frozen=True)
class IOIPrompt:
    template_ordering: str   # IO_S1_S2, S1_IO_S2
    template_size: str   # small, large
    clean: Prompt        # io singleton, s duplicated
    corrupt: Prompt    # ABC: three distinct names
    negative: Prompt     # clean with io/s exchanged
    scrambled: Prompt  # same names, scrambled frame of the same size

class Templates:
    def __init__(self):
        """
        Sample of IOI sentences. Each sentence is the same length and positions the N1, N2, N3 tokens at the same 
        sequence positions. There are few tokens between the {N2} and {N3} positions.
        """
        self.SMALL: list[str] = [
            "When{N1} and{N2} went to the store,{N3} gave a drink to{END}",
            "When{N1} and{N2} went to the park,{N3} gave the ball to{END}",
            "Then{N1} and{N2} went to the office,{N3} gave a pen to{END}",
            "After{N1} and{N2} went to the bar,{N3} gave the keys to{END}",
            "After{N1} and{N2} went to the ball,{N3} gave the money to{END}"
        ]
        
        # Sample of IOI sentences. Each sentence is the same length and positions the N1, N2, N3 tokens at the same
        # sequence positions. There are more tokens between the {N2} and {N3} positions compared to Small Templates.
        self.LARGE: list[str] = [
            "{N1} and{N2} went to the store. After finishing their grocery shopping they walked home, completely forgetting where the car was parked! Later, some friends arrived with news about dinner plans for tonight. Nobody could recall which aisle held that missing paper receipt either. While heading out again,{N3} gave a drink to{END}",
            "{N1} and{N2} went to the park. So many dogs were running around, playing catch, and having the best afternoon of their lives. What a beautiful day! Everyone grew tired eventually, and nobody wanted such simple fun ending, but soon enough it became time to head home.{N3} gave the ball to{END}",
            "{N1} and{N2} went to the office. Early in the morning is usually the best time for focused work. However, it can get tiring by the end of a long day. Every desk was full of people today, so nobody had any pens left. Helping out,{N3} gave a pen to{END}",
            "{N1} and{N2} went to the bar. It was a party, a night of celebration. Handing around beers, dancing with everyone in the crowded room, singing karaoke badly, they wanted to do it all. What a truly magnificent, wonderful evening! Before forgetting,{N3} gave the keys to{END}",
            "{N1} and{N2} went to the ball. A first date is always exciting, especially when it happens somewhere this grand. Cheerful couples danced into the night. There was love in the air, almost palpable. Music drifted softly around every table. Since admission remained strictly limited,{N3} gave the money to{END}"
        ] 

        # Sample of IOI sentences. Each sentence is the same length and positions the N1, N2, N3 tokens at the same
        # sequence positions as the Small Templates. There are few tokens between the {N2} and {N3} positions. Tokens between the {N2} and {N3}
        # positions have been randomly scrambled.
        self.scrambled_SMALL: list[str] = [
            "When{N1} and{N2} to store went the,{N3} gave a drink to{END}",
            "When{N1} and{N2} park went to the,{N3} gave the ball to{END}",
            "Then{N1} and{N2} office went to the,{N3} gave a pen to{END}",
            "After{N1} and{N2} the bar went to,{N3} gave the keys to{END}",
            "After{N1} and{N2} ball the went to,{N3} gave the money to{END}"
        ]

        # Sample of IOI sentences. Each sentence is the same length and positions the N1, N2, N3 tokens at the same
        # sequence positions as the Small Templates. There are as many tokens between the {N2} and {N3} positions as there are in Large Templates. Tokens between the {N2} and {N3}
        # positions have been randomly scrambled.
        self.scrambled_LARGE: list[str] = [
            "{N1} and{N2} grocery parked! finishing to about shopping which out aisle forgetting their dinner recall could store. heading that missing news for friends with car went receipt While Nobody the tonight. either. was arrived the home, held they paper completely some again Later, walked plans where After,{N3} gave a drink to{END}",
            "{N1} and{N2} around, beautiful home and best What soon nobody it wanted having dogs went fun ending, afternoon their lives. eventually, simple playing the the tired but were a head catch, to many enough and park. such Everyone to day! time of became grew So running.{N3} gave the ball to{END}",
            "{N1} and{N2} of a can focused long was Helping the time of best work. people Every tiring day. nobody office. get Early the However, left. the usually so by had desk end today, to it pens went full morning out the for in any is,{N3} gave a pen to{END}",
            "{N1} and{N2} truly the magnificent, do karaoke It What to dancing forgetting a went wanted it wonderful in badly, singing was of party, crowded Handing celebration. room, a with evening! everyone all. beers, they to around a Before bar. the night,{N3} gave the keys to{END}",
            "{N1} and{N2} when this almost it especially was remained danced palpable. went couples love the air, exciting, There the Music table. grand. is Cheerful to night. admission ball. drifted always limited happens Since date every the first A strictly into softly around somewhere in,{N3} gave the money to{END}"
        ]

        self.TEMPLATES = {
            "small": self.SMALL,   "large": self.LARGE,
            "scrambled_small": self.scrambled_SMALL, "scrambled_large": self.scrambled_LARGE,
        }
        self.BATCH_SIZE = len(self.SMALL) 
        assert len(self.SMALL) == len(self.LARGE) == len(self.scrambled_SMALL) == len(self.scrambled_LARGE)

        # Template placeholder integrity: every sentence must contain all four fill slots.
        # A missing slot makes str.format silently drop a name and corrupt the example.
        required_vars = ("{N1}", "{N2}", "{N3}", "{END}")
        for kind, sentences in self.TEMPLATES.items():
            for sentence in sentences:
                for var in required_vars:
                    assert var in sentence, f"{kind} template missing {var}: {sentence!r}"

class Names:
    def __init__(self):
        """
        Common Western first names, each intended to be a single GPT-2 token once given a
        leading space (see _name_tok). This single-token assumption is NOT verified here —
        dataset.py has no tokenizer — so validate it against the model's tokenizer upstream.
        """
        self.ALL_NAMES: list[str] = [
            "Michael",
            "Anna",
            "Tom",
            "Mike",
            "Jack",
            "Christopher",
            "Jessica",
            "Matthew",
            "Ashley",
            "Jennifer",
            "Joshua",
            "Amanda",
            "Daniel",
            "David",
            "James",
            "Dan",
            "Robert",
            "John",
            "Joseph",
            "Andrew",
            "Ryan",
            "Brandon",
            "Jason",
            "Justin",
            "Sarah",
            "William",
            "Jonathan",
            "Stephanie",
            "Brian",
            "Nicole",
            "Nicholas",
            "Anthony",
            "Heather",
            "Eric",
            "Elizabeth",
            "Adam",
            "Megan",
            "Melissa",
            "Kevin",
            "Steven",
            "Thomas",
            "Timothy",
            "Christina",
            "Kyle",
            "Rachel",
            "Laura",
            "Lauren",
            "Amber",
            "Brittany",
            "Danielle",
            "Richard",
            "Kimberly",
            "Jeffrey",
            "Amy",
            "Crystal",
            "Michelle",
            "Tiffany",
            "Jeremy",
            "Benjamin",
            "Mark",
            "Emily",
            "Aaron",
            "Charles",
            "Rebecca",
            "Jacob",
            "Stephen",
            "Patrick",
            "Sean",
            "Erin",
            "Jamie",
            "Kelly",
            "Samantha",
            "Nathan",
            "Sara",
            "Dustin",
            "Paul",
            "Angela",
            "Tyler",
            "Scott",
            "Katherine",
            "Andrea",
            "Gregory",
            "Erica",
            "Mary",
            "Travis",
            "Lisa",
            "Kenneth",
            "Bryan",
            "Lindsey",
            "Kristen",
            "Jose",
            "Alexander",
            "Alex",
            "Jesse",
            "Katie",
            "Lindsay",
            "Shannon",
            "Vanessa",
            "Courtney",
            "Christine",
            "Alicia",
            "Cody",
            "Allison",
            "Bradley",
            "Samuel",
        ]
        self.NAMES_PER_BATCH = 15
        self.NUMBER_OF_BATCHES = 5
        self.BASE_SEED = 20260716
        assert len(set(self.ALL_NAMES)) == len(self.ALL_NAMES)

    def _sample_names(self, seed: int) -> list[str]:
        """Obtain a sample of size self.NAMES_PER_BATCH from ALL_NAMES using the given seed."""
        sample_pool, names_per_batch = self.ALL_NAMES, self.NAMES_PER_BATCH
        return random.Random(seed).sample(sample_pool,names_per_batch)
    
    def sample_name_batches(self) -> list[list[tuple[str, str, str]]]:
        """
        Arrange the names into batches based on the names_per_batch. Each batch is arranged into a list of name triplets.  
        """
        assert self.NAMES_PER_BATCH % 3 == 0
        name_batches = []
        for batch_index in range(self.NUMBER_OF_BATCHES):
            seed = self.BASE_SEED + batch_index
            names = self._sample_names(seed = seed)
            # Each triple fills the three name slots (N1, N2, N3) of one prompt.
            triples = [
                tuple(names[triple_start : triple_start + 3])
                for triple_start in range(0, self.NAMES_PER_BATCH, 3)
            ]
            name_batches.append(triples)
        return name_batches
    
class IOI:
    def __init__(self):
        names, templates = Names(), Templates()
        self.name_batches = names.sample_name_batches() #[[(name1, name2, name3), ...., (name13, name14, name15)]]
        self.templates = templates.TEMPLATES # {small: List[str], large:List[str], scrambled_small: List[str], scrambled_large: List[str]}
        self.ioi_prompts : list[IOIPrompt] = []
        self.prompts = {}                               # dict[tuple[ordering, size], dict[variant, list[Prompt]]]
        # Each batch yields NAMES_PER_BATCH // 3 triples, one triple per template,
        # so the triple count must equal the number of templates (BATCH_SIZE).
        assert names.NAMES_PER_BATCH // 3 == templates.BATCH_SIZE

    def _get_position_to_role(self, template_ordering: str) -> dict[str, str]:
        """
        Computes the roles at each position n1, n2, n3 based on the template ordering.
        """
        assert template_ordering in ["IO_S1_S2", "S1_IO_S2"]
        if template_ordering == "IO_S1_S2":
            return {"N1":"io", "N2":"s1", "N3":"s2"}
        return  {"N1":"s1", "N2":"io", "N3":"s2"}

    def _get_role_to_name(self, name_triplet: tuple[str, str, str]) -> dict[str, dict[str, str]]:
        """
        Assigns the role to each name in the name triplet for each template type.
        Returns a nested dict for each template type with the name assignments for IO, S1, and S2
        """
        io_name, s_name, distractor = name_triplet
        return { "clean": {"io": io_name, "s1": s_name, "s2": s_name}, 
                "corrupt": {"io": io_name, "s1": s_name, "s2": distractor}, 
                "negative": {"io": s_name, "s1": io_name, "s2": io_name},
                "scrambled": {"io": io_name, "s1": s_name, "s2": s_name}
                }

    def _fill_single_prompt(self, prompt: str, name_triplet:  tuple[str, str, str], template_ordering: str, prompt_type: str) -> Prompt:
        """
        prompt            --> a small or large template string, filled via str.format
        name_triplet      --> (io_name, s_name, distractor) for this example
        template_ordering --> "IO_S1_S2" or "S1_IO_S2"; fixes which position holds which role
        prompt_type       --> "clean" | "corrupt" | "negative" | "scrambled"

        Derived internally: role_to_name (role -> name, e.g. {"io": name1, "s1": name2, "s2": name3})
        and position_to_role (position -> role, e.g. {"N1": "io", "N2": "s1", "N3": "s2"}).
        
        return -> Prompt(text: text, io: io, s1: s1, s2: s2)
        """
        role_to_name = self._get_role_to_name(name_triplet)[prompt_type]      # {"io": name, "s1": name, "s2": name}
        position_to_role = self._get_position_to_role(template_ordering)       # {"N1": role, "N2": role, "N3": role}
        position_to_name = {position: self._name_tok(role_to_name[role]) for position, role in position_to_role.items()}
        N1, N2, N3 = position_to_name["N1"], position_to_name["N2"], position_to_name["N3"]

        io, s1, s2 = self._name_tok(role_to_name["io"]), self._name_tok(role_to_name["s1"]), self._name_tok(role_to_name["s2"])
        text = prompt.format(N1 = N1, N2 = N2, N3 = N3, END = "")
        return Prompt(text=text, io=io, s1=s1, s2=s2 )

    def _single_ioi_prompt(self, prompt: str, scrambled_prompt: str, name_triplet:  tuple[str, str, str], template_ordering: str, template_size: str) -> IOIPrompt:
        """
        Uses _fill_single_prompt to fill out an IOIPrompt containing one Prompt per variant.
        """
        return IOIPrompt(template_ordering = template_ordering,
                         template_size = template_size,
                         clean = self._fill_single_prompt(prompt, name_triplet, template_ordering, prompt_type = "clean"),
                         corrupt = self._fill_single_prompt(prompt, name_triplet, template_ordering, prompt_type = "corrupt"),
                         negative = self._fill_single_prompt(prompt, name_triplet, template_ordering, prompt_type = "negative"),
                         scrambled = self._fill_single_prompt(scrambled_prompt, name_triplet, template_ordering, prompt_type = "scrambled")
                        )
    
    def _single_batch_index(self, prompt: str, scrambled_prompt: str, name_triplet:  tuple[str, str, str], template_ordering: str, template_size: str) -> None:
            """
            Stores the dataset generated from a single batch index (a given name triplet) with all IOI prompts. Additionally,
            stores data in a nested dict of (template_ordering, template_size) where each combination of (template_ordering, template_size)
            contains the prompt list for each prompt type.
            """
            ioi_prompt_obj = self._single_ioi_prompt(prompt, scrambled_prompt, name_triplet, template_ordering, template_size)
            self.ioi_prompts.append(ioi_prompt_obj)
            key = (template_ordering, template_size)

            self.prompts.setdefault(key, {"clean": [], "corrupt": [], "negative": [], "scrambled": []})
            self.prompts[key]["clean"].append(ioi_prompt_obj.clean)
            self.prompts[key]["corrupt"].append(ioi_prompt_obj.corrupt)
            self.prompts[key]["negative"].append(ioi_prompt_obj.negative)
            self.prompts[key]["scrambled"].append(ioi_prompt_obj.scrambled)
            return None
    
    def _process_batch(self, batch_names: list[tuple[str, str, str]]) -> None:
        """
        Uses process single batch index to process a complete batch containing list of batch_size name triplets as tuples. 
        """
        template_orderings, template_sizes = ["S1_IO_S2", "IO_S1_S2"], ["small", "large"]
        for in_batch_index, name_triplet in enumerate(batch_names):
            for template_ordering in template_orderings:
                for template_size in template_sizes:
                    prompt, scrambled_prompt = self.templates[template_size][in_batch_index], self.templates["scrambled_" + template_size][in_batch_index]
                    self._single_batch_index(prompt = prompt, scrambled_prompt = scrambled_prompt, name_triplet = name_triplet, template_ordering = template_ordering, template_size = template_size)
        return None
    
    def create_dataset(self) -> tuple[dict,list[IOIPrompt]]:
        """
        Creates the entire dataset by processing each batch one at a time using the _process_batch helper.
        """
        assert len(self.ioi_prompts) == 0
        for batch_names in self.name_batches:
            self._process_batch(batch_names = batch_names)
        return (self.prompts, self.ioi_prompts)

    def _name_tok(self, name: str) -> str:
        """Mid-sentence GPT-2 form of a name: a leading space + the name."""
        return " " + name


    """
    Create a shared class and move the prompt parameters to become input of the class. 
    """