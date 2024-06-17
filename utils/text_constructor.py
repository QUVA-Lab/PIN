
class TextConstructor:
    def __init__(self, args):
        """
        Initialize with a codec that contains prompts and text structure.
        codec should have 'preprompt', 'middle_prompt', 'end_text' attributes.
        """
        self.end_text = '.<|endofchunk|>' if args.vlm == 'openflamingo' else "</s>" 
        self.image_token = '<image>' if args.vlm == 'openflamingo' else ""
        
        self.start_prompt = args.start_prompt
        self.middle_prompt = args.middle_prompt
        self.preprompt = self.image_token + self.start_prompt
        self.vowels = 'aeiou'

    def add_article(self, object_name):
        """Determines and returns the correct article based on the object's initial letter."""
        return 'an' if object_name[0].lower() in self.vowels else 'a'

    def construct_prompt(self, object_name):
        """
        Constructs a prompt using the given object name.
        """
        # Determine the correct article
        article = self.add_article(object_name)
        
        # Construct the text target based on reconstruction preference
        prompt = f"{self.preprompt} {article} {object_name} {self.middle_prompt}"
        return prompt

    def construct_prompt_train(self, object_name, coord, reconstruct_obj_name):
        """
        Constructs text using the given object name and coordinate.
        Adds the correct article, and assembles the text according to whether the object name should be reconstructed.
        """
        # Determine the correct article
        article = self.add_article(object_name)
        
        # Construct the text target based on reconstruction preference
        if reconstruct_obj_name:
            text_target = f"{self.preprompt} {article}"
            text = f"{text_target} {object_name} {self.middle_prompt} {coord} {self.end_text}"
        else:
            text_target = f"{self.preprompt} {article} {object_name} {self.middle_prompt}"
            text = f"{text_target} {coord}{self.end_text}"
        return text, text_target

