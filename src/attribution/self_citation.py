from src.prompts import wrap_prompt_self_citation
from src.utils import *
import time
from src.models import create_model
from .attribute import *
import copy
class SelfCitationAttribution(Attribution):
    def __init__(self, llm, explanation_level,K=5,self_citation_model = "self",verbose = 1):
        super().__init__(llm,explanation_level,K,verbose)
        if "gpt" not in llm.name:
            self.model = llm.model
            self.tokenizer = llm.tokenizer
        else:
            self.model = llm
        if self_citation_model == "self":
            self.explainer = self.llm
        else:
            self.explainer = create_model(f'model_configs/{self.self_citation_model}_config.json')

    def attribute(self, question:str, contexts:list, answer:str):
        def remove_numbered_patterns(input_string):
            # Define the pattern to be removed, where \d+ matches one or more digits
            pattern = r'\[\d+\]'  
            # Use re.sub() to replace all occurrences of the pattern with an empty string
            result = re.sub(pattern, '', input_string)
            result = result.replace('\n', '')
            return result
        def extract_numbers_in_order(input_string):
            # Define the pattern to match numbers within square brackets
            pattern = r'\[(\d+)\]'
            # Use re.findall() to find all occurrences of the pattern and extract the numbers
            numbers = re.findall(pattern, input_string)
            # Convert the list of strings to a list of integers
            numbers = [int(num) for num in numbers]
            return numbers
        """
        Given question, contexts and answer, return attribution results
        """
        start_time = time.time()
        texts = split_context(self.explanation_level,contexts)
        citation_texts = copy.deepcopy(texts)
        for i,sentence in enumerate(citation_texts):
            #clean up existing numbered patterns
            sentence = remove_numbered_patterns(sentence)
            citation_texts[i]=f"[{str(i)}]: "+sentence
        prompt = wrap_prompt_self_citation(question, citation_texts,answer)
        start_time = time.time()
        self_citation = self.explainer.query(prompt)
        end_time = time.time()
        print("Self Citation: ", self_citation)
        important_ids = extract_numbers_in_order(self_citation)
        important_ids = [i for i in important_ids if i < len(citation_texts)]

        print("Important ids: ", important_ids)
        importance_scores = list(range(len(important_ids), 0, -1))
        return texts,important_ids, importance_scores, end_time-start_time,None