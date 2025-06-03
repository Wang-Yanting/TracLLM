from src.prompts import wrap_prompt
import torch
import math
from src.utils import *
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import matplotlib.pyplot as plt
class Attribution:
    def __init__(self,llm,explanation_level,K,verbose):
        self.llm = llm
        self.explanation_level = explanation_level
        self.verbose = verbose
        self.K = K
    def attribute(self):
        pass
            
    def context_value(self, question:str, contexts:list, answer:str) -> float:
        if "gpt" in self.llm.name: # use BLEU score for black-box models
            prompt = wrap_prompt(question, contexts)
            new_answer =self.llm.query(prompt)
            reference_tokens = answer.split()
            candidate_tokens = new_answer.split()

            # Calculate BLEU score
            similarity = sentence_bleu([reference_tokens], candidate_tokens)
            return similarity
        else:
            # First, encode the prompt and answer separately
            prompt = wrap_prompt(question, contexts)
            #print("prompt:", prompt)
            prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True).to(self.model.device)
            answer_ids = self.tokenizer.encode(answer, return_tensors='pt', add_special_tokens=False).to(self.model.device)

            # Aggregate token_ids by concatenating prompt_ids and answer_ids
            combined_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    
            # Compute the start position of the answer
            response_start_pos = prompt_ids.shape[1]-1
            #print("Response start position: ", response_start_pos)

            # Run the model with the combined input IDs
            with torch.no_grad():
                outputs = self.model(combined_ids)
                logits = outputs.logits

            # Shift logits and labels to align them
            shift_logits = logits[:, :-1, :]
            shift_labels = combined_ids[:, 1:]

            # Compute probabilities using softmax
            probs = torch.softmax(shift_logits, dim=-1)
            
            # Extract the probabilities corresponding to the correct next tokens
            response_probs = torch.gather(probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
            response_log_probs = torch.log(response_probs[0, response_start_pos:])

            # Compute the total log probability (value)
            value = torch.sum(response_log_probs).item()

            # Handle infinity values
            if math.isinf(value):
                value = -1000.0
            return value
    def visualize_results(self,texts,question,answer, important_ids,importance_scores, width = 200):
        #Only visualize top-K
        topk_ids,topk_scores = get_top_k(important_ids, importance_scores, self.K)
        plot_sentence_importance(question, texts, topk_ids, topk_scores, answer, width = width)

    def visualize_score_func_contribution(self,important_ids,importance_scores,ensemble_list):
        important_ids,importance_scores = get_top_k(important_ids, importance_scores, self.K)
    # Calculate the contribution of each score function
        score_func_contributions = {func: 0 for func in ensemble_list.keys()}
        for important_id in important_ids:
            max_score = 0
            for score_func in ensemble_list.keys():
                for id, score in ensemble_list[score_func]:
                    if id == important_id:
                        if score > max_score:
                            max_score = score
                            max_score_func = score_func
                        break  # Exit the loop once the id is found
            score_func_contributions[max_score_func] += 1

        plt.figure(figsize=(10, 6))
        bar_width = 0.3  # Set the bar width to be thinner
        plt.bar(score_func_contributions.keys(), score_func_contributions.values(), width=bar_width, color='skyblue')
        plt.xlabel('Score Function', fontsize=14)  # Increase font size
        plt.ylabel('Number of Important Texts', fontsize=14)  # Increase font size
        plt.title('Contribution of Each Score Function', fontsize=16)  # Increase font size
        plt.xticks(rotation=45, fontsize=13)  # Increase font size for x-ticks
        plt.yticks(fontsize=13)  # Increase font size for y-ticks
        plt.tight_layout()
        plt.show()

    def get_data_frame(self,texts,important_ids,importance_scores):
        important_ids,importance_scores = get_top_k(important_ids, importance_scores, self.K)
        data = {
            'Important Texts': [texts[id] for id in important_ids],
            'Important IDs': important_ids,
            'Importance Score': importance_scores
        }
        df = pd.DataFrame(data)
        df.style
        return df

