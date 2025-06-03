from .attribute import *
import numpy as np
import random
from src.utils import *
import time
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cosine
class PerturbationBasedAttribution(Attribution):
    def __init__(self, llm,explanation_level = "segment",K=5, attr_type = "tracllm",score_funcs=['stc','loo','denoised_shapley'], sh_N=5,w=2,beta = 0.2,verbose =1):
        super().__init__(llm,explanation_level,K,verbose)
        self.K=K
        self.w = w
        self.sh_N = sh_N
        self.attr_type = attr_type
        self.score_funcs = score_funcs
        self.beta = beta
        if "gpt" not in self.llm.name:
            self.model = llm.model
            self.tokenizer = llm.tokenizer

        self.func_map = {
            "shapley": self.shapley_scores,
            "denoised_shapley": self.denoised_shapley_scores,
            "stc": self.stc_scores,
            "loo": self.loo_scores
        }          


    def marginal_contributions(self, question: str, contexts: list, answer: str) -> list:
        """
        Estimate the Shapley values using a Monte Carlo approximation method, handling duplicate contexts.
        
        Each occurrence of a context, even if duplicated, is treated separately.

        Parameters:
        - contexts: a list of contexts, possibly with duplicates.
        - v: a function that takes a list of contexts and returns the total value for that coalition.
        - N: the number of random permutations to consider for the approximation.

        Returns:
        - A list with every context's Shapley value.
        """

        k = len(contexts)
        
        # Initialize a list of Shapley values for each context occurrence
        shapley_values = [[] for _ in range(k)]
        count = 0

        for j in range(self.sh_N):

            # Generate a random permutation of the indices of the contexts (to handle duplicates properly)
            perm_indices = random.sample(range(k), k)
            
            # Calculate the coalition value for the empty set + cf
            coalition_value = self.context_value(question, [""], answer)
            
            for i, index in enumerate(perm_indices):
                count += 1

                # Create the coalition up to the current context (based on its index in the permutation)
                coalition = [contexts[idx] for idx in perm_indices[:i + 1]]
                coalition = sorted(coalition, key=lambda x: contexts.index(x))  # Sort based on original context order

                # Calculate the value for the current coalition
                context_value = self.context_value(question, coalition, answer)               
                marginal_contribution = context_value - coalition_value

                # Update the Shapley value for the specific context at this index
                shapley_values[index].append(marginal_contribution)
                
                # Update the coalition value for the next iteration
                coalition_value = context_value
        return shapley_values

    def shapley_scores(self, question:str, contexts:list, answer:str) -> list:
        """
        Estimate the Shapley values using a Monte Carlo approximation method.
        Parameters:
        - contexts: a list of contexts.
        - v: a function that takes a list of contexts and returns the total value for that coalition.
        - N: the number of random permutations to consider for the approximation.

        Returns:
        - A dictionary with contexts as keys and their approximate Shapley values as values.
        - A list with every context's shapley value.
        """ 
        marginal_values= self.marginal_contributions(question, contexts, answer)
        shapley_values = np.zeros(len(marginal_values))
        for i,value_list in enumerate(marginal_values):
            shapley_values[i] = np.mean(value_list)

        return shapley_values
 
    def denoised_shapley_scores(self, question:str, contexts:list, answer:str) -> list:
        marginal_values = self.marginal_contributions(question, contexts, answer)
        new_shapley_values = np.zeros(len(marginal_values))
        for i,value_list in enumerate(marginal_values):
            new_shapley_values[i] = mean_of_percent(value_list,self.beta)
        return new_shapley_values
    
    def stc_scores(self, question:str, contexts:list, answer:str) -> list:
        k = len(contexts)
        scores = np.zeros(k)
        goal_score = self.context_value(question,[''],answer)
        for i,text in enumerate(contexts):
            scores[i] = (self.context_value(question, [text], answer) - goal_score)
        return scores.tolist()

    def loo_scores(self, question:str, contexts:list, answer:str) -> list:
        k = len(contexts)
        scores = np.zeros(k)
        v_all = self.context_value(question, contexts, answer)
        for i,text in enumerate(contexts):
            rest_texts = contexts[:i] + contexts[i+1:]
            scores[i] = v_all - self.context_value(question, rest_texts, answer)
        return scores.tolist()
 
    def tracllm(self, question:str, contexts:list, answer:str, score_func):
        current_nodes =[manual_zip(contexts, list(range(len(contexts))))]
        current_nodes_scores = None
        def get_important_nodes(nodes,importance_values):
            combined = list(zip(nodes, importance_values))
            combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
            # Determine the number of top nodes to keep
            k = min(self.K, len(combined))
            top_nodes = combined_sorted[:k]
            top_nodes_sorted = sorted(top_nodes, key=lambda x: combined.index(x))

            # Extract the top k important nodes and their scores in the original order
            important_nodes = [node for node, _ in top_nodes_sorted]
            important_nodes_scores = [score for _, score in top_nodes_sorted]
            
            return important_nodes, important_nodes_scores
        level = 0

        while len(current_nodes)>0 and any(len(node) > 1 for node in current_nodes):
            level+=1
            if self.verbose == 1:
                print(f"======= layer: {level}=======")
            new_nodes = []
            for node in current_nodes:
                if len(node)>1:
                    mid = len(node) // 2
                    node_left, node_right = node[:mid], node[mid:]
                    new_nodes.append(node_left)
                    new_nodes.append(node_right)
                else:
                    new_nodes.append(node)
            if len(new_nodes)<= self.K:
                current_nodes = new_nodes   
            else:
                importance_values= self.func_map[score_func](question, [" ".join(unzip_tuples(node)[0]) for node in new_nodes], answer)

                current_nodes,current_nodes_scores = get_important_nodes(new_nodes,importance_values)
        flattened_current_nodes = [item for sublist in current_nodes for item in sublist]
        return flattened_current_nodes, current_nodes_scores

    
    def vanilla_explanation(self, question:str, texts:list, answer:str,score_func):   
        texts_scores  = self.func_map[score_func](question, texts, answer)   
        return texts,texts_scores
    def attribute(self, question:str, contexts:list, answer:str):
        
        """
        Given question, contexts and answer, return attribution results
        """

        ensemble_list = dict()
        texts = split_context(self.explanation_level,contexts)
        start_time = time.time()
        importance_dict = {}
        max_score_func_dict = {}

        score_funcs = self.score_funcs
        for score_func in score_funcs:
            if self.verbose == 1:
                print(f"-Start {score_func}")
            if score_func == "loo":
                weight = self.w
            else:
                weight = 1
            
            if self.attr_type == "tracllm":
                important_nodes,importance_scores = self.tracllm(question, texts, answer,score_func)
                important_texts, important_ids = unzip_tuples(important_nodes)
            elif self.attr_type== "vanilla_perturb":
                important_texts,importance_scores = self.vanilla_explanation(question, texts, answer,score_func)
                texts = split_context(self.explanation_level,contexts)
                important_ids = [texts.index(text) for text in important_texts]
            else:
                raise ValueError("Unsupported attr_type.")      
            
            ensemble_list[score_func] = list(zip(important_ids,importance_scores))
            for idx, important_id in enumerate(important_ids):
                if important_id in importance_dict:
                    if importance_dict[important_id]<weight*importance_scores[idx]:
                        max_score_func_dict[important_id] = score_func
                    importance_dict[important_id] = max(importance_dict[important_id],weight*importance_scores[idx])
                else:
                    importance_dict[important_id] = weight*importance_scores[idx]
                    max_score_func_dict[important_id] = score_func
            
        end_time = time.time()

        important_ids = list(importance_dict.keys())  
        importance_scores = list(importance_dict.values())
        return texts,important_ids, importance_scores, end_time-start_time,ensemble_list

    