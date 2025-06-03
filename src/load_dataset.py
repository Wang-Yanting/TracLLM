'''
    Easily process & load LongBench, PoisonedRAG and NeedleInHaystack datasets.
'''
from src.utils import load_json
from datasets import load_dataset
import random
import json
from src.utils import contexts_to_sentences
def load_poison(dataset_name='nq-poison',retriever = 'contriever',top_k =5, num_poison = 5):
    result_path = f"datasets/PoisonedRAG/{dataset_name}-{retriever}-{num_poison}.json"
    results_list = load_json(result_path)
    processed_results = []
    for iter,iteration_result in enumerate(results_list):
        processed_results.extend(iteration_result[f'iter_{iter}'])
    for result in processed_results:
        result['topk_contents']=result['topk_contents'][:top_k]
        result['topk_results']=result['topk_results'][:top_k]
    print("Processed result size: ",len(processed_results))
    return processed_results
    
def insert_needle(dataset_name,haystack, needles,context_length):
    haystack ='\n'.join(haystack)
    haystack =  ' '.join(haystack.split(' ')[:context_length])
    haystack_sentences = contexts_to_sentences([haystack])
    num_sentences = len(haystack_sentences)
    
    for needle in needles:
        if dataset_name == "srt":
            inject_times =3
        elif dataset_name == "mrt":
            inject_times =1
        for iter in range(inject_times):
            # Generate a random position
            random_position = random.randint(int(num_sentences*0), num_sentences)
            
            # Insert the string at the random position
            haystack_sentences = haystack_sentences[:random_position] + [needle] + haystack_sentences[random_position:]

    return ''.join(haystack_sentences)

def load_needle(dataset_name,context_length):
    haystack_path = "datasets/NeedleInHaystack/PaulGrahamEssays.jsonl"
    # Initialize an empty list to store the JSON objects
    haystack = []

    # Open the JSONL file and read line by line
    with open(haystack_path, 'r') as file:
        for line in file:
            # Load each line as a JSON object and append to the list
            haystack.append(json.loads(line))

    haystack = [haystack[i]['text'] for i in range(20)]
    dataset = load_json(f"datasets/NeedleInHaystack/subjective_{dataset_name}.json")
    for data in dataset:
        data['needle_in_haystack'] = insert_needle(dataset_name,haystack, data['needles'],context_length)
    return dataset

def _load_dataset(dataset_name='nq-poison', retriever='contriever', retrieval_k=5, **kwargs):
    print("Load dataset: ",dataset_name)
    if dataset_name in ["narrativeqa","musique","qmsum"]:
        print("datset_name: ",dataset_name)
        dataset = load_dataset('THUDM/LongBench', dataset_name, split='test')
    elif dataset_name in ['nq-poison', 'hotpotqa-poison', 'msmarco-poison']:
        dataset = load_poison(dataset_name, retriever, retrieval_k)
    elif dataset_name in ['srt','mrt']:
        context_length = kwargs.get('context_length', 10000)
        dataset = load_needle(dataset_name,context_length)
    else: 
        raise NotImplementedError
    return dataset

        