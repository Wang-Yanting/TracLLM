import os
attr_type = 'tracllm'
dataset_name= 'mrt' #choose from 'srt'(single needle retrieval) and 'mrt'(multiple needles retrieval)
score_funcs = ['stc', 'loo','denoised_shapley']
model_name = "llama3.1-8b"
data_num = 100
explanation_level = "segment"
sh_N = 20
K = 5
gpu_id = 0

if not os.path.exists('./out'):
    os.makedirs('./out')

cmd = f'nohup python -u main.py \
--dataset_name {dataset_name} \
--score_funcs {" ".join(score_funcs)} \
--attr_type {attr_type} \
--K {K} \
--sh_N {sh_N} \
--explanation_level {explanation_level} \
--model_name {model_name} \
--gpu_id {gpu_id} \
--verbose 0 \
--data_num {data_num} \
> ./out/{dataset_name}_{model_name}__{attr_type}_{"_".join(score_funcs)}_{K}.out &'
os.system(cmd)