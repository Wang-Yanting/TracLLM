import os
attr_type = 'tracllm'
dataset_name= 'nq-poison' #choose from 'musique','narrativeqa','qmsum', 'nq-poison', 'hotpotqa-poison', and 'msmarco-poison'
score_funcs = ['stc', 'loo','denoised_shapley']
model_name = "chatglm4-9b"
data_num = 100
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
--model_name {model_name} \
--gpu_id {gpu_id} \
--verbose 0 \
--data_num {data_num} \
> ./out/{dataset_name}_{model_name}_{attr_type}_{"_".join(score_funcs)}{K}.out &'
os.system(cmd)
