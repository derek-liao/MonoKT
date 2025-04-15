#!/bin/bash

# dataset: algebra05, bridge06, assistments09, slepemapy, sampled_comp, linux, prob, statics, spanish, csedm
# model: akt, dkt, routerkt, cakt, atkt, cl4kt, corekt, deep_irt, diskt, dkvmn, folibikt, gkt, mikt, qiktmoe, sakt, simplekt, skvmn, sparsekt, dtransformer

datasets=("algebra05" "bridge06" "assistments09" "slepemapy" "statics" "spanish" "csedm" "sampled_comp" "linux" "prob" "database")
models=("akt" "routerkt" "cakt" "dkt" "atkt" "cl4kt" "corekt" "deep_irt" "diskt" "dkvmn" "folibikt" "gkt" "mikt" "sakt" "simplekt"  "sparsekt" "dtransformer")
# single_gpu_models=("akt" "atkt" "corekt" "diskt" "folibikt")

# dtransformer, skvmn 太慢 等会训练
# qikt 需要更大的显存，因此batch size最多只能设置为 32

task_id=0
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        gpu_id=$((task_id % 8))  # 使用取模运算分配到8张卡上
        echo "Running model: $model on dataset: $dataset using GPU $gpu_id"
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py --model_name $model --data_name $dataset &
        task_id=$((task_id + 1))
        
        # 每启动8个任务后等待一会，避免同时启动太多任务
        if [ $((task_id % 8)) -eq 0 ]; then
            wait
            # rm -rf saved_model  # save storage
            # rm -rf wandb  # save storage
        fi
    done
done

# 等待所有任务完成
wait
rm -rf saved_model  # save storage
rm -rf wandb  # save storage

