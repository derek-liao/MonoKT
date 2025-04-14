# dataset: algebra05, bridge06, assistments09, slepemapy, sampled_comp, linux, prob, statics, spanish, csedm
# model: akt, dkt, atkt, cl4kt, corekt, deep_irt, diskt, dkvmn, dtransformer, folibikt, gkt, mikt, qiktmoe, sakt, simplekt, skvmn, sparsekt

datasets=("algebra05" "bridge06" "assistments09" "slepemapy" "sampled_comp" "linux" "prob" "statics" "spanish" "csedm")
models=("akt" "dkt" "atkt" "cl4kt" "corekt" "deep_irt" "diskt" "dkvmn" "folibikt" "gkt" "mikt" "qiktmoe" "sakt" "simplekt"  "sparsekt")
# single_gpu_models=("akt" "atkt" "corekt" "diskt" "folibikt")

# dtransformer, skvmn 太慢 等会训练

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running model: $model on dataset: $dataset"
        CUDA_VISIBLE_DEVICES=0 python main.py --model_name $model --data_name $dataset
        # Check if model is in single GPU models list
        # if [[ " ${single_gpu_models[@]} " =~ " ${model} " ]]; then
        #     CUDA_VISIBLE_DEVICES=0 python main.py --model_name $model --data_name $dataset 
        # else
        #     python main.py --model_name $model --data_name $dataset
        # fi
        rm -rf saved_model # save storage
        rm -rf wandb # save storage
    done
done

