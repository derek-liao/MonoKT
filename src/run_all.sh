# dataset: algebra05, bridge06, assistments09, slepemapy, sampled_comp, linux, prob, statics, spanish, csedm
# model: akt, dkt, atkt, cl4kt, corekt, deep_irt, diskt, dkvmn, dtransformer, folibikt, gkt, mikt, qiktmoe, sakt, simplekt, skvmn, sparsekt

datasets=("algebra05" "bridge06" "assistments09" "slepemapy" "sampled_comp" "linux" "prob" "statics" "spanish" "csedm")
models=("akt" "dkt" "atkt" "cl4kt" "corekt" "deep_irt" "diskt" "dkvmn" "dtransformer" "folibikt" "gkt" "mikt" "qiktmoe" "sakt" "simplekt" "skvmn" "sparsekt")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "Running model: $model on dataset: $dataset"
        # Add your command here to run the model on the dataset
        # For example:
        if [ "$model" == "akt" ]; then
            CUDA_VISIBLE_DEVICES=0 python main.py --model_name $model --data_name $dataset # akt only supports 1 GPU
        else
            python main.py --model_name $model --data_name $dataset
        fi
    done
done

