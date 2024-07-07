# DisKT: Disentangled Knowledge Tracing for Alleviating Cognitive Bias

To run DisKT, please prepare the configuration file (`configs/example.yaml`) and the raw dataset (e.g., `datatset/algebra05/data.txt`, `datatset/assistments09/data.csv`, etc.).

For example, the `algebra05` dataset comes from the [KDD Cup 2010 EDM Challenge](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp). Datasets need to be downloaded and put inside each corresponding data folder in `dataset`.

Please use the following script to run data preprocessing:

```
python preprocess_data.py --data_name algebra05 --min_user_inter_num 5
```

Please use the following script to run the DisKT model:

```
CUDA_VISIBLE_DEVICES=0 python main.py --model_name diskt --data_name algebra05
```

Please use the following script to run the bias experiment:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model_name diskt --data_name ednet_high --test_name ednet_low --bias True
```