# 🔥DisKT: Disentangled Knowledge Tracing for Alleviating Cognitive Bias (WWW2025 submitted)

PyTorch implementation of DisKT

## 🌟Data and Data Preprocessing

Place the [assist09](https://sites.google.com/view/assistmentsdatamining/dataset?authuser=0), [algebra05, algebra06](https://pslcdatashop.web.cmu.edu/KDDCup), [statics](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507), [ednet](https://github.com/riiid/ednet), [prob, comp, linux, database](https://github.com/wahr0411/PTADisc), [spanish](https://github.com/robert-lindsey/WCRP), and [slepemapy](https://www.fi.muni.cz/adaptivelearning/?a=data) source files in the dataset directory, and process the data using the following commands respectively:

```python
python preprocess_data.py --data_name assistments09
python preprocess_data.py --data_name [algebra05, bridge_algebra06]
python preprocess_data.py --data_name statics
python preprocess_data.py --data_name ednet
python preprocess_data.py --data_name [prob, sampled_comp, linux, database]
python preprocess_data.py --data_name spanish
python preprocess_data.py --data_name sampled_slepemapy
```

The statistics of the 11 datasets after processing are as follows:

| Datasets  | #students | #questions | #concepts | #concepts* | #interactions |
|-----------|-----------|------------|-----------|------------|---------------|
| assist09  | 3,644     | 17,727     | 123       | 150        | 281,890       |
| algebra05 | 571       | 173,113    | 112       | 271        | 607,014       |
| algebra06 | 1,138     | 129,263    | 493       | 550        | 1,817,450     |
| statics   | 333       | 1,223      | N/A       | N/A        | 189,297       |
| ednet     | 5,000     | 12,117     | 189       | 1,769      | 676,276       |
| prob      | 512       | 1,054      | 247       | 247        | 42,869        |
| comp      | 5,000     | 7,460      | 445       | 445        | 668,927       |
| linux     | 4,375     | 2,672      | 281       | 281        | 365,027       |
| database  | 5,488     | 3,388      | 291       | 291        | 990,468       |
| spanish   | 182       | 409        | 221       | 221        | 578,726       |
| slepemapy | 5,000     | 2,723      | 1,391     | 1,391      | 625,523       |

**Table1: Statistics of 11 datasets. "#concepts\*" denotes the total number of concepts after converting multiple concepts into a new concept.**

## ➡️Quick Start

### Installation

Git clone this repository and create conda environment:

```python
conda create -n diskt python=3.10.13
conda activate diskt
pip install -r requirements.txt 
```
### Training & Testing

Our model experiments are conducted on two NVIDIA RTX 3090 24GB GPUs. You can execute it directly using the following commands:

```python
CUDA_VISIBLE_DEVICES=0 python main.py --model_name [diskt, dkt, dkvmn, skvmn, deep_irt, gkt, sakt, akt, atkt, cl4kt, corekt, dtransformer, simplekt, folibikt, sparsekt] --data_name [assist09, algebra05, algebra06, statics, ednet, prob, comp, linux ,database, spanish, slepemapy]
```

Please use the following script to run the bias experiment:
```python
CUDA_VISIBLE_DEVICES=0 python main.py --model_name [diskt, sparsekt, akt, corekt, deep_irt] --data_name [ednet_low, ednet_medium, ednet_high] --test_name ednet_low --bias True
```