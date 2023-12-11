# ReLLa-hf4.35.2

## Introduction

This is another pytorch implementation of the paper [ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation](https://arxiv.org/abs/2308.11131).

The difference of this repo and the original [repo](https://github.com/LaVieEnRose365/ReLLa) is the version of ```transformers``` package. While the original implementation is using ```transformers==4.28.1```, here we implement with ```transformers==4.35.2```.

Compared with the original [repo](https://github.com/LaVieEnRose365/ReLLa), we only modify the ```finetune.py``` file. Therefore, as for data preprocessing related staff, please refer to the original [repo](https://github.com/LaVieEnRose365/ReLLa).

## Requirements
~~~python
pip install -r requirments.txt
~~~

## Quick Start

```
python -u finetune.py \
--lr 0.001 \
--dataset ml-1m \
--train_size 8192 \
--train_type sequential \
--test_type sequential \
--K 30 \
--epochs 5 \
--total_batch_size 256 \
--output_path ml-1m_lora-Vicuna/vicuna-13b-v1.5/lr_0.001_shot_8192_sequential_sequential_K_30_5_bs256 \
--test_range all \
--model_path ./models/vicuna-13b-v1.5/ \
> ml-1m_logs/vicuna-13b-v1.5/lr_0.001_shot_8192_sequential_sequential_K_30_5_bs256.txt
```

By replacing the ```finetune.py``` file in the original [repo](https://github.com/LaVieEnRose365/ReLLa), you can run the experiment with the [scripts](https://github.com/LaVieEnRose365/ReLLa/blob/main/scripts/script_finetune.py).

## Citation

```
@article{lin2023rella,
  title={ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation},
  author={Lin, Jianghao and Shan, Rong and Zhu, Chenxu and Du, Kounianhua and Chen, Bo and Quan, Shigang and Tang, Ruiming and Yu, Yong and Zhang, Weinan},
  journal={arXiv preprint arXiv:2308.11131},
  year={2023}
}
```
