# Multi-GPT2
The implementation of EMNLP2020-Findings paper 

**Pretrained Language Models for Dialogue Generationwith Multiple Input Sources**

![Block architecture](https://github.com/caoyu-noob/Multi-GPT2/blob/main/block.PNG)

## Requirements
1. Python 3.7
2. Pytorch==1.1.0
3. transformers==2.5.1
4. tensorboardX==2.0
5. git-python
5. tqdm

Some other dependencies may be needed, please take requirements.txt as a reference.

We run our standard experiment using one 32GB V100 GPU. If you use a GPU with smaller memory, please increase the 
`batch_split` or decrease the `train_batch_size` defined in `config.py`.

To obtain the automatic metrics, you also need to install `java`-1.8.0, `perl` and related perl library including 
`XML::Twig`, `Sort::Naturally`, `String::Util` (I use `cpanm` to install them on Linux).

## How to run

#### Download pretrained models
You need to download the small-size [GPT2 model](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin)
or [OpenAI GPT model](https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-pytorch_model.bin), rename them as 
`pytorch_model.bin` and put them under `gpt2-small` or `openai-gpt` folder respectively.

#### Download datasets
You can download PersonaChat datasets from [ParlAI](https://github.com/facebookresearch/ParlAI) or use 
[our zipped version](https://drive.google.com/file/d/1zQVO5MuEy3wBUfZpM39uYmloD3-Rld4T/view?usp=sharing)
(`train_self_original.txt` contains ~65k training samples while `train_both_original.txt` contains ~131k training samples), 
then put these raw txt files under `Dataset/ConvAI2` folder.

we also provide dummy samples `dummy.txt` under this folder which can be used for test.

#### Run the experiment
Run the experiment on the dummy data using default mean attention fusion and GPT2-based encoder-decoder model. It can 
verify whether your environment is ready for running the experiment.
```
python train.py \
--train_datasets datasets/ConvAI2/dummy.txt \
--valid_datasets datasets/ConvAI2/dummy.txt \
--test_datasets datasets/ConvAI2/dummy.txt \
--train_datasets_cache datasets/dummy_cache_gpt2 \
--valid_datasets_cache datasets/dummy_cache_gpt2 \
--test_datasets_cache datasets/dummy_cache_gpt2 \
--model_type gpt2 \
```

`model_type` can be changed to `GPT2`, `GPT` or `seq2seq`. 

Run the experiment on PersonaChat dataset using Source weight and GPT2-based encoder-decoder model.
 ```
python train.py \
--train_datasets datasets/ConvAI2/train_self_original.txt \
--valid_datasets datasets/ConvAI2/valid_self_original.txt \
--test_datasets datasets/ConvAI2/test_self_original.txt \
--train_datasets_cache datasets/train_cache_gpt2 \
--valid_datasets_cache datasets/valid_cache_gpt2 \
--test_datasets_cache datasets/test_cache_gpt2 \
--model_type gpt2 \
--attention_fusion_type sw \
--lr 5e-4 \
--extra_module_lr_rate 5.0 \
--shared_module 0 \
--shared_attention 0 \
----max_history_size 9 \
```

`train_datasets_cache` will only be created once and it differs between different base models.
`attention_fusion_type` indicates the way to fuse attentions from different sources.
`extra_module_lr_rate` indicates how many times the learning of extra module (attention fusion) is than the pretrained modules.

Run the experiment on PersonaChat using Transformer-based Seq2seq model and `single input` indicates that different 
information will be concatenated together as one input such as `SI-` models mentioned in the paper.
```
python train.py \
--train_datasets datasets/ConvAI2/train_self_original.txt \
--valid_datasets datasets/ConvAI2/valid_self_original.txt \
--test_datasets datasets/ConvAI2/test_self_original.txt \
--train_datasets_cache datasets/train_cache_gpt2 \
--valid_datasets_cache datasets/valid_cache_gpt2 \
--test_datasets_cache datasets/test_cache_gpt2 \
--model_type seq2seq \
--pointer_gen \
--single_input \
--n_epochs 50 \
```

We also provide inference script in this repo for inference using existed models. Here is an example.
```
python inference.py \
--valid_datasets datasets/ConvAI2/valid_self_original.txt \
--valid_datasets_cache datasets/valid_cache_gpt2 \
--test_datasets datasets/ConvAI2/test_self_original.txt \
--test_datasets_cache datasets/test_cache_gpt2 \
--model_type gpt2 \
--load_last ./test/best_model \
--inference_mode sampling \
--response_k 3 \
```
The default inference method is beam search, you can use top-k sampling instead.
