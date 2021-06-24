# Training Steps
This file provides a high-level overview of the steps we took to train the model and measure its performance. It is mostly designed to help make the training reproducible. 


## Pretraining Data
First, you will need a working install of Mecab (this is used to split sentences), along with [NEologd](https://github.com/neologd/mecab-ipadic-neologd) dictionary. The [install_mecab.sh](training/data/install_mecab.sh) script may be helpful here, but ultimately the installation method depends on your environment. After that, the data can be downloaded and preprocessed with the following commands:

```shell
wget https://dumps.wikimedia.org/other/cirrussearch/current/jawiki-20210510-cirrussearch-content.json.gz
training/data/preprocess.sh jawiki-20210510-cirrussearch-content.json.gz preprocessed.txt
pyton to_examples.py --input_data preprocessed.txt --output_data all_examples.jsonl
```

## Pretraining

Pretraining the model only requires one command, but it will take quite a while to run. Below is the command we used for training on 8 GPUs, but if you have more compute available you can tune the batch size and gradient accumulation steps accordingly. Note that we looked at performance on downstream tasks for the 15k/30k/45k/60k checkpoints and found the 45k checkpoint to perform the best.

```shell
python training/train.py \
--data all_examples.jsonl \
--logging_steps 50 \
--max_steps 60000 \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 500 \
--save_steps 500 \
--learning_rate 0.0004 \
--adam_beta2 0.98 \
--adam_epsilon 1e-06 \
--dropout 0.1 \
--weight_decay 0.01  \
--output_dir ~/runs/pretrained_shiba \
--masking_type rand_span \
--gradient_accumulation_steps 6 \
--masking_type rand_span \
--per_device_eval_batch_size 22 \
--per_device_train_batch_size 22 
```


## Livedoor News Classification

First, you'll need to get the livedoor news data and convert it to json.

```shell
wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
tar -xf ldcc-20140209.tar.gz
python training/data/livedoor_news/data_to_jsonl.py --input text --output livedoor_data.jsonl
```

Then the model can be fine-tuned like this:
```shell
python finetune_livedoor_classification.py --output_dir ~/runs/livedoor_classification --data livedoor_data.jsonl --resume_from_checkpoint ~/pretrained_checkpoint.pt --num_train_epochs 6 --save_strategy no
```

## Word Segmentation

The word segmentation fine-tuning script will download necessary data, and can be run like this:
```shell
python finetune_word_segmentation.py --output_dir ~/runs/wordseg --resume_from_checkpoint ~/pretrained_checkpoint.pt --num_train_epochs 6 --save_strategy no

```


