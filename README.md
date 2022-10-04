# Ruleformer: Context-aware Rule Mining over Knowledge Graph

This repository provides the source code & data of our paper: [Ruleformer: Context-aware Rule Mining over Knowledge Graph (COLING 2022)](https://arxiv.org/abs/2209.05815).

## Requirement

- torch==1.10.0
- numpy

The dataset should contain `train.txt, valid.txt, test.txt` in the format of `h r t` for each line, `entities.txt` listing all the entities shown in KG, `relations.txt` listing all the relations shown in KG.

## Preprocess

Run the following command to generate subgraph on your dataset:

```shell
python transformer/dataset.py -data=DATA -maxN=MAXN -padding=PADDING -jump=JUMP
```

`-data`: string, the relative/absolute path of dataset

`-maxN`: integer, filter nodes whose degree exceed it

`-padding`: integer, cutting off too long sequence

`-jump`: integer, length of rule


## Train the model

```bash
python translate.py -data=DATASET/DATA -jump=JUMP -padding=PADDING -batch_size=BATCH_SIZE -desc=DESC
```

`-d_v`, `-n_head`, `-n_layers`: integer, hyper parameters of transformers

`-subgraph`: [OPT] integer, to select another subgraph while train `JUMP` hops rule.

## Decode the rules

```bash
python translate.py -data=DATASET/DATA -jump=JUMP -padding=PADDING -batch_size=BATCH_SIZE -desc=DESC -ckpt=CKPT -decode_rule
```

`-ckpt`: string, select which checkpoint to decode rules 

`-the_rel`: float, relative threshold of the next relation

`-the_rel_min`: float, absolute threshold of the next relation

`-the_all`: float, absolute threshold of the whole rule

## Demonstration

```shell
# preprocess
python transformer/dataset.py -data=umls -maxN=40 -padding=140 -jump=3

# train
python translate.py -data=DATASET/umls -jump=3 -padding=140 -batch_size=5 -epoch=50 -n_head=6 -d_v=64 -desc=umls -savestep=5

# decode rule
python translate.py -data=DATASET/umls -jump=3 -padding=140 -batch_size=5 -epoch=50 -n_head=6 -d_v=64 -desc=umls-rule -ckpt=EXPS/umls-j3-mul-XX/TranslatorXX.ckpt -decode_rule
```

## Acknowledgement

We refer to the code of [Transformers](https://github.com/jadore801120/attention-is-all-you-need-pytorch). Thanks for their contributions.