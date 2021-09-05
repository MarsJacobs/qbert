# BERT Quantization Project

## Introduction
This is Minsoo's Private Repository for RoBERTa Quantization Project.

This Repository is based on Fairseq based I-BERT base Branch Code.
Github Link : https://github.com/kssteven418/I-BERT 
(please use "base" branch)

## PACT Quantization

This Repository contains PACT Quantization Implementation.
Following table is 4bit FFN Weight Quantization Experiment Results. (QAT While RoBERTa model SST-2 Task Down stream Fine Tuning Stage) 

**SST-2 Full Precision Accuracy : 94.6**
4bit Weight Quantization on FFN Layer

Init Clip Value | LR Scale | Weight Decay | Accruacy
-- | -- | -- | --
1 | 1000 | 0.5 | **94.2** (-0.4)
1 | 100 | 0.5 | 93.8
1 | 10 | 0.4 | 86.9


## LSQ Quantization

This Repository also contains LSQ Quantization Implementation.  
LSQ Quantization is applied on SST-2 Task Downstreamed Model. (QAT after RoBERTa model is downstreamed to SST-2 Task Model)

![image](https://user-images.githubusercontent.com/54992207/132116336-3a058b03-2f4f-4eef-ba3b-c1b626d5251b.png)


We Quantize the weights of Word Embedding Layer, Feed Forward Network's FC Linear Layer, QKV Mapping Matrix.


This Layer's Weight Parameters accounts for most of all parameters in RoBERTa Model. (Each Layer accounts for 46%, 31%, 23% Parameters Respectively)


Following Table is 4bit and 2bit Weight Quantization Experiment Results.

**SST-2 Full Precision Accuracy : 94.6**

Layer/bit | FFN | EMB | QKV | FFN+EMB | EMB+QKV | FFN+QKV | All Together
-- | -- | -- | -- | -- | -- | -- | --
4bit | 94.8 | 94.7 | 94.7 | 94.2(-0.4) | 94.8 | 93.6(-1) | 93.7(-0.9)
2bit | 91.3(-3.3) | 94.6(0) | 92.7(-1.9) | 90.1(-4.5) | 92.3(-2.3) | 90.1(-4.5) | 89.6(-5)

- - -
### TODO
- Apply Knowledge Distillation 
- Try Other Tasks (Ex. SQUAD)
- Gradient Scale Exploration

Updated(5 SEP 2021)
