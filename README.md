# BERT Quantization Project

## Introduction
This is Minsoo's Private Repository for RoBERTa Quantization Project.

This Repository is based on Fairseq based I-BERT base Branch Code.
Github Link : https://github.com/kssteven418/I-BERT 
(Please use "base" branch, Refer main branch README base section)


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

**RTE Full Precision Accuracy : 81.2**

Weight/bit | FFN | EMB | QKV | FFN+EMB | EMB+QKV | FFN+QKV | All Together
-- | -- | -- | -- | -- | -- | -- | --
4bit | 80.9 | 80.9 | 78.7 | 80.9 | 79.8 | 75.8 | 75.5(-5.7)
2bit | 62.1(-19.1) | 80.5(-0.7) | 54.9(-26.3) | 59.2(-22) | 54.5(-26.7) | 52.7(-28.5) | 53.1(-28.1)


## KD Effects (0913 Updated)

We added TernaryBERT paper's Knowledge Distillation aware Training Method. (Link : https://arxiv.org/abs/2009.12812). 
Teacher model is task specific fine tuned full precision model. And It distills knowledge to student model(Quantized Model) in three different part.  
Transformer layer outputs(Hidden States), Attention Scores(Attention Weights), Prediction Output(Logit Vectors)

**2bit Weight Quantization KD Effects Table**

Task | FP | -KD | KD | KD Effect | Diff with FP
-- | -- | -- | -- | -- | --
SST-2 | 94.6 | 89.4 | 89.9 | 0.5 | 4.7
CoLA | 61.56 | 13.7 | 52.9 | 39.2 | 8.6
RTE | 81.2 | 58.8 | 64.6 | 5.8 | 16.6

**2bit Quantization KD Effects (Different Part Applied) TODO**

  | -KD | KD(Trm+Logits) | Logits | Trm | Output | Atts
-- | -- | -- | -- | -- | -- | --
CoLA | 28.52 | 48.62(+20.1) | 39.85(+11.33) | 58.05(+29.53) | 45.19(+16.67) | 24.42(-4.1)
RTE | 62.1 | 80.4(+18.3) | 69.1(+7) | 74.4(+12.3) | 72.9(+10.8) | 73.6(+11.5)
SST-2 | 90.4 | 91.2(+0.8) | 91.2(+0.8) | 91.4(+1) | 91.4(+1) | 91.2(+0.8)

**FFN 2bit Quantization KD Effects (Different Part Applied)**

  | -KD | KD(Trm+Logits) | Logits | Trm | Output | Atts
-- | -- | -- | -- | -- | -- | --
CoLA | 28.52 | 48.62(+20.1) | 39.85(+11.33) | 58.05(+29.53) | 45.19(+16.67) | 24.42(-4.1)
RTE | 62.1 | 80.4(+18.3) | 69.1(+7) | 74.4(+12.3) | 72.9(+10.8) | 73.6(+11.5)
SST-2 | 90.4 | 91.2(+0.8) | 91.2(+0.8) | 91.4(+1) | 91.4(+1) | 91.2(+0.8)


### TODO
- Task Specific Quantization Setting Optimization
- KD Effects Sweep Experiments

Updated(19 SEP 2021)
