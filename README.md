# Hyper-CL: Conditioning Sentence Representations with Hypernetworks
<div align=center>
  <img alt="Static Badge" src="https://img.shields.io/badge/HyperCL-1.0-blue">
  <img alt="Github Created At" src="https://img.shields.io/github/created-at/HYU-NLP/Hyper-CL">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/HYU-NLP/Hyper-CL">
  <br>
</div>


#### Official Repository for "Hyper-CL: Conditioning Sentence Representations with Hypernetworks" [[Paper(arXiv)]](https://arxiv.org/abs/2403.09490))
##### Young Hyun Yoo, Jii Cha, Changhyeon Kim and Taeuk Kim. *Accepted to ACL2024 long paper*. 
---
### Table of Contents
- [C-STS](#c-sts)
  - [Data](#data)
  - [Train Hyper-CL](#train_HyperCL)
- [SimKGC](#SimKGC)
- [Citation](#citation)

## C-STS <a name="c-sts"></a>
In the following section, we introduce how to train aHyper-CL model by using our code.

### Data <a name="data"></a>
Download the C-STS dataset and locate the file at data/ (reference the [C-STS repository](https://github.com/princeton-nlp/c-sts/tree/main) for more details.)

### Train Hyper-CL <a name="train_HyperCL"></a>
#### Requirements
Run the following script, the requirements are the same as C-STS.
```bash
pip install -r requirements.txt
```

#### Training
We provide example training scripts for finetuning and evaluating the models in the paper. Go to C-STS/ and execute the following command
```bash
bash run_sts.sh
```

Following the arguments of [C-STS](https://github.com/princeton-nlp/c-sts/tree/main), we explain the additional arguments in following :
* `--objective`: (If you train Hyper-CL, you should use `triplet_cl_mse`)
* `--cl_temp`: Temperature for contrastive loss
* `--cl_in_batch_neg`: Add in-batch negative loss to main loss
* `--hypernet_scaler`: To set the value of K for low-rank implemented Hyper-CL _(i.e., hyper64-cl, hyper85-cl)_, we determine the divisor of the embedding size. For instance, in the base model, 'K=64' for hyper64-cl means the embedding size 768 is divided by 12. Thus, the hypernet_scaler is set to `12`.

* `--hypernet_dual`: Dual encoding that uses separate 2 encoders for sentences 1 and 2 and for the condition.

##### Hyperparameters
We use the following hyperparamters for training Hyper-CL:
|Emb.Model  | Learning rate (lr) | Weight decay (wd) | Temperature (temp)   |
|:--------------|:-----------:|:--------------:|:---------:|
| DiffCSE_base+hyper-cl   | 3e-5          | 0.1            | 1.5       |
| DiffCSE_base+hyper64-cl  | 1e-5 | 0.0 | 1.5 |
| SimCSE_base+hyper-cl | 3e-5 | 0.1 | 1.9 |
| SimCSE_base+hyper64-cl | 2e-5 | 0.1 | 1.7 |
| SimCSE_large+hyper-cl  | 2e-5 | 0.1 | 1.5 |
| SimCSE_large+hyper85-cl  | 1e-5 | 0.1 | 1.9 |

## SimKGC <a name="SimKGC"></a>

### Training

We provide example training scripts for finetuning and evaluating the models in the paper. Go to sim-kcg/ and execute the following command.
This code is based on [SimKCG](https://github.com/intfloat/SimKGC)

#### WN18RR dataset

##### Preprocessing

```bash
bash scripts/preprocess.sh WN18RR
```

##### Training

```bash
bash scripts/train_wn.sh
```

We explain the arguments in following:

- `--pretrained-model`: Backbone model checkpoint (`bert-base-uncased` or `bert-large-uncased`)
- `--encoding_type`: Encoding type (`bi_encoder` or `tri_encoder`)
- `--triencoder_head`: Triencoder head (`concat`, `hadamard` or `hypernet`)
- Refer to `config.py` for other arguments.

Evaluation for Perfomance and Inference Time

```bash
bash scripts/eval.sh ./checkpoint/WN18RR/model_best.mdl WN18RR
```
