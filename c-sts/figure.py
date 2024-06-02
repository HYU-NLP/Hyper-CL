import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

from load_trained_model import get_model_and_tokenizer
from run_sts import ModelArguments
from utils.sts.dataset_preprocessing import get_preprocessing_function


def visualize(before_mlp, after_mlp, before_labels, after_labels, condition_list, perplexity):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    before_mlp = tsne.fit_transform(np.array(before_mlp))
    after_mlp_n_cond = tsne.fit_transform(np.array(after_mlp))

    color = ['r', 'g', 'b']
    cond_sz = 20
    df_before = pd.DataFrame(before_mlp)
    df_after = pd.DataFrame(after_mlp_n_cond)
    sns.scatterplot(x=df_before.iloc[:, 0], y=df_before.iloc[:, 1], data=df_before, hue=condition_list)  # , palette='coolwarm'
    # plt.scatter(centers[:,0], centers[:,1], c='black', alpha=0.8, s=150)
    plt.show()
    plt.savefig('visualize/cluster/fig1_' + str(perplexity) + '.png', dpi=300)

    sns.scatterplot(x=df_after.iloc[:, 0], y=df_after.iloc[:, 1], data=df_after, hue=condition_list)  # , palette='coolwarm'
    # plt.scatter(centers[:,0], centers[:,1], c='black', alpha=0.8, s=150)
    plt.show()
    plt.savefig('visualize/cluster/fig2_' + str(perplexity) + '.png', dpi=300)


def inference(inputs_sent_str_, inputs_cond_str_, before_mlp_, after_mlp_):
    inputs_ = tokenizer([*inputs_sent_str_, *inputs_cond_str_], padding=True, truncation=True, return_tensors="pt").to(device)

    outputs_ = model(
        input_ids=inputs_["input_ids"][0].unsqueeze(0),
        attention_mask=inputs_["attention_mask"][0].unsqueeze(0),
        input_ids_3=inputs_["input_ids"][1].unsqueeze(0),
        attention_mask_3=inputs_["attention_mask"][1].unsqueeze(0),
        output_hidden_states=True
    )

    before_mlp_.append(outputs_.hidden_states[0].squeeze().tolist())
    after_mlp_.append(outputs_.hidden_states[1].squeeze().tolist())


parser = HfArgumentParser(ModelArguments)
model_args, = parser.parse_args_into_dataclasses()
device = "cuda:0"
path = "/home/somebodil/workspace/private-projects/Sentence-Representation/c-sts/output/princeton-nlp__sup-simcse-roberta-base/enc_tri_encoder__lr_2e-5__wd_0.1__trans_False__obj_triplet_cl_mse__temp_1.5__tri_hypernet__hn_s_12__s_42"

raw_datasets = load_dataset("csv", data_files={"validation": "data/csts_validation_ex.csv"}, split="validation")
model, tokenizer = get_model_and_tokenizer(path, device)

sentence1_key, sentence2_key, condition_key, similarity_key = (
    "sentence1",
    "sentence2",
    "condition",
    "label",
)

preprocess_function = get_preprocessing_function(
    tokenizer,
    sentence1_key,
    sentence2_key,
    condition_key,
    similarity_key,
    False,
    None,
    model_args,
)

dataLoader = DataLoader(raw_datasets, batch_size=1)

before_mlp = []
after_mlp = []
condition_list = []
for batch in dataLoader:
    condition_list.append(batch["condition"])
    inference(batch["sentence1"], batch["condition"], before_mlp, after_mlp)
    inference(batch["sentence2"], batch["condition"], before_mlp, after_mlp)

visualize(before_mlp, after_mlp, None, None, condition_list, 1)