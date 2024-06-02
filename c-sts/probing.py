import json
import re

import numpy as np
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

COND_PART_SPAN_LEN = 2
DEP_WORD_TOP_K = 2


def jaccard_similarity(str1, str2):
    str1 = str1.lower()
    str2 = str2.lower()
    a = set(str1.split(" "))
    b = set(str2.split(" "))
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def f1_score(referenced_str, predicted_str):
    referenced_str = referenced_str.lower()
    predicted_str = predicted_str.lower()
    a = set(referenced_str.split(" "))
    b = set(predicted_str.split(" "))
    c = a.intersection(b)
    precision = float(len(c)) / len(b)
    recall = float(len(c)) / len(a)
    if precision + recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def is_continue(cond_part):
    if not cond_part:
        return True

    if len(cond_part.split(" ")) != COND_PART_SPAN_LEN:
        return True

    return False


def main():
    datasets = load_dataset("csv", data_files={"validation": "data/csts_train_labeling_200_ver3.csv"})
    datasets = datasets["validation"]

    def map_process(ori_batch):
        batch = {
            "sentence": [],
            "condition": [],
            "cond_part": [],
            "llm_predict_part": [],
            "llm_jacc_sim": [],
            "llm_f1_score": [],
        }

        for i in range(len(ori_batch['sentence1'])):
            if is_continue(ori_batch['cond_part1'][i]):
                continue

            batch["sentence"].append(ori_batch['sentence1'][i])
            batch["condition"].append(ori_batch['condition'][i])
            batch["cond_part"].append(ori_batch['cond_part1'][i])
            batch["llm_predict_part"].append(ori_batch['llm_predict_part1'][i])
            batch["llm_jacc_sim"].append(jaccard_similarity(ori_batch['cond_part1'][i], ori_batch['llm_predict_part1'][i]))
            batch["llm_f1_score"].append(f1_score(ori_batch['cond_part1'][i], ori_batch['llm_predict_part1'][i]))

        for i in range(len(ori_batch['sentence2'])):
            if is_continue(ori_batch['cond_part2'][i]):
                continue

            batch["sentence"].append(ori_batch['sentence2'][i])
            batch["condition"].append(ori_batch['condition'][i])
            batch["cond_part"].append(ori_batch['cond_part2'][i])
            batch["llm_predict_part"].append(ori_batch['llm_predict_part2'][i])
            batch["llm_jacc_sim"].append(jaccard_similarity(ori_batch['cond_part2'][i], ori_batch['llm_predict_part2'][i]))
            batch["llm_f1_score"].append(f1_score(ori_batch['cond_part2'][i], ori_batch['llm_predict_part2'][i]))

        return batch

    datasets = datasets.map(
        map_process,
        remove_columns=datasets.column_names,
        batched=True,
        load_from_cache_file=False
    )

    def avg(lst):
        return sum(lst) / len(lst)

    print(f"llm_jacc_sim: {avg(datasets['llm_jacc_sim'])}")
    print(f"llm_f1_score: {avg(datasets['llm_f1_score'])}")

    # ===

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    seq_max_length = 512

    # path = "/home/somebodil/workspace/private-projects/Sentence-Representation/c-sts/output/princeton-nlp__sup-simcse-roberta-base/enc_bi_encoder__lr_3e-5__wd_0.1__trans_False__obj_triplet_cl_mse__temp_1.5__s_42"
    # from load_trained_model import get_model_and_tokenizer
    # model, tokenizer = get_model_and_tokenizer(device, path)
    # file_name = "csts-finetuned-impact-span.json"
    #
    # def inference(_batch):
    #     return model(**_batch, output_hidden_states=True).hidden_states[0]  # cls

    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
    file_name = "sup-simcse-impact-span.json"

    def inference(b):
        return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)["last_hidden_state"][0][0]  # cls
        # return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)["pooler_output"][0]  # cls

    def map_matrix_process(ori_batch):
        batch = {
            "sentence": [],
            "condition": [],
            "cond_part": [],
            "llm_predict_part": [],
            "llm_jacc_sim": [],
            "impact_sentence": [],
            "masked_tokens": [],
        }

        for i, sentence in enumerate(ori_batch["sentence"]):
            words_ary = re.split("( |\.)", sentence)

            batch["sentence"].append(ori_batch["sentence"][i])
            batch["condition"].append(ori_batch["condition"][i])
            batch["cond_part"].append(ori_batch["cond_part"][i])
            batch["llm_predict_part"].append(ori_batch["llm_predict_part"][i])
            batch["llm_jacc_sim"].append(ori_batch["llm_jacc_sim"][i])
            batch["impact_sentence"].append(ori_batch["sentence"][i])
            batch["masked_tokens"].append("")

            for j in range(COND_PART_SPAN_LEN, 0, -1):
                for k, word in enumerate(words_ary):
                    if word == " ":
                        continue

                    tmp_ary = words_ary.copy()
                    masked_tokens = []
                    l = 0
                    s = 0

                    while l < j and k + l + s <= len(tmp_ary) - 1:
                        v = tmp_ary[k + l + s]
                        if v == " " or v == "":
                            s += 1
                            continue

                        masked_tokens.append(tmp_ary[k + l + s])
                        tmp_ary[k + l + s] = tokenizer.mask_token
                        l += 1

                    if k + l + s > len(tmp_ary) - 1:
                        continue

                    sentence = "".join(tmp_ary)

                    batch["sentence"].append(ori_batch["sentence"][i])
                    batch["condition"].append(ori_batch["condition"][i])
                    batch["cond_part"].append(ori_batch["cond_part"][i])
                    batch["llm_predict_part"].append(ori_batch["llm_predict_part"][i])
                    batch["llm_jacc_sim"].append(ori_batch["llm_jacc_sim"][i])
                    batch["impact_sentence"].append(sentence)
                    batch["masked_tokens"].append(" ".join(masked_tokens))

        return batch

    datasets = datasets.map(map_matrix_process, remove_columns=datasets.column_names, batched=True)

    def collate_fn(batch):
        batch = Dataset.from_list(batch)

        tokenized = tokenizer(
            batch['impact_sentence'],
            batch['condition'],
            padding=True,
            truncation=True,
            max_length=seq_max_length,
            return_tensors="pt"
        )

        return {
            **tokenized,
            "sentence": batch["sentence"],
            "impact_sentence": batch["impact_sentence"],
            "condition": batch["condition"],
            "cond_part": batch["cond_part"],
            "masked_tokens": batch["masked_tokens"],
        }

    dataloader = DataLoader(datasets, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # create empty json file
    with open(file_name, "w") as f:
        json.dump({}, f)

    with torch.no_grad():  # assuming batch_size = 1, original sentence always first
        list_results = []
        ori_cls = None
        before_sent_key = ""
        sent_key = ""
        for batch in tqdm(dataloader):
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)

            outputs_cls = inference(batch)  # cls
            original = True if batch["sentence"][0] == batch["impact_sentence"][0] else False

            if original:
                ori_cls = outputs_cls
                before_sent_key = sent_key
                sent_key = batch["sentence"][0] + "<SEP>" + batch["condition"][0]

                if before_sent_key:
                    # read json file
                    with open(file_name, "r") as f:
                        data = json.load(f)

                    list_results = sorted(list_results, key=lambda x: x["dist"], reverse=True)
                    del list_results[DEP_WORD_TOP_K:]
                    data[before_sent_key] = list_results

                    # dump list to json file
                    with open(file_name, "w") as f:
                        json.dump(data, f, indent=4)

                    list_results = []

            else:
                item = {
                    "dist": float(np.linalg.norm((ori_cls - outputs_cls).cpu())),
                    "impact_sentence": batch["impact_sentence"][0],
                    "cond_part": batch["cond_part"][0],
                    "masked_tokens": batch["masked_tokens"][0],
                    "model_jacc_sim": jaccard_similarity(batch["cond_part"][0], batch["masked_tokens"][0]),
                    "model_f1_score": f1_score(batch["cond_part"][0], batch["masked_tokens"][0]),
                }
                list_results.append(item)

    # ===

    # read json file
    with open(file_name, "r") as f:
        data = json.load(f)

    jacc_sim_acc = 0.0
    f1_score_acc = 0.0
    total_idx = 0.0
    for key in data:
        value = data[key]
        jacc_sim_acc += value[0]["model_jacc_sim"]
        f1_score_acc += value[0]["model_f1_score"]
        total_idx += 1.0

    print("jacc_sim_acc:" + str(jacc_sim_acc / total_idx))
    print("f1_score_acc:" + str(f1_score_acc / total_idx))


if __name__ == '__main__':
    main()
