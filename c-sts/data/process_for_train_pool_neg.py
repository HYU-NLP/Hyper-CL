import faiss
import numpy as np
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader


def main(file_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seq_max_length = 512
    d = 768
    k = 20
    label_gt = -1
    model_name = "princeton-nlp/sup-simcse-roberta-base"

    datasets = load_dataset("csv", data_files={"train": f"{file_name}.csv"})
    train_dataset = datasets["train"]

    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def collate_fn(batch):
        batch = Dataset.from_list(batch)

        tokenized = tokenizer(
            batch['condition'],
            padding=True,
            truncation=True,
            max_length=seq_max_length,
            return_tensors="pt"
        )

        return {
            **tokenized
        }

    dataloader = DataLoader(train_dataset, batch_size=100, collate_fn=collate_fn)

    ary = []
    for batch in dataloader:
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)

        predict = model(**batch)
        ary.extend(predict.pooler_output.detach().tolist())

    ary = np.array(ary)
    index = faiss.IndexFlatL2(d)
    index.add(ary)
    distance, index = index.search(ary, k)

    def map_process(example, idx):
        knn_vectors = index[idx]
        knn_vectors = knn_vectors.tolist()

        try:
            knn_vectors.remove(idx)
        except ValueError:
            pass

        failed = True
        for i in knn_vectors:
            if train_dataset["label"][i] >= label_gt:
                example = {
                    **example,
                    "sim_con_sim_sent_1": train_dataset["sentence1"][i],
                    "sim_con_sim_sent_2": train_dataset["sentence2"][i],
                    "sim_con": train_dataset["condition"][i],
                }
                failed = False
                break

        if failed:
            raise RuntimeError

        return example

    train_dataset = train_dataset.map(map_process, with_indices=True)
    train_dataset.to_csv(f"{file_name}_v2.csv")


# def test():
#     import numpy as np
#     d = 64  # dimension
#     nb = 100000  # database size
#     nq = 10000  # nb of queries
#     np.random.seed(1234)  # make reproducible
#     xb = np.random.random((nb, d)).astype('float32')
#     xb[:, 0] += np.arange(nb) / 1000.
#     xq = np.random.random((nq, d)).astype('float32')
#     xq[:, 0] += np.arange(nq) / 1000.
#
#     import faiss  # make faiss available
#     index = faiss.IndexFlatL2(d)  # build the index
#     print(index.is_trained)
#     index.add(xb)  # add vectors to the index
#     print(index.ntotal)
#
#     k = 4  # we want to see 4 nearest neighbors
#     D, I = index.search(xb[:5], k)  # sanity check
#     print(I)
#     print(D)
#     D, I = index.search(xq, k)  # actual search
#     print(I[:5])  # neighbors of the 5 first queries
#     print(I[-5:])  # neighbors of the 5 last queries


if __name__ == "__main__":
    main("csts_train")
    main("csts_validation")
    # test()
