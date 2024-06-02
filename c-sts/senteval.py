import sys

import torch
from prettytable import PrettyTable
from transformers import RobertaTokenizer

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def main():
    device = "cuda:0"

    from transformers import AutoModel
    path = "voidism/diffcse-roberta-base-sts"
    tokenizer = RobertaTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(device)

    def inf_model(batch):
        outputs = model(**batch)
        return outputs.last_hidden_state[:, 0].cpu()

    # from utils.sts.modeling_encoders import TriEncoderForClassification
    #
    # parser = HfArgumentParser(ModelArguments)
    # model_args, = parser.parse_args_into_dataclasses()
    #
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     num_labels=1,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    #
    # config.update({
    #     "use_auth_token": model_args.use_auth_token,
    #     "model_revision": model_args.model_revision,
    #     "cache_dir": model_args.cache_dir,
    #     "model_name_or_path": model_args.model_name_or_path,
    #     "encoding_type": model_args.encoding_type,
    #     "objective": model_args.objective,
    #     "pooler_type": model_args.pooler_type,
    #     "transform": model_args.transform,
    #     "triencoder_head": model_args.triencoder_head,
    #     "hypernet_scaler": model_args.hypernet_scaler,
    #     "hypernet_dual": model_args.hypernet_dual,
    #     "cl_temp": model_args.cl_temp,
    # })
    #
    # path = "/home/somebodil/workspace/private-projects/Sentence-Representation/c-sts/output_hypernet_freeze_dual_full/voidism__diffcse-roberta-base-sts/enc_tri_encoder__lr_2e-5__wd_0.1__trans_False__obj_triplet_cl_mse__temp_1.5__tri_hypernet__hn_s_12__s_53"
    # tokenizer = RobertaTokenizer.from_pretrained(path)
    # model = TriEncoderForClassification.from_pretrained(path, config=config).to(device)
    #
    # def inf_model(batch):
    #     outputs = model(**batch)
    #     # return outputs.hidden_states[0].cpu()  # models backbones + "pretrained pooler layer" cls token
    #     return outputs.hidden_states[1].cpu()  # models backbones "raw" cls token

    # ============================================================================================================

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)

        # Get raw embeddings
        with torch.no_grad():
            return inf_model(batch)

    args = {}
    args["tasks"] = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}}

    results = {}
    for task in args["tasks"]:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    task_names = []
    scores = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
            else:
                scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)

    task_names = []
    scores = []
    for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
        task_names.append(task)
        if task in results:
            scores.append("%.2f" % (results[task]['acc']))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)


if __name__ == "__main__":
    main()
