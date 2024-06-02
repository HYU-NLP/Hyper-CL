import copy
import sys

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import PreTrainedModel, AutoModel, AutoConfig
from transformers.activations import ACT2FN

from logger_config import logger
from triplet_mask import construct_mask


def concat_features(*features):
    return torch.cat(features, dim=0) if features[0] is not None else None


class Similarity(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class InfoNCE(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.sim_fct = Similarity(temp)

    def forward(self, pos1, pos2, neg1, neg2):
        p_sim = self.sim_fct(pos1, pos2)
        n_sim = self.sim_fct(neg1, neg2)

        cos_sim_labels = torch.zeros(int(pos1.shape[0])).long().to(pos1.device)
        cos_sim = torch.stack([p_sim, n_sim], dim=1)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, cos_sim_labels)
        return loss


class Pooler(nn.Module):
    '''
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    '''

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ['cls', 'cls_before_pooler', 'avg', 'avg_top2',
                                    'avg_first_last'], 'unrecognized pooling type %s' % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            pooled_result = last_hidden[:, 0]
        elif self.pooler_type == 'avg':
            pooled_result = ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == 'avg_first_last':
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == 'avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError

        pooled_result = nn.functional.normalize(pooled_result, dim=-1)
        return pooled_result


class TriMLP(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1, scaler=0, ablation=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.scaler = scaler

        self.ablation = ablation

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.activation = nn.GELU()

        if self.ablation:
            self.big_mlp_1 = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size * int(hidden_size / scaler)),
            )

            self.big_mlp_2 = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size * int(hidden_size / scaler), hidden_size),
            )

            self.linear = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
            )

            return

        if scaler <= 0:
            raise NotImplementedError

        if scaler > 1:
            self.hypernet_proj = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size * int(hidden_size / scaler)),
            )

            self.hypernet_scale = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size * int(hidden_size / scaler)),
            )

        else:  # scaler == 1
            self.hypernet = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size * hidden_size),
            )

    def forward(self, c_emb):
        if self.scaler > 1:
            c_mlp1 = self.hypernet_proj(c_emb).view(c_emb.size(0), self.hidden_size, int(self.hidden_size / self.scaler))
            c_mlp2 = self.hypernet_scale(c_emb).view(c_emb.size(0), int(self.hidden_size / self.scaler), self.hidden_size)
            c_mlp = c_mlp1 @ c_mlp2
            return c_mlp

        else:  # scaler == 1
            c_mlp = self.hypernet(c_emb).view(c_emb.size(0), self.hidden_size, self.hidden_size)
            return c_mlp


class TriEncoderForClassification(PreTrainedModel):
    def __init__(self, args):
        config = AutoConfig.from_pretrained(args.pretrained_model, num_labels=1)
        config.update({
            "model_name_or_path": args.pretrained_model,
            "encoding_type": args.encoding_type,
            "objective": args.objective,
            "pooler_type": args.pooler_type,
            "transform": args.transform,
            "freeze_encoder": args.freeze_encoder,
            "triencoder_head": args.triencoder_head,
            "hypernet_scaler": args.hypernet_scaler,
            "hypernet_dual": args.hypernet_dual,
            "hypernet_ablation": args.hypernet_ablation,
            "cl_temp": args.cl_temp,
        })

        super().__init__(config)

        self.args = args
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.tensor_cache = {}
        self.cache_hits = 0
        self.cache_reqs = 0

        self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            add_pooling_layer=True,
        )
        if config.hypernet_dual:
            self.sub_backbone = copy.deepcopy(self.backbone)
        self.triencoder_head = config.triencoder_head
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        if config.transform:
            self.transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                ACT2FN[config.hidden_act],
            )
        else:
            self.transform = None
        self.condition_transform = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        if self.triencoder_head == 'concat':
            self.concat_transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                ACT2FN[config.hidden_act],
            )
        elif self.triencoder_head == 'hypernet':
            self.concat_transform = TriMLP(
                config.hidden_size,
                scaler=config.hypernet_scaler,
            )
        elif self.triencoder_head == 'hadamard':
            self.concat_transform = None
        else:
            raise RuntimeError
        self.pooler = Pooler(config.pooler_type)
        if config.pooler_type in {'avg_first_last', 'avg_top2'}:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False
        if config.num_labels == 1:
            self.reshape_function = lambda x: x.reshape(-1)
            if config.objective == 'mse':
                self.loss_fct_cls = nn.MSELoss
                self.loss_fct_kwargs = {}
            elif config.objective in {'triplet_cl_mse'}:
                self.loss_fct_cls = InfoNCE
                self.loss_fct_kwargs = {'temp': config.cl_temp}
            else:
                raise ValueError()
        else:
            self.reshape_function = lambda x: x.reshape(-1, config.num_labels)
            self.loss_fct_cls = nn.CrossEntropyLoss

        self.post_init()

    def restore_tensor(self, input_ids, attention_mask, token_type_ids):
        bsz = input_ids.shape[0]
        hit_ids = []
        miss_ids = []
        hit_tensors = []
        for i in range(bsz):
            self.cache_reqs += 1
            k = tuple(input_ids[i].tolist())
            if k in self.tensor_cache:
                self.cache_hits += 1
                v = self.tensor_cache[k]
                hit_ids.append(i)
                hit_tensors.append(v)
            else:
                miss_ids.append(i)

        return input_ids[miss_ids], attention_mask[miss_ids], token_type_ids[miss_ids], hit_ids, miss_ids, hit_tensors

    def save_tensor(self, input_ids, tensor):
        bsz = input_ids.shape[0]
        for i in range(bsz):
            k = tuple(input_ids[i].tolist())
            if k not in self.tensor_cache:
                self.tensor_cache[k] = tensor[i].clone()

    def reset_cache(self):
        self.tensor_cache = {}

    def reset_cache_status(self):
        self.cache_hits = 0
        self.cache_reqs = 0

    def print_cache_status(self):
        total_size = 0.0
        for k, v in self.tensor_cache.items():
            total_size += sys.getsizeof(v.storage())
        logger.info(f"Status of cache_hits / cache_reqs: {self.cache_hits} / {self.cache_reqs}")
        logger.info(f"Size of cache: {(total_size / 1024.0 / 1024.0):.2f}MB")

    def repack_tensor(self, bsz, hit_ids, miss_ids, feature, hit_tensors):
        j, max_j = 0, len(hit_ids)
        k = 0

        if bsz == max_j:
            return torch.stack(hit_tensors, dim=0)

        ary = []
        for i in range(bsz):
            if j < max_j and i == hit_ids[j]:
                ary.append(hit_tensors[j])
                j += 1
            else:  # i == head_miss_ids[k]
                ary.append(feature[k])
                k += 1

        return torch.stack(ary, dim=0)

    # Warning: "cache_feature" should only be used in torch.no_grad() mode
    def forward(
            self,

            head_token_ids=None,
            head_mask=None,
            head_token_type_ids=None,

            relation_token_ids=None,
            relation_mask=None,
            relation_token_type_ids=None,

            tail_token_ids=None,
            tail_mask=None,
            tail_token_type_ids=None,

            **kwargs,
    ):
        if "only_ent_embedding" in kwargs:  # No cache feature here
            tail_outputs = self.backbone(
                input_ids=tail_token_ids,
                attention_mask=tail_mask,
                token_type_ids=tail_token_type_ids,
                output_hidden_states=self.output_hidden_states,
            )

            tail_feature = self.pooler(tail_mask, tail_outputs)
            if self.transform is not None:
                tail_feature = self.transform(tail_feature)

            return {'ent_vectors': tail_feature.detach()}

        bsz = head_token_ids.shape[0]

        if "retrieve_cache" in kwargs:
            head_token_ids, head_mask, head_token_type_ids, head_hit_ids, head_miss_ids, head_hit_tensors = self.restore_tensor(head_token_ids, head_mask, head_token_type_ids)
            relation_token_ids, relation_mask, relation_token_type_ids, relation_hit_ids, relation_miss_ids, relation_hit_tensors = self.restore_tensor(relation_token_ids, relation_mask, relation_token_type_ids)
            tail_token_ids, tail_mask, tail_token_type_ids, tail_hit_ids, tail_miss_ids, tail_hit_tensors = self.restore_tensor(tail_token_ids, tail_mask, tail_token_type_ids)

        head_feature = []
        if len(head_token_ids) > 0:
            head_outputs = self.backbone(
                input_ids=head_token_ids,
                attention_mask=head_mask,
                token_type_ids=head_token_type_ids,
                output_hidden_states=self.output_hidden_states,
            )

            head_feature = self.pooler(head_mask, head_outputs)
            if self.transform is not None:
                head_feature = self.transform(head_feature)

        tail_feature = []
        if len(tail_token_ids) > 0:
            tail_outputs = self.backbone(
                input_ids=tail_token_ids,
                attention_mask=tail_mask,
                token_type_ids=tail_token_type_ids,
                output_hidden_states=self.output_hidden_states,
            )

            tail_feature = self.pooler(tail_mask, tail_outputs)
            if self.transform is not None:
                tail_feature = self.transform(tail_feature)

        relation_feature = []
        if len(relation_token_ids) > 0:
            if not self.config.hypernet_dual:
                relation_encoder = self.backbone
            else:
                relation_encoder = self.sub_backbone

            relation_outputs = relation_encoder(
                input_ids=relation_token_ids,
                attention_mask=relation_mask,
                token_type_ids=relation_token_type_ids,
                output_hidden_states=self.output_hidden_states,
            )

            relation_feature = self.pooler(relation_mask, relation_outputs)
            relation_feature = self.condition_transform(relation_feature)
            relation_feature = relation_feature.to(torch.float32)  # Why?
            if self.triencoder_head == 'hypernet':
                relation_feature = self.concat_transform(relation_feature)

        if "save_cache" in kwargs:
            self.save_tensor(head_token_ids, head_feature)
            self.save_tensor(tail_token_ids, tail_feature)
            self.save_tensor(relation_token_ids, relation_feature)

        if "retrieve_cache" in kwargs:
            head_feature = self.repack_tensor(bsz, head_hit_ids, head_miss_ids, head_feature, head_hit_tensors)
            relation_feature = self.repack_tensor(bsz, relation_hit_ids, relation_miss_ids, relation_feature, relation_hit_tensors)
            tail_feature = self.repack_tensor(bsz, tail_hit_ids, tail_miss_ids, tail_feature, tail_hit_tensors)

        if self.triencoder_head == 'hypernet':
            head_feature = head_feature.unsqueeze(1)
            hr_feature = (head_feature @ relation_feature).view(-1, self.config.hidden_size)
        elif self.triencoder_head == 'concat':
            hr_feature = torch.cat([head_feature, relation_feature], dim=-1)
            hr_feature = self.concat_transform(hr_feature)
        else:  # self.triencoder_head == 'hadamard':
            hr_feature = head_feature * relation_feature

        return {
            'hr_vector': hr_feature,
            'tail_vector': tail_feature,
            'head_vector': head_feature
        }

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits
