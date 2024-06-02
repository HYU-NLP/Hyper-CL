import copy
import logging

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import cosine_similarity
from transformers import PreTrainedModel, AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


def concat_features(*features):
    return torch.cat(features, dim=0) if features[0] is not None else None


class QuadrupletLoss:
    def __init__(self, distance_function, margin=1.0):
        'A cosine distance margin quadruplet loss'
        self.margin = margin
        self.distance_function = distance_function

    def __call__(self, pos1, pos2, neg1, neg2):
        dist_pos = self.distance_function(pos1, pos2)
        dist_neg = self.distance_function(neg1, neg2)
        loss = torch.clamp_min(self.margin + dist_pos - dist_neg, 0)
        return loss.mean()


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


class MatNBiasOp(nn.Module):
    def __init__(self, dim, requires_grad):
        super().__init__()

        self.weight = nn.Parameter(torch.randn([dim, dim]), requires_grad=requires_grad)
        self.bias = nn.Parameter(torch.randn([1, dim]), requires_grad=requires_grad)

    def forward(self, x_vec, scale_vec):
        x_vec = x_vec.unsqueeze(1)

        # identity = x_vec
        x_vec = x_vec @ (scale_vec.unsqueeze(1) * self.weight) + self.bias
        # x_vec = identity + nn.functional.tanh(x_vec)
        # x_vec = identity + x_vec * 0.1

        x_vec = x_vec.squeeze()
        return x_vec


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

    def forward(self, x):
        s_emb, c_emb = x[:, :self.hidden_size], x[:, self.hidden_size:]

        if self.ablation:
            c_emb = self.big_mlp_1(c_emb)
            c_emb = self.big_mlp_2(c_emb)
            # s_emb = self.linear(torch.cat([c_emb, s_emb], dim=-1))
            s_emb = c_emb * s_emb
            return s_emb

        if self.scaler > 1:
            c_mlp1 = self.hypernet_proj(c_emb).view(x.size(0), self.hidden_size, int(self.hidden_size / self.scaler))
            c_mlp2 = self.hypernet_scale(c_emb).view(x.size(0), int(self.hidden_size / self.scaler), self.hidden_size)
            c_mlp = c_mlp1 @ c_mlp2

        else:  # scaler == 1
            c_mlp = self.hypernet(c_emb).view(x.size(0), self.hidden_size, self.hidden_size)

        s_emb = s_emb.unsqueeze(1)
        s_emb = (s_emb @ c_mlp).view(-1, self.hidden_size)
        return s_emb #, c_mlp


# Pooler class. Copied and adapted from SimCSE code
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
            return last_hidden[:, 0]
        elif self.pooler_type == 'avg':
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == 'avg_first_last':
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == 'avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class CrossEncoderForClassification(PreTrainedModel):
    'Encoder model with backbone and classification head.'

    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            add_pooling_layer=False,
        ).base_model
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
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.pooler = Pooler(config.pooler_type)
        if config.pooler_type in {'avg_first_last', 'avg_top2'}:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False
        if config.num_labels == 1:
            self.reshape_function = lambda x: x.reshape(-1)
            if config.objective == 'mse':
                self.loss_fct_cls = nn.MSELoss
            elif config.objective in {'triplet', 'triplet_mse'}:
                raise NotImplementedError('Triplet loss is not implemented for CrossEncoderForClassification')
            else:
                raise ValueError(
                    f'Only regression and triplet objectives are supported for CrossEncoderForClassification with num_labels=1. Got {config.objective}.')
        else:
            assert config.objective == 'classification'
            self.reshape_function = lambda x: x.reshape(-1, config.num_labels)
            self.loss_fct_cls = nn.CrossEntropyLoss
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            **kwargs,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
        )
        features = self.pooler(attention_mask, outputs)
        if self.transform is not None:
            features = self.transform(features)
        logits = self.classifier(features)
        reshaped_logits = self.reshape_function(logits)
        loss = None
        if labels is not None:
            loss = self.loss_fct_cls()(reshaped_logits, labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BiEncoderForClassification(PreTrainedModel):
    '''Encoder model with backbone and classification head.'''

    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            add_pooling_layer=False,
        ).base_model
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
        self.pooler = Pooler(config.pooler_type)
        if config.pooler_type in {'avg_first_last', 'avg_top2'}:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False
        if config.objective == 'mse':
            self.loss_fct_cls = nn.MSELoss
            self.loss_fct_kwargs = {}
        elif config.objective in {'triplet', 'triplet_mse'}:
            self.loss_fct_cls = QuadrupletLoss
            self.loss_fct_kwargs = {'distance_function': lambda x, y: 1.0 - cosine_similarity(x, y)}
        elif config.objective in {'triplet_cl_mse'}:
            self.loss_fct_cls = InfoNCE
            self.loss_fct_kwargs = {'temp': config.cl_temp}
        else:
            raise ValueError()
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            input_ids_2=None,
            attention_mask_2=None,
            token_type_ids_2=None,
            position_ids_2=None,
            head_mask_2=None,
            inputs_embeds_2=None,
            labels=None,
            **kwargs,
    ):
        if input_ids is not None and input_ids_2 is not None:
            bsz = input_ids.shape[0]
            input_ids = concat_features(input_ids, input_ids_2)
            attention_mask = concat_features(attention_mask, attention_mask_2)
            token_type_ids = concat_features(token_type_ids, token_type_ids_2)
            position_ids = concat_features(position_ids, position_ids_2)
            head_mask = concat_features(head_mask, head_mask_2)
            inputs_embeds = concat_features(inputs_embeds, inputs_embeds_2)
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=self.output_hidden_states,
            )
            features = self.pooler(attention_mask, outputs)
            if self.transform is not None:
                features = self.transform(features)
            features_1, features_2 = torch.split(features, bsz, dim=0)  # [sentence1, condtion], [sentence2, condition]
            loss = None
            if self.config.objective in {'triplet', 'triplet_mse', 'triplet_cl_mse'}:
                positives1, negatives1 = torch.split(features_1, bsz // 2, dim=0)
                positives2, negatives2 = torch.split(features_2, bsz // 2, dim=0)
                if labels is not None:
                    loss = self.loss_fct_cls(**self.loss_fct_kwargs)(positives1, positives2, negatives1, negatives2)
                logits = cosine_similarity(features_1, features_2, dim=1)
                if self.config.objective in {'triplet_mse', 'triplet_cl_mse'} and labels is not None:
                    loss += nn.MSELoss()(logits, labels)
                else:
                    logits = logits.detach()
            else:
                logits = cosine_similarity(features_1, features_2, dim=1)
                if labels is not None:
                    loss = self.loss_fct_cls(**self.loss_fct_kwargs)(logits, labels)
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
            )
        elif input_ids is not None and input_ids_2 is None:  # only for embeddings
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=self.output_hidden_states,
            )
            features = self.pooler(attention_mask, outputs)
            if self.transform is not None:
                features = self.transform(features)
            return SequenceClassifierOutput(
                hidden_states=features,
            )


class TriEncoderForClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            add_pooling_layer=True,
        )
        # self.debug = copy.deepcopy(self.backbone)
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
                ablation=config.hypernet_ablation,
            )
        elif self.triencoder_head == 'hadamard':
            self.concat_transform = None
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
            elif config.objective in {'triplet', 'triplet_mse'}:
                self.loss_fct_cls = QuadrupletLoss
                self.loss_fct_kwargs = {'distance_function': lambda x, y: 1.0 - cosine_similarity(x, y)}
            elif config.objective in {'triplet_cl_mse'}:
                self.loss_fct_cls = InfoNCE
                self.loss_fct_kwargs = {'temp': config.cl_temp}
            else:
                raise ValueError()
        else:
            self.reshape_function = lambda x: x.reshape(-1, config.num_labels)
            self.loss_fct_cls = nn.CrossEntropyLoss

        self.post_init()

    def proj_feature(self, features_1, features_2, features_3):
        if self.triencoder_head == 'concat' or self.triencoder_head == 'hypernet':
            features_1_proj = torch.cat([features_1, features_3], dim=-1)
            features_2_proj = torch.cat([features_2, features_3], dim=-1)
            features_1_proj = self.concat_transform(features_1_proj)
            features_2_proj = self.concat_transform(features_2_proj)
        elif self.triencoder_head == 'hadamard':
            features_1_proj = features_1 * features_3
            features_2_proj = features_2 * features_3
        else:  # self.triencoder_head == 'none'
            features_1_proj = features_1
            features_2_proj = features_2
        return features_1_proj, features_2_proj

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            input_ids_2=None,
            attention_mask_2=None,
            token_type_ids_2=None,
            position_ids_2=None,
            head_mask_2=None,
            inputs_embeds_2=None,
            input_ids_3=None,
            attention_mask_3=None,
            token_type_ids_3=None,
            position_ids_3=None,
            head_mask_3=None,
            inputs_embeds_3=None,
            input_ids_4=None,
            attention_mask_4=None,
            token_type_ids_4=None,
            input_ids_5=None,
            attention_mask_5=None,
            token_type_ids_5=None,
            labels=None,
            output_hidden_states=None,
            **kwargs,
    ):
        device = input_ids.device
        bsz = input_ids.shape[0]

        if input_ids is not None and input_ids_2 is not None and input_ids_3 is not None:  # For training
            if self.config.hypernet_dual:
                input_ids = concat_features(input_ids, input_ids_2)
                attention_mask = concat_features(attention_mask, attention_mask_2)
                token_type_ids = concat_features(token_type_ids, token_type_ids_2)
                position_ids = concat_features(position_ids, position_ids_2)
                head_mask = concat_features(head_mask, head_mask_2)
                inputs_embeds = concat_features(inputs_embeds, inputs_embeds_2)
            else:
                input_ids = concat_features(input_ids, input_ids_2, input_ids_3)
                attention_mask = concat_features(attention_mask, attention_mask_2, attention_mask_3)
                token_type_ids = concat_features(token_type_ids, token_type_ids_2, token_type_ids_3)
                position_ids = concat_features(position_ids, position_ids_2, position_ids_3)
                head_mask = concat_features(head_mask, head_mask_2, head_mask_3)
                inputs_embeds = concat_features(inputs_embeds, inputs_embeds_2, inputs_embeds_3)

            with torch.set_grad_enabled(not self.config.freeze_encoder):
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=self.output_hidden_states,
                )

            features = self.pooler(attention_mask, outputs)
            if self.config.hypernet_dual:
                features_1, features_2 = torch.split(features, bsz, dim=0)
                sub_outputs = self.sub_backbone(
                    input_ids=input_ids_3,
                    attention_mask=attention_mask_3,
                    token_type_ids=token_type_ids_3,
                    position_ids=position_ids_3,
                    head_mask=head_mask_3,
                    inputs_embeds=inputs_embeds_3,
                    output_hidden_states=self.output_hidden_states,
                )
                features_3 = self.pooler(attention_mask_3, sub_outputs)  # overwrite condition features

            else:
                features_1, features_2, features_3 = torch.split(features, bsz, dim=0)

            features_3 = self.condition_transform(features_3)

            if self.transform is not None:
                features_1 = self.transform(features_1)
                features_2 = self.transform(features_2)

            features_1_proj, features_2_proj = self.proj_feature(features_1, features_2, features_3)

            loss = torch.tensor(0.0).to(device)
            if self.config.objective in {'triplet', 'triplet_mse', 'triplet_cl_mse'}:
                logits = cosine_similarity(features_1_proj, features_2_proj, dim=1)
                if labels is not None:
                    positive_idxs = torch.arange(0, features_1_proj.shape[0] // 2)
                    negative_idxs = torch.arange(features_1_proj.shape[0] // 2, features_1_proj.shape[0])
                    pos1 = features_1_proj[positive_idxs]
                    pos2 = features_2_proj[positive_idxs]
                    neg1 = features_1_proj[negative_idxs]
                    neg2 = features_2_proj[negative_idxs]

                    if self.config.cl_in_batch_neg:
                        loss_fct = nn.CrossEntropyLoss()
                        cos_sim_labels = torch.zeros(int(pos1.shape[0])).long().to(pos1.device)

                        sim_fct = Similarity(self.config.cl_temp)

                        sim_pos = sim_fct(pos1, pos2)
                        sim_neg = sim_fct(neg1, neg2)

                        s1 = features_1[positive_idxs]  # same as features_1[negative_idxs]
                        s2 = features_2[positive_idxs]  # same as features_2[negative_idxs]

                        # sim_neg_add1 = sim_fct(pos1, s1)
                        # sim_neg_add2 = sim_fct(pos2, s2)
                        sim_neg_add3 = sim_fct(pos1, neg1) * 0.5
                        sim_neg_add4 = sim_fct(pos2, neg2) * 0.5

                        cos_sim = torch.stack((sim_pos, sim_neg, sim_neg_add3, sim_neg_add4), dim=-1)
                        loss += loss_fct(cos_sim, cos_sim_labels)

                    else:
                        loss += self.loss_fct_cls(**self.loss_fct_kwargs)(pos1, pos2, neg1, neg2)

                    if self.config.objective in {'triplet_mse', 'triplet_cl_mse'}:
                        loss += nn.MSELoss()(logits, labels)
            else:
                logits = cosine_similarity(features_1_proj, features_2_proj, dim=1)
                if labels is not None:
                    loss += self.loss_fct_cls(**self.loss_fct_kwargs)(logits, labels)

            hidden_states = None
            if output_hidden_states:
                hidden_states = [
                    outputs.pooler_output,
                    features_1.detach(), features_1_proj.detach(),
                    features_2.detach(), features_2_proj.detach(),
                ]

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits.detach(),
                hidden_states=hidden_states,
            )

        else:  # For only embedding

            if not output_hidden_states:
                raise ValueError

            # Input ==
            if input_ids is not None and input_ids_3 is not None:
                input_ids = concat_features(input_ids, input_ids_3)
                attention_mask = concat_features(attention_mask, attention_mask_3)
                token_type_ids = concat_features(token_type_ids, token_type_ids_3)
                position_ids = concat_features(position_ids, position_ids_3)
                head_mask = concat_features(head_mask, head_mask_3)
                inputs_embeds = concat_features(inputs_embeds, inputs_embeds_3)
            elif input_ids is not None and input_ids_3 is None:
                pass  # Only use input_ids
            else:
                raise NotImplementedError

            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=self.output_hidden_states,
            )

            if input_ids is not None and input_ids_3 is not None:
                features = self.pooler(attention_mask, outputs)
                features_1, features_3 = torch.split(features, bsz, dim=0)
                features_3 = self.condition_transform(features_3)

                if self.transform is not None:
                    features_1 = self.transform(features_1)

                if self.triencoder_head == 'concat' or self.triencoder_head == 'hypernet':
                    features_1_proj = torch.cat([features_1, features_3], dim=-1)
                    features_1_proj = self.concat_transform(features_1_proj)
                elif self.triencoder_head == 'hadamard':
                    features_1_proj = features_1 * features_3
                else:  # self.triencoder_head == 'none'
                    features_1_proj = features_1

            else:  # if input_ids is not None and input_ids_3 is None
                features_1 = self.pooler(attention_mask, outputs)

            if input_ids is not None and input_ids_3 is not None:
                hidden_states = [outputs.pooler_output, features_1.detach(), features_1_proj.detach()]
            else:  # if input_ids is not None and input_ids_3 is None
                hidden_states = [outputs.pooler_output, features_1.detach()]

            return SequenceClassifierOutput(
                hidden_states=hidden_states
            )
