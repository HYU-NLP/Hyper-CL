import logging
import random
from collections import defaultdict
from typing import Any, Callable, List, Optional

import datasets
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import Trainer
from transformers.trainer_utils import seed_worker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


class TripletBatchSampler(Sampler[List[int]]):
    r'''Samples elements from the dataset, grouping pairs of (positives, negatives) together.

    Args:
        batch_size (int)
        generator (Generator): Generator used in sampling
    '''
    batch_size: int
    generator: Optional[Callable[[int], int]]

    def __init__(
            self,
            batch_size: int,
            sentence1_key,
            sentence2_key,
            trainer: Any,
            generator=None
    ) -> None:
        self.batch_size = batch_size
        assert self.batch_size % 2 == 0, 'Batch size must be even for triplet loss'
        self.generator = generator
        self.trainer = trainer
        # self.trainer.train_dataset contains all the original data and feature names
        self.pairs = self._get_idx_pairs(
            self.trainer.train_dataset,
            sentence1_key,
            sentence2_key,
            self.trainer.model.config.encoding_type,
            str(self.trainer.tokenizer.sep_token_id),
        )

    def _get_idx_pairs(self, dataset: Dataset, sentence1_key: str, sentence2_key: str, encoding_type: str, sep_token_id: str) -> List[int]:
        '''Get the index order of the dataset, where each index is paired with a positive and negative index
        '''
        pairs = defaultdict(list)
        for ix, datum in enumerate(dataset):
            if encoding_type in {'tri_encoder'}:
                pairs[' '.join([str(e) for e in datum[sentence1_key]]) + '<SEP>' + ' '.join([str(e) for e in datum[sentence2_key]])].append(ix)
            elif encoding_type in {'bi_encoder'}:
                delim = "_" + sep_token_id + "_" + sep_token_id + "_"
                pairs['_'.join([str(e) for e in datum[sentence1_key]]).split(delim)[0] + '<SEP>' + '_'.join([str(e) for e in datum[sentence2_key]]).split(delim)[0]].append(ix)
            else:
                raise NotImplementedError

        pair_idxs = list(pairs.keys())
        drop_count = 0
        for pair_idx in pair_idxs:
            if len(pairs[pair_idx]) != 2:
                drop_count += len(pairs[pair_idx])
                pairs.pop(pair_idx)
        logger.warning('Dropping %d indices for missing pairs. Dataset has %d pairs total' % (drop_count, len(pair_idxs)))

        pairs = list(map(lambda x: sorted(pairs[x], key=lambda idx: -dataset[idx]['labels']), pairs.keys()))
        # negative because we want to sort in descending order (highest similarity first)
        for idx1, idx2 in pairs:
            if encoding_type in {'tri_encoder'}:
                if (dataset[idx1][sentence1_key] != dataset[idx2][sentence1_key]) or (dataset[idx1][sentence2_key] != dataset[idx2][sentence2_key]):
                    raise ValueError('Pairing of indices is incorrect, sentences do not match for pair %d and %d' % (idx1, idx2))
            elif encoding_type in {'bi_encoder'}:
                delim = "_" + sep_token_id + "_" + sep_token_id + "_"
                idx1_s1 = '_'.join([str(e) for e in dataset[idx1][sentence1_key]]).split(delim)[0]
                idx1_s2 = '_'.join([str(e) for e in dataset[idx1][sentence2_key]]).split(delim)[0]
                idx2_s1 = '_'.join([str(e) for e in dataset[idx2][sentence1_key]]).split(delim)[0]
                idx2_s2 = '_'.join([str(e) for e in dataset[idx2][sentence2_key]]).split(delim)[0]
                if (idx1_s1 != idx2_s1) or (idx1_s2 != idx2_s2):
                    raise ValueError('Pairing of indices is incorrect, sentences do not match for pair %d and %d' % (idx1, idx2))

            if (dataset[idx1]['labels'] < dataset[idx2]['labels']):
                raise ValueError('Pairing of indices is incorrect, similarity is not in descending order for pair %d and %d' % (idx1, idx2))

        return pairs

    def __iter__(self):
        '''Generate a batch of indices with tiled positive and negative indices
        '''
        random.shuffle(self.pairs)
        for i in range(0, len(self.pairs), self.batch_size // 2):
            batch = self.pairs[i:i + self.batch_size // 2]
            positives, negatives = zip(*batch)
            yield list(positives) + list(negatives)

    def __len__(self) -> int:
        return len(self.pairs) * 2 // self.batch_size


class TripletTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if not return_outputs:  # When Train
            return super().compute_loss(model, inputs, return_outputs)
        else:  # When Eval
            outputs = model(**inputs, is_inference=True)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        '''
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        '''
        if self.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description='training')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='training')
        train_sampler = TripletBatchSampler(
            self.args.train_batch_size,
            'input_ids',
            'input_ids_2',
            self,
        )
        return DataLoader(
            train_dataset,
            # batch_size=self._train_batch_size,
            batch_sampler=train_sampler,
            collate_fn=data_collator,
            # drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
