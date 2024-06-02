"""
Adapted code from HuggingFace run_glue.py

Author: Ameet Deshpande, Carlos E. Jimenez
"""
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, Union

import datasets
import numpy as np
import transformers
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PrinterCallback,
    Trainer,
)
from transformers import TrainingArguments as HFTrainingArguments
from transformers import default_data_collator, set_seed
from transformers.trainer_utils import get_last_checkpoint, IntervalStrategy, SchedulerType

from utils.progress_logger import LogCallback
from utils.sts.dataset_preprocessing import get_preprocessing_function
from utils.sts.modeling_utils import DataCollatorWithPadding, get_model
from utils.sts.triplet_trainer import TripletTrainer
from load_trained_model import get_model_and_tokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# Default params == run_sts.sh default params

@dataclass
class TrainingArguments(HFTrainingArguments):
    output_dir: str = field(default="output/0")
    overwrite_output_dir: bool = field(default=True)

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=True, metadata={"help": "Whether to run predictions on the test set."})

    evaluation_strategy: Union[IntervalStrategy, str] = field(default="epoch")
    per_device_train_batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(default=1)

    learning_rate: float = field(default=3e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.1, metadata={"help": "Weight decay for AdamW if we apply some."})
    max_grad_norm: float = field(default=0.0, metadata={"help": "Max gradient norm."})
    num_train_epochs: float = field(default=5, metadata={"help": "Total number of training epochs to perform."})
    lr_scheduler_type: Union[SchedulerType, str] = field(default="linear")
    warmup_ratio: float = field(default=0.1,
                                metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    log_level: Optional[str] = field(default="info")
    disable_tqdm: Optional[bool] = field(default=False)
    save_strategy: Union[IntervalStrategy, str] = field(default="epoch")
    save_total_limit: Optional[int] = field(default=1)
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    data_seed: Optional[int] = field(default=42, metadata={"help": "Random seed to be used with data samplers."})
    fp16: bool = field(default=True)
    log_time_interval: int = field(default=15)

    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default='eval_spearmanr')


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=True,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default="data/csts_train.csv",
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default="data/csts_validation.csv",
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default="data/csts_test.csv",
        metadata={"help": "A csv or a json file containing the test data."},
    )
    # Dataset specific arguments
    max_similarity: Optional[float] = field(
        default=None, metadata={"help": "Maximum similarity score."}
    )
    min_similarity: Optional[float] = field(
        default=None, metadata={"help": "Minimum similarity score."}
    )
    condition_only: Optional[bool] = field(
        default=False, metadata={"help": "Only use condition column."}
    )
    sentences_only: Optional[bool] = field(
        default=False, metadata={"help": "Only use sentences column."}
    )

    def __post_init__(self):
        validation_extension = self.validation_file.split(".")[-1]
        if self.train_file is not None:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            assert (
                    train_extension == validation_extension
            ), "`train_file` and `validation_file` should have the same extension."
        if self.test_file is not None:
            test_extension = self.test_file.split(".")[-1]
            assert test_extension in [
                "csv",
                "json",
            ], "`test_file` should be a csv or a json file."
            assert (
                    test_extension == validation_extension
            ), "`test_file` and `validation_file` should have the same extension."

    def __str__(self):
        self_as_dict = asdict(self)
        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="princeton-nlp/sup-simcse-roberta-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    objective: Optional[str] = field(
        default="triplet_cl_mse",
        metadata={
            "help": "Objective function for training. Options:\
            1) mse: Regression task (uses MSELoss).\
            2) triplet: Regression task (uses QuadrupletLoss).\
            3) triplet_mse: Regression task uses QuadrupletLoss with MSE loss.\
            4) triplet_cl_mse: Regression task uses InfoNCE with MSE loss."  # Only in Tri-encoder model
        },
    )
    cl_temp: float = field(default=1.5, metadata={"help": "Temperature for contrastive loss."})
    cl_in_batch_neg: bool = field(default=False)  # FIXME

    # What type of modeling
    encoding_type: Optional[str] = field(
        default="tri_encoder",
        metadata={
            "help": "What kind of model to choose. Options:\
            1) cross_encoder: Full encoder model.\
            2) bi_encoder: Bi-encoder model.\
            3) tri_encoder: Tri-encoder model."
        },
    )
    # Pooler for bi-encoder
    pooler_type: Optional[str] = field(
        default="cls",
        metadata={
            "help": "Pooler type: Options:\
            1) cls: Use [CLS] token.\
            2) avg: Mean pooling."
        },
    )
    freeze_encoder: Optional[bool] = field(
        default=False, metadata={"help": "Freeze encoder weights."}
    )
    transform: Optional[bool] = field(
        default=False,
        metadata={"help": "Use a linear transformation on the encoder output"},
    )
    triencoder_head: Optional[str] = field(
        default="hypernet",
        metadata={
            "help": "Tri-encoder head type: Options:\
            1) hadamard.\
            2) hypernet.\
            3) concat."
        },
    )
    hypernet_scaler: int = field(default=12)
    hypernet_dual: bool = field(default=False)
    hypernet_ablation: bool = field(default=False)  # FIXME

    def __str__(self):
        self_as_dict = asdict(self)
        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]),
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    training_args.log_level = "info"
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if model_args.objective in {"triplet", "triplet_mse", "triplet_cl_mse"}:
        training_args.dataloader_drop_last = True
        training_args.per_device_eval_batch_size = 2
    logger.info("Training/evaluation parameters %s" % training_args)
    logger.info("Model args %s" % model_args)
    logger.info("Data args %s" % data_args)
    last_checkpoint = None
    if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
                last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    set_seed(training_args.seed)
    data_files = {"validation": data_args.validation_file}
    if training_args.do_train:
        data_files["train"] = data_args.train_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    elif training_args.do_predict:
        raise ValueError("test_file argument is missing. required for do_predict.")
    for key, name in data_files.items():
        logger.info(f"load a local file for {key}: {name}")
    if data_args.validation_file.endswith(".csv") or data_args.validation_file.endswith(".tsv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.validation_file.endswith(".json"):
        # Loading a dataset from local json files
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise ValueError("validation_file should be a csv or a json file.")
    labels = set()
    for key in set(raw_datasets.keys()) - {"test"}:
        labels.update(raw_datasets[key]["label"])
    if data_args.min_similarity is None:
        data_args.min_similarity = min(labels)
        logger.warning(
            f"Setting min_similarity: {data_args.min_similarity}. Override by setting --min_similarity."
        )
    if data_args.max_similarity is None:
        data_args.max_similarity = max(labels)
        logger.warning(
            f"Setting max_similarity: {data_args.max_similarity}. Override by setting --max_similarity."
        )
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=1,
        # finetuning_task=None,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model_cls = get_model(model_args)
    config.update({
        "use_auth_token": model_args.use_auth_token,
        "model_revision": model_args.model_revision,
        "cache_dir": model_args.cache_dir,
        "model_name_or_path": model_args.model_name_or_path,
        "encoding_type": model_args.encoding_type,
        "objective": model_args.objective,
        "pooler_type": model_args.pooler_type,
        "transform": model_args.transform,
        "freeze_encoder": model_args.freeze_encoder,
        "triencoder_head": model_args.triencoder_head,
        "hypernet_scaler": model_args.hypernet_scaler,
        "hypernet_dual": model_args.hypernet_dual,
        "hypernet_ablation": model_args.hypernet_ablation,
        "cl_temp": model_args.cl_temp,
        "cl_in_batch_neg": model_args.cl_in_batch_neg,
    })
    model = model_cls(config=config)
    if model_args.freeze_encoder:
        for param in model.backbone.parameters():
            param.requires_grad = False

    # To load already trained model and tokenizer (but config should be same, and developer should check it manually)
    if not training_args.do_train and "/output/" in model_args.model_name_or_path:
        model, tokenizer = get_model_and_tokenizer(training_args.device, model_args.model_name_or_path, config)

    sentence1_key, sentence2_key, condition_key, similarity_key = (
        "sentence1",
        "sentence2",
        "condition",
        "label",
    )
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "The max_seq_length passed (%d) is larger than the maximum length for the "
            "model (%d). Using max_seq_length=%d."
            % (
                data_args.max_seq_length,
                tokenizer.model_max_length,
                tokenizer.model_max_length,
            )
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    preprocess_function = get_preprocessing_function(
        tokenizer,
        sentence1_key,
        sentence2_key,
        condition_key,
        similarity_key,
        padding,
        max_seq_length,
        model_args,
        scale=(data_args.min_similarity, data_args.max_similarity)
        if model_args.objective in {"mse", "triplet", "triplet_mse", "triplet_cl_mse"}
        else None,
        condition_only=data_args.condition_only,
        sentences_only=data_args.sentences_only,
        cl_in_batch_neg=model_args.cl_in_batch_neg,
    )
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            remove_columns=raw_datasets["validation"].column_names,
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    if training_args.do_eval:
        if (
                "validation" not in raw_datasets
                and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            input_ids = train_dataset[index]["input_ids"]
            logger.info(f"tokens: {tokenizer.decode(input_ids)}")
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def compute_metrics(output: EvalPrediction):
        preds = (
            output.predictions[0]
            if isinstance(output.predictions, tuple)
            else output.predictions
        )
        preds = np.squeeze(preds)
        return {
            "mse": ((preds - output.label_ids) ** 2).mean().item(),
            "pearsonr": pearsonr(preds, output.label_ids)[0],
            "spearmanr": spearmanr(preds, output.label_ids)[0],
        }

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            pad_token_type_id=tokenizer.pad_token_type_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    # Initialize our Trainer
    trainer_cls = (
        TripletTrainer
        if model_args.objective in {"triplet", "triplet_mse", "triplet_cl_mse"}
        else Trainer
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LogCallback)
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    # Evaluation
    combined = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        combined.update(metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", combined)
        # if training_args.do_train:  # FIXME Uncomment when testing
        #     metrics = trainer.evaluate(
        #         eval_dataset=train_dataset, metric_key_prefix="train"
        #     )
        #     max_eval_samples = (
        #         data_args.max_eval_samples
        #         if data_args.max_eval_samples is not None
        #         else len(eval_dataset)
        #     )
        #     metrics["train_samples"] = min(max_eval_samples, len(train_dataset))
        #     trainer.log_metrics("train", metrics)
        #     trainer.save_metrics("train", combined)
    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predict_dataset = predict_dataset.remove_columns("labels")
        predictions = trainer.predict(
            predict_dataset, metric_key_prefix="predict"
        ).predictions
        if model_args.encoding_type == "tri_encoder":  # Fixed code
            predictions = np.squeeze(predictions[1])
        else:  # Original code
            predictions = (np.squeeze(predictions)
                           if model_args.objective in {"mse", "triplet", "triplet_mse", "triplet_cl_mse"}
                           else np.argmax(predictions, axis=1))
        predictions = dict(enumerate(predictions.tolist()))
        output_predict_file = os.path.join(
            training_args.output_dir, f"test_predictions.json"
        )
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w", encoding="utf-8") as outfile:
                json.dump(predictions, outfile)
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "CSTS"}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
