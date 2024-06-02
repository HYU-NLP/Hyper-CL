#!/bin/bash

model=${MODEL:-princeton-nlp/sup-simcse-roberta-base}  # pre-trained model
encoding=${ENCODER_TYPE:-tri_encoder}  # cross_encoder, bi_encoder, tri_encoder
freeze_encoder=${FREEZE_ENCODER:-False}  # whether to freeze the encoder
batch_size=${BS:-32}  # batch size
gradient_accumulation_steps=${G_ACC:-1}  # gradient accumulation steps
lr=${LR:-3e-5}  # learning rate
wd=${WD:-0.1}  # weight decay
transform=${TRANSFORM:-False}  # whether to use an additional linear layer after the encoder
objective=${OBJECTIVE:-triplet_cl_mse}  # mse, triplet, triplet_mse, triplet_cl_mse
cl_temp=${TEMP:-1.5}  # temperature for contrastive loss
cl_in_batch_neg=${CL_IN_BATCH_NEG:-False}
triencoder_head=${TRIENCODER_HEAD:-hypernet}
hypernet_scaler=${HN_S:-12}
hypernet_dual=${HN_DUAL:-False}
seed=${SEED:-42}
output_dir=${OUTPUT_DIR:-output}

config=enc_${encoding}
if [[ "${freeze_encoder}" == "True" ]]; then
  config=${config}__freeze
fi
if [[ "${hypernet_dual}" == "True" ]]; then
  config=${config}__dual
fi

config=${config}__lr_${lr}__wd_${wd}__trans_${transform}
if [[ "${objective}" == "triplet_cl_mse" ]]; then
  config=${config}__obj_${objective}__temp_${cl_temp}
else
  config=${config}__obj_${objective}
fi


if [[ "${encoding}" == "tri_encoder" && ${triencoder_head} == "hypernet" ]]; then
  config=${config}__tri_${triencoder_head}__hn_s_${hypernet_scaler}
elif [[ "${encoding}" == "tri_encoder" ]]; then
  config=${config}__tri_${triencoder_head}
fi
config=${config}__s_${seed}

train_file=${TRAIN_FILE:-data/csts_train.csv}
eval_file=${EVAL_FILE:-data/csts_validation.csv}
test_file=${TEST_FILE:-data/csts_test.csv}

output_dir="${output_dir}/${model//\//__}/${config}"
mkdir -p $output_dir

# FIXME Change "do_predict False" when testing

python run_sts.py \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --model_name_or_path ${model} \
  --objective ${objective} \
  --encoding_type ${encoding} \
  --pooler_type cls \
  --freeze_encoder ${freeze_encoder} \
  --transform ${transform} \
  --triencoder_head ${triencoder_head} \
  --max_seq_length 512 \
  --train_file ${train_file} \
  --validation_file ${eval_file} \
  --test_file ${test_file} \
  --condition_only False \
  --sentences_only False \
  --do_train \
  --do_eval \
  --do_predict False \
  --evaluation_strategy epoch \
  --per_device_train_batch_size ${batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --learning_rate ${lr} \
  --weight_decay ${wd} \
  --max_grad_norm 0.0 \
  --num_train_epochs 5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.1 \
  --log_level info \
  --disable_tqdm False \
  --save_strategy epoch \
  --save_total_limit 1 \
  --seed ${seed} \
  --data_seed ${seed} \
  --fp16 True \
  --cl_temp ${cl_temp} \
  --cl_in_batch_neg ${cl_in_batch_neg} \
  --hypernet_scaler ${hypernet_scaler} \
  --hypernet_dual ${hypernet_dual} \
  --log_time_interval 15 \
  "$@" >"${output_dir}/run.log" 2>&1