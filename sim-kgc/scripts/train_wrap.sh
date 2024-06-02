#!/usr/bin/env bash

trap "kill 0" SIGINT SIGTERM

CUDA_VISIBLE_DEVICES=0 bash ./scripts/train_wn.sh --pretrained-model bert-large-uncased --batch-size 256 &
sleep 3
CUDA_VISIBLE_DEVICES=1 bash ./scripts/train_wn.sh --pretrained-model bert-large-uncased --encoding_type tri_encoder --batch-size 256 &
sleep 3
CUDA_VISIBLE_DEVICES=2 bash ./scripts/train_wn.sh --pretrained-model bert-large-uncased --encoding_type tri_encoder --hypernet_scaler 12 --batch-size 256 &
wait