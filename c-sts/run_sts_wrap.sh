#!/bin/bash

trap "kill 0" SIGINT SIGTERM

current_date_time=$(date)
echo "Start Current date and time: $current_date_time"

CUDA_VISIBLE_DEVICES=0 LR=1e-5 bash ./run_sts.sh &
CUDA_VISIBLE_DEVICES=1 LR=2e-5 bash ./run_sts.sh &
CUDA_VISIBLE_DEVICES=2 LR=3e-5 TEMP=1.7 bash ./run_sts.sh &

echo -e "started procs: \n$(jobs -p)"
wait

#output_dir=output_hypernet_freeze_dual_full_24_32
#model=(princeton-nlp/sup-simcse-roberta-base)
#hn_s=(12)
#lr=(1e-5)
#wd=(0.1)
#temp=(1.5 1.7 2.0)
#seed=(53 54 55)
#
#for m in ${!model[@]}; do
#  for h in ${!hn_s[@]}; do
#    for l in ${!lr[@]}; do
#      for w in ${!wd[@]}; do
#        for t in ${!temp[@]}; do
#  #       for s in ${!seed[@]}; do
#            i=0
#            s=$i
#            CUDA_VISIBLE_DEVICES=$i OUTPUT_DIR=${output_dir} MODEL=${model[$m]} LR=${lr[$l]} WD=${wd[$w]} TEMP=${temp[$t]} HN_S=${hn_s[$h]} SEED=${seed[$s]} bash ./run_sts.sh &
#            i=1
#            s=$i
#            CUDA_VISIBLE_DEVICES=$i OUTPUT_DIR=${output_dir} MODEL=${model[$m]} LR=${lr[$l]} WD=${wd[$w]} TEMP=${temp[$t]} HN_S=${hn_s[$h]} SEED=${seed[$s]} bash ./run_sts.sh &
#            i=2
#            s=$i
#            CUDA_VISIBLE_DEVICES=$i OUTPUT_DIR=${output_dir} MODEL=${model[$m]} LR=${lr[$l]} WD=${wd[$w]} TEMP=${temp[$t]} HN_S=${hn_s[$h]} SEED=${seed[$s]} bash ./run_sts.sh &
#            wait
#            (find ${output_dir} -name "pytorch_model.bin" | xargs -I {} rm -rf {}) && (find ${output_dir} -name "optimizer.pt" | xargs -I {} rm -rf {}) && echo "done"
#  #       done
#        done
#      done
#    done
#  done
#done

#current_date_time=$(date)
#echo "End Current date and time: $current_date_time"
