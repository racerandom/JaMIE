#!/bin/zsh

GPU_ID=$1
CORPUS=$2
DOC_OR_SENT=$3
MODEL_DIR="checkpoints/pipemod/${CORPUS}_${DOC_OR_SENT}"
DATA_DIR="data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll"
OUT_DIR="tmp/pipemod_${CORPUS}_${DOC_OR_SENT}"
mkdir -p $OUT_DIR

for cv_id in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python clinical_pipeline_mod.py \
    --train_file "${DATA_DIR}/cv${cv_id}_train.conll" \
    --dev_file "${DATA_DIR}/cv${cv_id}_dev.conll" \
    --dev_output "${OUT_DIR}/cv${cv_id}_dev.out" \
    --saved_model "${MODEL_DIR}/cv${cv_id}" \
    --enc_lr 2e-5 \
    --warmup_epoch 2 \
    --num_epoch 12 \
    --batch_size 8 \
    --do_train
#    --non_bert \
#    --non_scheduled_lr \


    CUDA_VISIBLE_DEVICES=${GPU_ID} python clinical_pipeline_mod.py \
    --saved_model "${MODEL_DIR}/cv${cv_id}" \
    --test_file "${DATA_DIR}/cv${cv_id}_test.conll" \
    --test_output "${OUT_DIR}/cv${cv_id}_test.out" \
#    --non_bert \
#    --non_scheduled_lr \
    --batch_size 16
done

cat "${OUT_DIR}/cv0_test.out" "${OUT_DIR}/cv1_test.out" "${OUT_DIR}/cv2_test.out" "${OUT_DIR}/cv3_test.out" "${OUT_DIR}/cv4_test.out" > "${OUT_DIR}/test.out"

cat "${DATA_DIR}/cv0_test.conll" "${DATA_DIR}/cv1_test.conll" "${DATA_DIR}/cv2_test.conll" "${DATA_DIR}/cv3_test.conll" "${DATA_DIR}/cv4_test.conll" > "${DATA_DIR}/test.conll"

python clinical_eval.py --gold_file "${DATA_DIR}/test.conll" --pred_file "${OUT_DIR}/test.out" --eval_level 2 --print_level 2