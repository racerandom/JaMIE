#!/bin/zsh

CORPUS=$2
DOC_OR_SENT=$3
MODEL=$4
SAVED_MODEL="checkpoints/${MODEL}/${CORPUS}_${DOC_OR_SENT}"
DATA_DIR="data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll"
OUTFILE_DIR="tmp/${MODEL}_${CORPUS}_${DOC_OR_SENT}"
mkdir -p $OUTFILE_DIR

for cv_id in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$1 python clinical_${MODEL}.py \
    --train_file "${DATA_DIR}/cv${cv_id}_train.conll" \
    --dev_file "${DATA_DIR}/cv${cv_id}_dev.conll" \
    --dev_output "${OUTFILE_DIR}/cv${cv_id}_dev.out" \
    --saved_model "${SAVED_MODEL}/cv${cv_id}" \
    --enc_lr 5e-5 \
    --warmup_epoch 2 \
    --num_epoch 25 \
    --do_train \
    --fp16 \
    --batch_size 2

    CUDA_VISIBLE_DEVICES=$1 python clinical_${MODEL}.py \
    --saved_model "${SAVED_MODEL}/cv${cv_id}" \
    --test_file "${DATA_DIR}/cv${cv_id}_test.conll" \
    --test_output "${OUTFILE_DIR}/cv${cv_id}_test.out" \
    --batch_size 4
done

cat "${OUTFILE_DIR}/cv0_test.out" "${OUTFILE_DIR}/cv1_test.out" "${OUTFILE_DIR}/cv2_test.out" "${OUTFILE_DIR}/cv3_test.out" "${OUTFILE_DIR}/cv4_test.out" > "${OUTFILE_DIR}/test.out"

cat "${DATA_DIR}/cv0_test.conll" "${DATA_DIR}/cv1_test.conll" "${DATA_DIR}/cv2_test.conll" "${DATA_DIR}/cv3_test.conll" "${DATA_DIR}/cv4_test.conll" > "${DATA_DIR}/test.conll"

python clinical_eval.py --gold_file "${DATA_DIR}/test.conll" --pred_file "${OUTFILE_DIR}/test.out" --eval_level 0 --print_level 2
