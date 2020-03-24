#!/bin/zsh

for cv_i in 0 1 2 3 4
do
    TRAIN_FILE="${1}/cv${cv_i}_train.conll"
    TEST_FILE="${1}/cv${cv_i}_test.conll"

    python clinical_ner.py \
    --train_file ${TRAIN_FILE} \
    --test_file ${TEST_FILE} \
    --batch 16 \
    --epoch 15 \
    --output "outputs/ner/seq_${2}_cv${cv_i}.conll" \
    --do_train \
    --model "checkpoints/ner/${2}_cv${cv_i}/seq" \
    --fine_epoch 15 \
    --do_crf \
    --freeze 5
done

