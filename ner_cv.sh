#!/bin/zsh


for cv_i in 0 1 2 3 4
do
    TRAIN_FILE="${1}/cv${cv_i}_train+ou.conll"
    TEST_FILE="${1}/cv${cv_i}_test.conll"

    python clinical_ner.py \
    --train_file ${TRAIN_FILE} \
    --test_file ${TEST_FILE} \
    --batch 16 \
    --epoch 10 \
    --output "outputs/ner/seq_${2}_cv${cv_i}+ou.conll" \
    --do_train \
    --model "checkpoints/ner/${2}_cv${cv_i}+ou/seq" \
    --fine_epoch 10 \
    --do_crf
done

