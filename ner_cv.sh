#!/bin/zsh

for cv_i in 0 1 2 3 4
do
    TRAIN_FILE="${1}/cv${cv_i}_train.conll"
    DEV_FILE="${1}/cv${cv_i}_dev.conll"
    TEST_FILE="${1}/cv${cv_i}_test.conll"

    python clinical_ner.py \
    --train_file ${TRAIN_FILE} \
    --test_file ${TEST_FILE} \
    --dev_file ${DEV_FILE} \
    --batch 16 \
    --epoch 12 \
    --output "outputs/ner/${2}_cv${cv_i}.conll" \
    --do_train \
    --model "checkpoints/ner/${2}_cv${cv_i}" \
    --fine_epoch 12 \
    --do_crf \
    --fp16 \
    --bottomup_freeze \
    --later_eval \
    --joint \
    --save_step_interval 50 \
    --save_best
done

cat "outputs/ner/${2}_cv$*.conll" > "outputs/ner/${2}.conll"

