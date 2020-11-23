#!/bin/zsh

CORPUS=$2
DOC_OR_SENT=$3
SAVED_MODEL="checkpoints/piperel/${CORPUS}_${DOC_OR_SENT}"
TEST_FILE="data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll"
TEST_OUTPUT="tmp/piperel_${CORPUS}_${DOC_OR_SENT}"

for cv_id in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_rel.py \
    --train_file "data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll/cv${cv_id}_train.conll" \
    --dev_file "data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll/cv${cv_id}_dev.conll" \
    --dev_output "tmp/piperel_${CORPUS}_${DOC_OR_SENT}_cv${cv_id}_dev.out" \
    --saved_model "${SAVED_MODEL}_cv${cv_id}" \
    --do_train \
    --batch_size 1

    CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_rel.py \
    --saved_model "${SAVED_MODEL}_cv${cv_id}" \
    --test_file "${TEST_FILE}/cv${cv_id}_test.conll" \
    --test_output "${TEST_OUTPUT}_cv${cv_id}_test.out" \
    --batch_size 4
done

