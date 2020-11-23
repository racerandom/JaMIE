#!/bin/zsh

CORPUS=$2
DOC_OR_SENT=$3
SAVED_MODEL="checkpoints/joint/${CORPUS}_${DOC_OR_SENT}_cv${cv_id}"
TEST_FILE="data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll/cv${cv_id}_test.conll"
TEST_OUTPUT="tmp/joint_${CORPUS}_${DOC_OR_SENT}_cv${cv_id}_test.out"

for cv_id in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$1 python clinical_joint.py \
    --train_file "data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll/cv${cv_id}_train.conll" \
    --dev_file "data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll/cv${cv_id}_dev.conll" \
    --dev_output "tmp/joint_${CORPUS}_${DOC_OR_SENT}_cv${cv_id}_dev.out" \
    --test_file $TEST_FILE \
    --test_output $TEST_OUTPUT \
    --saved_model $SAVED_MODEL \
    --do_train \
    --batch_size 1

    CUDA_VISIBLE_DEVICES=$1 python clinical_joint.py \
    --saved_model $SAVED_MODEL \
    --test_file $TEST_FILE \
    --test_output $TEST_OUTPUT \
    --batch_size 4
done

