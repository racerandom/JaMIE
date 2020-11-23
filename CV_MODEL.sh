#!/bin/zsh

CORPUS=$2
DOC_OR_SENT=$3
cv_id=$4

#for cv_id in 0 1 2 3 4; do
CUDA_VISIBLE_DEVICES=$1 python clinical_joint.py \
--train_file "data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll/cv${cv_id}_train.conll" \
--dev_file "data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll/cv${cv_id}_dev.conll" \
--test_file "data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll/cv${cv_id}_test.conll" \
--dev_output "tmp/joint_${CORPUS}_${DOC_OR_SENT}_cv${cv_id}_dev.out" \
--test_output "tmp/joint_${CORPUS}_${DOC_OR_SENT}_cv${cv_id}_test.out" \
--saved_model "checkpoints/tmp/joint/${CORPUS}_${DOC_OR_SENT}_cv${cv_id}" \
--do_train \
--batch_size 1

CUDA_VISIBLE_DEVICES=$1 python clinical_joint.py \
--saved_model "checkpoints/tmp/joint/${CORPUS}_${DOC_OR_SENT}_cv${cv_id}" \
--test_file "data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll/cv${cv_id}_test.conll" \
--test_output "tmp/joint_${CORPUS}_${DOC_OR_SENT}_cv${cv_id}_test.out" \
--batch_size 4
#done

