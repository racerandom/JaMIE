#!/bin/zsh

CORPUS=$1
CORPUS2=$2
OUT=$3
DOC_OR_SENT=$4
INPUT_DIR="data/2021Q1/${CORPUS}/${DOC_OR_SENT}_conll/"
INPUT2_DIR="data/2021Q1/${CORPUS2}/${DOC_OR_SENT}_conll/"
OUTPUT_DIR="data/2021Q1/${OUT}/${DOC_OR_SENT}_conll/"

for cv_id in 0 1 2 3 4; do
    cat "${INPUT_DIR}/cv${cv_id}_train.conll" "${INPUT2_DIR}/cv${cv_id}_train.conll" > "${OUTPUT_DIR}/cv${cv_id}_train.conll"
    cat "${INPUT_DIR}/cv${cv_id}_dev.conll" "${INPUT2_DIR}/cv${cv_id}_dev.conll" > "${OUTPUT_DIR}/cv${cv_id}_dev.conll"
    cat "${INPUT_DIR}/cv${cv_id}_test.conll" "${INPUT2_DIR}/cv${cv_id}_test.conll" > "${OUTPUT_DIR}/cv${cv_id}_test.conll"
done
