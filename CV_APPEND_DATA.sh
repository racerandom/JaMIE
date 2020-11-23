#!/bin/zsh

CORPUS=$1
CORPUS2=$2
DOC_OR_SENT=$3
INPUT_DIR="data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll/"
INPUT2_FILE="data/2020Q2/${CORPUS2}/${DOC_OR_SENT}_conll/test.conll"
OUTPUT_DIR="data/2020Q2/${CORPUS}/${DOC_OR_SENT}2_conll/"

for cv_id in 0 1 2 3 4; do
    cat "${INPUT_DIR}/cv${cv_id}_train.conll" $INPUT2_FILE > "${OUTPUT_DIR}/cv${cv_id}_train.conll"
    cp "${INPUT_DIR}/cv${cv_id}_dev.conll" "${OUTPUT_DIR}/cv${cv_id}_dev.conll"
    cp "${INPUT_DIR}/cv${cv_id}_test.conll" "${OUTPUT_DIR}/cv${cv_id}_test.conll"
done