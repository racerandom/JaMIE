#!/bin/zsh

CORPUS=$1
DOC_OR_SENT=$2
MODEL=$3
EVAL_LEVEL=$4

DATA_DIR="data/2020Q2/${CORPUS}/${DOC_OR_SENT}_conll/"
TMP_DIR="tmp/${MODEL}_${CORPUS}_${DOC_OR_SENT}"

cat "${TMP_DIR}_cv0_test.out" "${TMP_DIR}_cv1_test.out" "${TMP_DIR}_cv2_test.out" "${TMP_DIR}_cv3_test.out" "${TMP_DIR}_cv4_test.out" > "${TMP_DIR}_test.out"

cat "${DATA_DIR}/cv0_test.conll" "${DATA_DIR}/cv1_test.conll" "${DATA_DIR}/cv2_test.conll" "${DATA_DIR}/cv3_test.conll" "${DATA_DIR}/cv4_test.conll" > "${DATA_DIR}/test.conll"

python clinical_eval.py --gold_file "${DATA_DIR}/test.conll" --pred_file "${TMP_DIR}_test.out" --eval_level $EVAL_LEVEL --print_level 2
