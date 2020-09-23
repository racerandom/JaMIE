#!/bin/zsh

CUDA_VISIBLE_DEVICES=$1 python clinical_joint.py \
--train_file "data/clinical20200605/doc_cv5_mecab_${2}_wo_dct_filtered/${3}_train.conll" \
--dev_file "data/clinical20200605/doc_cv5_mecab_${2}_wo_dct_filtered/${3}_dev.conll" \
--test_file "data/clinical20200605/doc_cv5_mecab_${2}_wo_dct_filtered/${3}_test.conll" \
--pred_file "tmp/doc_mecab_${2}_wo_dct_filtered_${3}.conll" \
--saved_model "checkpoints/tmp/doc_cv5_mecab_${2}_wo_dct_filtered/${3}" \
--do_train \
--batch_size 2
