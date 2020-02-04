#!/bin/zsh

CUDA_VISIBLE_DEVICES=$1 python clinical_cert.py \
-p /home/feicheng/transformers/examples/lm_finetuning/new_lm/finetuing_lm_${2} \
-m checkpoints/cert_${2} \
-o outputs/cert_lm_${2}_goku_ep${3}_out.txt \
-n outputs/ner_lm_${2}_goku_ep5_out.txt \
-e $3 \
-b 24 \
--do_train 
