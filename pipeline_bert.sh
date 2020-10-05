#!/bin/zsh

BERT_URL="/home/feicheng/Tools/ClinicalBERT"

CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_ner.py \
--saved_model "checkpoints/tmp/pipeline/ner/clinicalbert/" \
--dev_output "tmp/pipeline/dev_clinicalbert.rel" \
--batch_size 32 \
--enc_lr 5e-5 \
--num_epoch 15 \
--encoder_hidden_size 768 \
--do_train

CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_mod.py \
--saved_model "checkpoints/tmp/pipeline/mod/clinicalbert/" \
--batch_size 32 \
--enc_lr 5e-5 \
--num_epoch 10 \
--encoder_hidden_size 768 \
--do_train

CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_rel.py \
--saved_model "checkpoints/tmp/pipeline/rel/clinicalbert/" \
--batch_size 16 \
--num_epoch 10 \
--do_train

