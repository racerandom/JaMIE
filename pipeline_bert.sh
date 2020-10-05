#!/bin/zsh

BERT="clinicalbert"
BERT_URL="/home/feicheng/Tools/ClinicalBERT"
#BERT="bert"
#BERT_URL="bert-base-uncased"

CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_ner.py \
--pretrained_model $BERT_URL \
--saved_model "checkpoints/tmp/pipeline/ner/${BERT}/" \
--dev_output "tmp/pipeline/dev_${BERT}.ner" \
--batch_size 32 \
--enc_lr 5e-5 \
--num_epoch 15 \
--encoder_hidden_size 768 \
--do_train

CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_mod.py \
--pretrained_model $BERT_URL \
--saved_model "checkpoints/tmp/pipeline/mod/${BERT}/" \
--dev_output "tmp/pipeline/dev_${BERT}.mod" \
--batch_size 32 \
--enc_lr 5e-5 \
--num_epoch 10 \
--encoder_hidden_size 768 \
--do_train

CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_rel.py \
--pretrained_model $BERT_URL \
--saved_model "checkpoints/tmp/pipeline/rel/${BERT}/" \
--dev_output "tmp/pipeline/dev_${BERT}.rel" \
--batch_size 16 \
--num_epoch 10 \
--do_train

