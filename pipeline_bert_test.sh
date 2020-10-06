#!/bin/zsh

BERT="clinicalbert"
BERT_URL="/home/feicheng/Tools/ClinicalBERT"
#BERT="bert"
#BERT_URL="bert-base-uncased"

CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_ner.py \
--saved_model "checkpoints/tmp/pipeline/ner/${BERT}/" \
--test_output "tmp/pipeline/test_${BERT}.ner" \
--batch_size 64 \

CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_mod.py \
--saved_model "checkpoints/tmp/pipeline/mod/${BERT}/" \
--test_file "tmp/pipeline/test_${BERT}.ner" \
--test_output "tmp/pipeline/test_${BERT}.mod" \
--batch_size 64 \

CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_rel.py \
--saved_model "checkpoints/tmp/pipeline/rel/${BERT}/" \
--test_file "tmp/pipeline/test_${BERT}.mod" \
--test_output "tmp/pipeline/test_${BERT}.rel" \
--batch_size 32 \

