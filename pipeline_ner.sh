#!/bin/zsh


for lr in 1e-5 2e-5 5e-5;do
	for bs in 16 32 64;do
		CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_rel.py \
		--saved_model "checkpoints/tmp/rel/run${1}_lr${lr}_bs${bs}_fp32" \
		--dev_output "tmp/dev_run${1}_lr${lr}_bs${bs}_fp32.rel" \
		--batch_size $bs \
		--enc_lr $lr \
		--num_epoch 15 \
		--do_train

		CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_rel.py \
		--saved_model "checkpoints/tmp/rel/run${1}_lr${lr}_bs${bs}_fp32" \
		--test_output "tmp/test_run${1}_lr${lr}_bs${bs}_fp32.rel" \
		--batch_size 768 
	done
done
