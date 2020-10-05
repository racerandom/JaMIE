#!/bin/zsh


for lr in 1e-3;do
	for bs in 64 128 256;do
        for hs in 600;do
            saved_model="checkpoints/tmp/rel/run${1}_lr${lr}_bs${bs}_hs${hs}_lstm_fp32"
            dev_output="tmp/dev_run${1}_lr${lr}_bs${bs}_hs${hs}_lstm_fp32.ner"
            test_output="tmp/test_run${1}_lr${lr}_bs${bs}_hs${hs}_lstm_fp32.ner"

		    CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_ner.py \
		    --saved_model $saved_model \
		    --dev_output $dev_output \
		    --batch_size $bs \
		    --enc_lr $lr \
            --encoder_hidden_size $hs \
		    --num_epoch 20 \
            --non_bert \
            --non_scheduled_lr \
            --do_train

		    CUDA_VISIBLE_DEVICES=$1 python clinical_pipeline_ner.py \
		    --saved_model $saved_model \
		    --test_output $test_output \
		    --batch_size 1024 \
            --non_bert 
        done
	done
done
