#!/bin/zsh


for lr in 1e-5 2e-5 5e-5;do
	for bs in 16 32 64;do
		echo "evaluating tmp/test_run${1}_lr${lr}_bs${bs}_${2}.rel"
		python clinical_eval.py \
		--test_file "data/i2b2_cleaned/i2b2_test.conll" \
		--pred_file "tmp/test_run${1}_lr${lr}_bs${bs}_${2}.rel" \
		--eval_level ${3}
	done
done
