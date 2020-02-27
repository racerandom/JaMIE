#!/bin/zsh

# usage: $1: test dir, $2 output dir, $3: model

for f in $(find $1 -name '*.conll'); do

fname=$(basename $f)
fbname=${fname%.*}

echo "Processing $f..."
python clinical_ner.py \
--train_file data/train_full.conll \
--test_file $f \
--batch 16 \
--fine_epoch 5 \
--epoch 10 \
--model $3 \
--do_crf \
--output $2/$fbname

done
