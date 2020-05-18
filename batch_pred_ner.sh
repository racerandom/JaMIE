#!/bin/zsh

# usage: $1: test dir, $2 output dir, $3: model

for f in $(find $1 -name '*.conll'); do

fname=$(basename $f)
fbname=${fname%.*}

echo "Processing $f..."
python clinical_ner.py \
--train_file data/ou_1225/ou_kuroda2018_full2.conll \
--dev_file data/kuroda2018_10/kuroda2018_10.conll \
--dev_output outputs/tmp_dev.conll \
--test_file $f \
--test_output $2/$fbname.pred.conll \
--batch 16 \
--model $3 \
--do_crf \
--joint \
--fp16 \

done
