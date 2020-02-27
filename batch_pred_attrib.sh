#!/bin/zsh

# usage: $1: ner_out dir, $2 output dir, $3: model, $4: attrib

for f in $(find $1 -name '*.pred.conll'); do

fname=$(basename $f)
fbname=${fname%%.*}

echo "Processing $f ..."
python clinical_cert.py \
--train_file data/train_full.conll \
--ner_out $f \
--model $3 \
--batch 16 \
--epoch 5 \
--attrib $4 \
--output $2/$fbname.$4.conll

done
