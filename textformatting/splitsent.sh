#!/bin/bash

for filename in $1/*.txt; do
    fwop=$(basename "$filename")
    fwoe="${fwop%.*}"
    sent_file="$1/$fwoe.sent"
    echo $sent_file
    cat $filename | perl -I sentence-splitter.pl | python split_tnm.py > $sent_file
#    cat $filename | perl -I sentence-splitter.pl > $sent_file
done
