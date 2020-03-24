#!/bin/zsh

FILE=$PWD/$1

cd conlleval
python conlleval.py < $FILE
