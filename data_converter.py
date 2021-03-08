import importlib
import os
import re
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from utils import *
import mojimoji
from pyknp import Juman
from transformers import *
from textformatting import ssplit


def convert_single(in_dir, out_dir, doc_level, segmenter, bert_tokenizer):
    train_scale = 1.0
    with_dct = True
    file_list = [os.path.join(in_dir, file) for file in sorted(os.listdir(in_dir)) if file.endswith(".xml")]
    print(f"total files: {len(file_list)}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    batch_convert_document_to_conll(
        file_list,
        os.path.join(
            out_dir,
            f"single.conll"
        ),
        sent_tag=True,
        contains_modality=True,
        with_dct = with_dct,
        is_raw=False,
        morph_analyzer_name=segmenter,
        bert_tokenizer=bert_tokenizer,
        is_document=doc_level
    )


def cross_validation(in_dir, out_dir, doc_level, segmenter, cv_num, bert_tokenizer):
    train_scale = 1.0
    with_dct = True
    cv_data_split = doc_kfold(in_dir, train_scale=train_scale, cv=cv_num)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for cv_id, (train_files, dev_files, test_files) in enumerate(cv_data_split):
        print(len(train_files), len(dev_files), len(test_files))
        batch_convert_document_to_conll(
            train_files,
            os.path.join(
                out_dir,
                f"cv{cv_id}_train.conll"
            ),
            sent_tag=True,
            contains_modality=True,
            with_dct = with_dct,
            is_raw=False,
            morph_analyzer_name=segmenter,
            bert_tokenizer=bert_tokenizer,
            is_document=doc_level
        )
        batch_convert_document_to_conll(
            dev_files,
            os.path.join(
                out_dir,
                f"cv{cv_id}_dev.conll"
            ),
            sent_tag=True,
            contains_modality=True,
            with_dct = with_dct,
            is_raw=False,
            morph_analyzer_name=segmenter,
            bert_tokenizer=bert_tokenizer,
            is_document=doc_level
        )
        batch_convert_document_to_conll(
            test_files,
            os.path.join(
                out_dir,
                f"cv{cv_id}_test.conll"
            ),
            sent_tag=True,
            contains_modality=True,
            with_dct = with_dct,
            is_raw=False,
            morph_analyzer_name=segmenter,
            bert_tokenizer=bert_tokenizer,
            is_document=doc_level
        )

from argparse import ArgumentParser

parser = ArgumentParser(description='Convert xml to conll for training')

parser.add_argument("--xml", dest="xml_dir",
                    help="input xml dir")

parser.add_argument("--conll", dest="conll_dir",
                    help="output conll dir")

parser.add_argument("--doc_level",
                    action='store_true',
                    help="document-level extraction or sentence-level extraction")

parser.add_argument("--cv_num", default=5, type=int,
                    help="k-fold cross-validation, 0 presents not to split data")

parser.add_argument("--segmenter", default='jumanpp', type=str,
                    help="segmenter: jumanpp (w/ pyknp package) or mecab (w/ MeCab package)")

parser.add_argument("--bert_dir", type=str,
                    help="BERT dir for initializing tokenizer")

args = parser.parse_args()

bert_tokenizer = BertTokenizer.from_pretrained(
    args.bert_dir,
    do_lower_case=False,
    do_basic_tokenize=False,
    tokenize_chinese_chars=False
)
bert_tokenizer.add_tokens(['[JASP]'])

if args.cv_num == 0:
    convert_single(args.xml_dir, args.conll_dir, args.doc_level, args.segmenter, bert_tokenizer)
elif args.cv_num > 0:
    cross_validation(args.xml_dir, args.conll_dir, args.doc_level, args.segmenter, args.cv_num, bert_tokenizer)
else:
    raise Exception(f"Incorrect cv number {args.cv_num}...")

