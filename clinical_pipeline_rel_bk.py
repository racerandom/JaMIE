#!/usr/bin/env python
# coding: utf-8
import warnings

import os
import argparse

from tqdm import tqdm
import torch
from sklearn.metrics import classification_report
from imblearn.metrics import classification_report_imbalanced
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss
from transformers import *
from model import *

import utils
warnings.filterwarnings("ignore")


def eval_rel(model, dataloader, rel2ix, device, is_reported=False, file_out=None):

    ix2rel = {v: k for k, v in rel2ix.items() if k != 'N'}

    if isinstance(file_out, str):
        dir_name, file_name = os.path.split(file_out)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    model.eval()
    pred_all, gold_all = [], []
    with torch.no_grad():
        for batch in dataloader:
            b_toks, b_attn_mask, b_ner, b_tail_mask, b_tail_ne, b_head_mask, b_head_ne, b_rel = tuple(
                t.to(device) for t in batch
            )
            logits = model(b_toks, b_attn_mask, b_tail_mask, b_tail_ne, b_head_mask, b_head_ne)[0]
            b_pred = torch.argmax(logits, -1)
            pred_all += b_pred.tolist()
            gold_all += b_rel.tolist()
    report = classification_report(
        gold_all, pred_all,
        list(ix2rel.keys()), target_names=list(ix2rel.values()),
        digits=4
    )

    micro_f1 = report.split('\n')[-4].split()[4]
    macro_f1 = report.split('\n')[-3].split()[4]
    if is_reported:
        print(report)
        # print(micro_f1)
    return float(macro_f1)


def main():
    parser = argparse.ArgumentParser(description='PRISM tag recognizer')

    parser.add_argument("--train_file", default="data/CoNLL04/train.txt", type=str,
                        help="train file, multihead conll format.")

    parser.add_argument("--dev_file", default="data/CoNLL04/dev.txt", type=str,
                        help="dev file, multihead conll format.")

    parser.add_argument("--test_file", default="data/CoNLL04/test.txt", type=str,
                        help="test file, multihead conll format.")

    parser.add_argument("--pretrained_model",
                        default='bert-base-uncased',
                        type=str,
                        help="pre-trained model dir")

    parser.add_argument("--do_lower_case",
                        # action='store_True',
                        default=True,
                        type=bool,
                        help="tokenizer: do_lower_case")

    # parser.add_argument("--train_file", default="data/clinical2020Q1/cv1_train.conll", type=str,
    #                     help="train file, multihead conll format.")
    #
    # parser.add_argument("--dev_file", default="data/clinical2020Q1/cv1_dev.conll", type=str,
    #                     help="dev file, multihead conll format.")
    #
    # parser.add_argument("--test_file", default="data/clinical2020Q1/cv1_test.conll", type=str,
    #                     help="test file, multihead conll format.")
    #
    # parser.add_argument("--pretrained_model",
    #                     default="/home/feicheng/Tools/Japanese_L-12_H-768_A-12_E-30_BPE",
    #                     type=str,
    #                     help="pre-trained model dir")
    #
    # parser.add_argument("--do_lower_case",
    #                     # action='store_True',
    #                     default=False,
    #                     type=bool,
    #                     help="tokenizer: do_lower_case")

    parser.add_argument("--save_model", default='checkpoints/rel', type=str,
                        help="save/load model dir")

    parser.add_argument("--batch_size", default=32, type=int,
                        help="BATCH SIZE")

    parser.add_argument("--num_epoch", default=10, type=int,
                        help="fine-tuning epoch number")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--lr", default=5e-5, type=float,
                        help="learning rate")

    parser.add_argument("--ne_size", default=64, type=int,
                        help="size of name entity embedding")

    parser.add_argument("--save_best", default='f1', type=str,
                        help="save the best model, given dev scores (f1 or loss)")

    parser.add_argument("--save_step_interval", default=100, type=int,
                        help="save best model given a step interval")

    parser.add_argument("--neg_ratio", default=1.0, type=float,
                        help="negative sample ratio")

    args = parser.parse_args()

    print(args)

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    print(device)

    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model,
        do_lower_case=args.do_lower_case,
        do_basic_tokenize=False
    )
    tokenizer.add_tokens(['[JASP]'])

    from collections import defaultdict

    rel_count = defaultdict(lambda: 0)

    train_toks, train_labs, train_rels, bio2ix, ne2ix, rel2ix = utils.extract_rel_data_from_mh_conll(args.train_file, args.neg_ratio)
    print(bio2ix)
    print(ne2ix)
    print(rel2ix)
    print('max sent len:', utils.max_sents_len(train_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in train_rels]), max([len(sent_rels) for sent_rels in train_rels]))
    print()

    dev_toks, dev_labs, dev_rels, _, _, _ = utils.extract_rel_data_from_mh_conll(args.dev_file, 1.0)
    print('max sent len:', utils.max_sents_len(dev_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in dev_rels]), max([len(sent_rels) for sent_rels in dev_rels]))
    print()

    test_toks, test_labs, test_rels, _, _, _ = utils.extract_rel_data_from_mh_conll(args.test_file, 1.0)
    print('max sent len:', utils.max_sents_len(test_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in test_rels]), max([len(sent_rels) for sent_rels in test_rels]))
    print()

    for sent_rels in train_rels:
        for rel in sent_rels:
            rel_count[rel[-1]] += 1

    for sent_rels in dev_rels:
        for rel in sent_rels:
            rel_count[rel[-1]] += 1

    for sent_rels in test_rels:
        for rel in sent_rels:
            rel_count[rel[-1]] += 1

    print(rel_count)

    max_len = max(
        utils.max_sents_len(train_toks, tokenizer),
        utils.max_sents_len(dev_toks, tokenizer),
        utils.max_sents_len(test_toks, tokenizer)
    )

    train_dataset = utils.convert_rels_to_tensors(train_toks, train_labs, train_rels, tokenizer, bio2ix, ne2ix, rel2ix, max_len)
    dev_dataset = utils.convert_rels_to_tensors(dev_toks, dev_labs, dev_rels, tokenizer, bio2ix, ne2ix, rel2ix, max_len)
    test_dataset = utils.convert_rels_to_tensors(test_toks, test_labs, test_rels, tokenizer, bio2ix, ne2ix, rel2ix, max_len)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.batch_size)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    num_epoch_steps = len(train_dataloader)
    num_training_steps = args.num_epoch * num_epoch_steps
    warmup_ratio = 0.1

    model = BertRel.from_pretrained(args.pretrained_model, ne_size=args.ne_size, num_ne=len(ne2ix), num_rel=len(rel2ix))

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        correct_bias=False
    )

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps * warmup_ratio,
                                                num_training_steps=num_training_steps)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    best_dev_f1 = float('-inf')

    model.zero_grad()
    for epoch in range(1, args.num_epoch + 1):
        epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(epoch_iterator, start=1):
            model.train()
            b_toks, b_attn_mask, b_ner, b_tail_mask, b_tail_ne, b_head_mask, b_head_ne, b_rel = tuple(
               t.to(device) for t in batch
            )
            loss = model(b_toks, b_attn_mask, b_tail_mask, b_tail_ne, b_head_mask, b_head_ne, b_rel)[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if epoch > 3 and ((step % args.save_step_interval == 0) or (step == num_epoch_steps)):
                dev_f1 = eval_rel(model, dev_dataloader, rel2ix, device)

                if best_dev_f1 < dev_f1:
                    print("Previous best dev f1 %.4f; current f1 %.4f; best model saved '%s'" % (
                        best_dev_f1,
                        dev_f1,
                        args.save_model
                    ))
                    best_dev_f1 = dev_f1

                    """ save the trained model per epoch """
                    if not os.path.exists(args.save_model):
                        os.makedirs(args.save_model)
                    model.save_pretrained(args.save_model)
                    tokenizer.save_pretrained(args.save_model)

    model = BertRel.from_pretrained(args.save_model, ne_size=args.ne_size, num_ne=len(ne2ix), num_rel=len(rel2ix))
    model.to(device)
    eval_rel(model, test_dataloader, rel2ix, device, is_reported=True)


if __name__ == '__main__':
    main()
