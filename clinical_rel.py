#!/usr/bin/env python
# coding: utf-8
import warnings

import pandas as pd
import csv
import random
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

TOK_ID_COL = 0
TOK_COL = 1
BIO_COL = 2
RELS_COL = 3
HEAD_IDS_COL = 4


def read_multihead_conll(file_name):
    col_name_list = ['token_id', 'token', "BIO", "relation", 'head']
    conll_data = pd.read_csv(
        file_name, names=col_name_list, encoding="utf-8",
        engine='python', sep="\t", quoting=csv.QUOTE_NONE
    ).values.tolist()
    return conll_data


def convert_multihead_conll_2d_to_3d(list_2d, sep='#doc'):
    list_3d = []
    sent_cache = []
    for entry in list_2d:
        if entry[0].startswith(sep):
            if sent_cache:
                list_3d.append(sent_cache)
                sent_cache = []
            continue
        sent_cache.append(entry)
    if sent_cache:
        list_3d.append(sent_cache)
    return list_3d


def extract_entity_ids_from_conll_sent(conll_sent):
    entity2ids = {}
    pos_rels = {}
    entity_cache = [[], 'O']
    prev_bio, prev_tag = 'O', 'N'
    for sent_id, toks in enumerate(conll_sent):
        bio, tag = toks[BIO_COL].split('-') if len(toks[BIO_COL].split('-')) > 1 else ('O', 'N')
        for head, rel in zip(eval(toks[HEAD_IDS_COL]), eval(toks[RELS_COL])):
            pos_rels[(toks[TOK_ID_COL], str(head))] = rel
        if bio == 'B':
            #             import pdb; pdb.set_trace()
            if entity_cache[0]:
                entity2ids[entity_cache[0][-1]] = entity_cache.copy()
                entity_cache = [[], 'O']
            entity_cache[0].append(toks[TOK_ID_COL])
            entity_cache[1] = tag
        elif bio == 'I':
            if tag != prev_tag and entity_cache[0]:
                entity2ids[entity_cache[0][-1]] = entity_cache.copy()
                entity_cache = [[], 'O']
            entity_cache[0].append(toks[TOK_ID_COL])
            entity_cache[1] = tag
        elif bio == 'O':
            if prev_bio != 'O' and entity_cache[0]:
                entity2ids[entity_cache[0][-1]] = entity_cache.copy()
                entity_cache = [[], 'O']
        else:
            raise Exception("[ERROR] Unknown bio tag '%s'.." % bio)

        # if is the last line
        if sent_id == (len(conll_sent) - 1):
            if entity_cache[0]:
                entity2ids[entity_cache[0][-1]] = entity_cache.copy()
                entity_cache = [[], 'O']
        else:
            prev_bio, prev_tag = bio, tag
    return entity2ids, pos_rels


def extract_rels_from_conll_sent(conll_sent, down_neg=1.0):
    entity2ids, pos_rels = extract_entity_ids_from_conll_sent(conll_sent)
    keys = list(entity2ids.keys())
    sent_rels = []
    for tail_id in range(len(keys)):
        for head_id in range(len(keys)):
            if tail_id != head_id:
                rel = [
                    entity2ids[keys[tail_id]][0],
                    entity2ids[keys[tail_id]][1],
                    entity2ids[keys[head_id]][0],
                    entity2ids[keys[head_id]][1]
                ]
                if (keys[tail_id], keys[head_id]) in pos_rels:
                    rel.append(pos_rels[(keys[tail_id], keys[head_id])])
                    sent_rels.append(rel)
                else:
                    rel.append('N')
                    if random.random() < down_neg:
                        sent_rels.append(rel)

    return sent_rels


def max_sents_len(toks, tokenizer):
    return max([len(tokenizer.tokenize(' '.join(sent_toks))) for sent_toks in toks])


def extract_rel_data_from_mh_conll(conll_file, down_neg):
    conll_data = read_multihead_conll(conll_file)
    conll_sents = convert_multihead_conll_2d_to_3d(conll_data)
    ner_toks = [[tok[1] for tok in sent] for sent in conll_sents]
    ner_labs = [[tok[2] for tok in sent] for sent in conll_sents]

    rel_tuples = []
    for index, sent in enumerate(conll_sents):
        rel_tuples.append(extract_rels_from_conll_sent(sent, down_neg=down_neg))

    assert len(ner_toks) == len(ner_labs) == len(rel_tuples)
    print('number of sents:', len(ner_toks))
    print('number of ne:', len([ner for sent_ner in ner_labs for ner in sent_ner if ner.startswith('B-')]))
    print(
        'pos rels:', len([rel for sent_rel in rel_tuples for rel in sent_rel if rel[-1] != 'N']),
        'neg rels:', len([rel for sent_rel in rel_tuples for rel in sent_rel if rel[-1] == 'N'])
    )
    bio2ix = utils.get_label2ix(ner_labs)
    ne2ix = utils.get_label2ix([[lab.split('-')[-1] for lab in labs if '-' in lab] for labs in ner_labs])
    rel2ix = utils.get_label2ix([eval(tok[3]) for sent in conll_sents for tok in sent])

    return ner_toks, ner_labs, rel_tuples, bio2ix, ne2ix, rel2ix


def mask_one_entity(entity_tok_ids):
    mask_seq = [0] * (int(entity_tok_ids[-1]) + 1)
    for index in entity_tok_ids:
        mask_seq[int(index)] = 1
    return mask_seq


def match_bpe_mask(bpe_x, mask):
    bpe_mask = mask.copy()
    for i in range(len(bpe_x)):
        if i > 0 and bpe_x[i].startswith('##'):
            bpe_mask.insert(i, bpe_mask[i - 1])
    assert len(bpe_x) == len(bpe_mask)
    return bpe_mask


def convert_rels_to_tensors(ner_toks, ner_labs, rels,
                            tokenizer, bio2ix, ne2ix, rel2ix,
                            max_len,
                            cls_tok='[CLS]',
                            sep_tok='[SEP]',
                            pad_tok='[PAD]',
                            pad_id=0,
                            pad_mask_id=0,
                            pad_lab_id=-1):
    doc_toks, doc_attn_masks, doc_labs = [], [], []
    doc_tail_masks, doc_tail_labs, doc_head_masks, doc_head_labs, doc_rel_labs = [], [], [], [], []

    for sent_toks, sent_labs, sent_rels in zip(ner_toks, ner_labs, rels):

        sbw_sent_toks = tokenizer.tokenize(' '.join(sent_toks))
        sbw_sent_labs = utils.match_sbp_label(sbw_sent_toks, sent_labs)
        sbw_sent_tok_padded = utils.padding_1d(
            [cls_tok] + sbw_sent_toks + [sep_tok],
            max_len + 2,
            pad_tok=pad_tok)

        sbw_sent_labs_padded = utils.padding_1d(
            [pad_lab_id] + [bio2ix[lab] for lab in sbw_sent_labs] + [pad_lab_id],
            max_len + 2,
            pad_tok=pad_lab_id
        )

        sbw_sent_attn_mask_padded = utils.padding_1d(
            [1] * len([cls_tok] + sbw_sent_toks + [sep_tok]),
            max_len + 2,
            pad_tok=pad_mask_id
        )

        sbw_sent_tok_ids_padded = tokenizer.convert_tokens_to_ids(sbw_sent_tok_padded)

        #         sent_tail_masks, sent_tail_labs, sent_head_masks, sent_head_labs, sent_rel_labs = [], [], [], [], []
        for tail_ids, tail_lab, head_ids, head_lab, rel_lab in sent_rels:
            tail_mask = mask_one_entity(tail_ids)
            tail_mask += [pad_id] * (len(sent_toks) - len(tail_mask))
            sbw_tail_mask = match_bpe_mask(sbw_sent_toks, tail_mask)
            sbw_tail_mask_padded = utils.padding_1d(
                [pad_mask_id] + sbw_tail_mask + [pad_mask_id],
                max_len + 2,
                pad_tok=pad_mask_id
            )

            head_mask = mask_one_entity(head_ids)
            head_mask += [pad_id] * (len(sent_toks) - len(head_mask))
            sbw_head_mask = match_bpe_mask(sbw_sent_toks, head_mask)
            sbw_head_mask_padded = utils.padding_1d(
                [pad_mask_id] + sbw_head_mask + [pad_mask_id],
                max_len + 2,
                pad_tok=pad_mask_id
            )

            # print(tail_lab, head_lab, rel_lab)
            doc_toks.append(sbw_sent_tok_ids_padded)
            doc_attn_masks.append(sbw_sent_attn_mask_padded)
            doc_labs.append(sbw_sent_labs_padded)
            doc_tail_masks.append(sbw_tail_mask_padded)
            doc_tail_labs.append(ne2ix[tail_lab])
            doc_head_masks.append(sbw_head_mask_padded)
            doc_head_labs.append(ne2ix[head_lab])
            doc_rel_labs.append(rel2ix[rel_lab])

    doc_toks_t = torch.tensor(doc_toks)
    doc_attn_masks_t = torch.tensor(doc_attn_masks)
    doc_labs_t = torch.tensor(doc_labs)

    doc_tail_masks_t = torch.tensor(doc_tail_masks)
    doc_tail_labs_t = torch.tensor(doc_tail_labs)
    doc_head_masks_t = torch.tensor(doc_head_masks)
    doc_head_labs_t = torch.tensor(doc_head_labs)
    doc_rel_labs_t = torch.tensor(doc_rel_labs)

    # print(doc_toks_t.shape, doc_attn_masks_t.shape, doc_labs_t.shape)
    # print(doc_tail_masks_t.shape, doc_tail_labs_t.shape, doc_head_masks_t.shape, doc_head_labs_t.shape,
    #       doc_rel_labs_t.shape)

    return TensorDataset(
        doc_toks_t,
        doc_attn_masks_t,
        doc_labs_t,
        doc_tail_masks_t,
        doc_tail_labs_t,
        doc_head_masks_t,
        doc_head_labs_t,
        doc_rel_labs_t
    )


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
    if is_reported:
        print(report)
        print(micro_f1)
    return float(micro_f1)


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

    parser.add_argument("--save_model", default='checkpoints/rel', type=str,
                        help="save/load model dir")

    parser.add_argument("--batch_size", default=16, type=int,
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

    parser.add_argument("--ne_size", default=128, type=int,
                        help="size of name entity embedding")

    parser.add_argument("--save_best", default='f1', type=str,
                        help="save the best model, given dev scores (f1 or loss)")

    parser.add_argument("--save_step_interval", default=200, type=int,
                        help="save best model given a step interval")

    parser.add_argument("--neg_ratio", default=1.0, type=float,
                        help="negative sample ratio")

    args = parser.parse_args()

    print(args)

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, do_basic_tokenize=False)

    train_toks, train_labs, train_rels, bio2ix, ne2ix, rel2ix = extract_rel_data_from_mh_conll(args.train_file, args.neg_ratio)
    print(bio2ix)
    print(ne2ix)
    print(rel2ix)
    print('max sent len:', max_sents_len(train_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in train_rels]), max([len(sent_rels) for sent_rels in train_rels]))
    print()

    dev_toks, dev_labs, dev_rels, _, _, _ = extract_rel_data_from_mh_conll(args.dev_file, 1.0)
    print('max sent len:', max_sents_len(dev_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in dev_rels]), max([len(sent_rels) for sent_rels in dev_rels]))
    print()

    test_toks, test_labs, test_rels, _, _, _ = extract_rel_data_from_mh_conll(args.test_file, 1.0)
    print('max sent len:', max_sents_len(test_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in test_rels]), max([len(sent_rels) for sent_rels in test_rels]))
    print()

    max_len = max(
        max_sents_len(train_toks, tokenizer),
        max_sents_len(dev_toks, tokenizer),
        max_sents_len(test_toks, tokenizer)
    )

    train_dataset = convert_rels_to_tensors(train_toks, train_labs, train_rels, tokenizer, bio2ix, ne2ix, rel2ix, max_len)
    dev_dataset = convert_rels_to_tensors(dev_toks, dev_labs, dev_rels, tokenizer, bio2ix, ne2ix, rel2ix, max_len)
    test_dataset = convert_rels_to_tensors(test_toks, test_labs, test_rels, tokenizer, bio2ix, ne2ix, rel2ix, max_len)

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

            if (step % args.save_step_interval == 0) or (step == num_epoch_steps):
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
