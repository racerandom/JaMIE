#!/usr/bin/env python
# coding: utf-8
import os
import mojimoji
from pyknp import Juman
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import TensorDataset
from xml.sax.saxutils import escape
from textformatting import ssplit
from gensim.models import KeyedVectors
from transformers import *

juman = Juman()

DEUNK_COL=0
TOK_COL=1
NER_COL=2
CERT_COL=3
TYPE_COL=4
STAT_COL=5


def get_label2ix(y_data, default=None):
    label2ix = default if default is not None else {}
    for line in y_data:
        for label in line:
            if label not in label2ix:
                label2ix[label] = len(label2ix)
    return label2ix


def padding_1d(seq_1d, max_len, pad_tok=None, direct='right'):
    tmp_seq_1d = seq_1d.copy()
    for i in range(0, max_len - len(tmp_seq_1d)):
        if direct in ['right']:
            tmp_seq_1d.append(pad_tok)
        else:
            tmp_seq_1d.insert(0, pad_tok)
    return tmp_seq_1d


def padding_2d(seq_2d, max_len, pad_tok=0, direct='right'):
    tmp_seq_2d = seq_2d.copy()
    for seq_1d in tmp_seq_2d:
        for i in range(0, max_len - len(seq_1d)):
            if direct in ['right']:
                seq_1d.append(pad_tok)
            else:
                seq_1d.insert(0, pad_tok)
    return tmp_seq_2d


def match_sbp_label(bpe_x, y):
    bpe_y = y.copy()
    for i in range(len(bpe_x)):
        if bpe_x[i].startswith('##'):
            if '-' in bpe_y[i-1]:
                bpe_y.insert(i, 'I' + bpe_y[i-1][1:])
            else:
                bpe_y.insert(i, bpe_y[i-1])
    return bpe_y


def match_sbp_cert_labs(bpe_x, y):
    bpe_y = y.copy()
    for i in range(len(bpe_x)):
        if bpe_x[i].startswith('##'):
            bpe_y.insert(i, '_')
    return bpe_y


def explore_unk(bpe_x, ori_x):
    
    ix_count = 0
    deunk_bpe_x = []
    
    for tok in bpe_x:
        if not tok.startswith('##'):
            if tok != '[UNK]':
                deunk_bpe_x.append(tok)
            else:
                deunk_bpe_x.append(ori_x[ix_count])
            ix_count += 1
        else:
            deunk_bpe_x.append(tok)
    assert len(bpe_x) == len(deunk_bpe_x)
    return deunk_bpe_x


def write_data_to_txt(np_findings, file_name):
    with open(file_name, "w") as txt_file:
        for d in np_findings:
            if isinstance(d, str):
                txt_file.write(d + '\n')


def out_xml(orig_tok, pred_ix, ix2label):
    lines = []
    for sent_tok in orig_tok:
        label_prev = 'O'
        line_str = ''
        for tok in sent_tok:
            label = ix2label[pred_ix.pop()]
            if label_prev.startswith('O'):
                if label.startswith('B'):
                    line_str += '<%s>%s' % (label.split('-')[-1], tok)
                elif label.startswith('I'):
                    line_str += '<%s>%s' % (label.split('-')[-1], tok)
                else:
                    line_str += tok
            elif label_prev.startswith('B'):
                if label.startswith('B'):
                    line_str += '</%s><%s>%s' % (label_prev.split('-')[-1], label.split('-')[-1], tok)
                elif label.startswith('I'):
                    line_str += tok
                else:
                    line_str += '</%s>%s' % (label_prev.split('-')[-1], tok)
            elif label_prev.startswith('I'):
                if label.startswith('B'):
                    line_str += '</%s><%s>%s' % (label_prev.split('-')[-1], label.split('-')[-1], tok)
                elif label.startswith('I'):
                    line_str += tok
                else:
                    line_str += '</%s>%s' % (label_prev.split('-')[-1], tok)
            label_prev = label 
        lines.append(line_str)
        
    return lines


def retrieve_w2v(embed_file, binary=True, add_unk=True):
    # return: word2ix, weights
    w2v = KeyedVectors.load_word2vec_format(
        embed_file,
        binary=binary
    )
    weights = w2v.vectors
    word_list = w2v.index2word
    if add_unk:
        word_list.insert(0, '[UNK]')
        weights = np.insert(weights, 0, np.zeros(weights.shape[1]), 0)
    word2ix = {tok: tok_ix for tok_ix, tok in enumerate(word_list)}
    return word2ix, weights


def convert_clinical_data_to_conll(clinical_file, fo, tokenizer, sent_tag=True, defaut_cert='_', is_raw=False):
    x_data, y_data, sent_stat = [], [], []
    with open(clinical_file, 'r') as fi:
        for index, pre_line in enumerate(fi):
            # convert number&alphabet to han-kaku for spliting sentences
            pre_line = mojimoji.zen_to_han(pre_line, kana=False)
#             sent_list = ssplit(pre_line)
            if sent_tag:
                sent_list = [pre_line]
            else:
                sent_list = ssplit(pre_line)

            for line in sent_list:
                
                try:

                    line = line.strip().replace('\n', '').replace('\r', '')

                    line = line.replace('>>', '>＞').replace('<<', '＜<')

                    if is_raw:
                        line = line.replace('#', '＃')  # a solution to fix juman casting #
                        line = line.replace('<', '＜')
                        line = line.replace('>', '＞')
#                         print()
#                         print(line)
                    else:
                        if line in ['<CHEST: CT>',
                                    '<CHEST>',
                                    '<胸部CT>',
                                    '<CHEST；CT>',
                                    '<CHEST;CT>',
                                    '<胸部単純CT>',
                                    '<胸部CT>',
                                    '<ABD US>', '<Liver>', '<胸部CT>']:
                            continue

                    if sent_tag:
                        line = '<sentence>' + line + '</sentence>' 

                    if not is_raw:
                        st = ET.fromstring(line)
                        toks, labs, cert_labs, ttype_labs, state_labs = [], [], [], [], []
                        for item in st.iter():
                            if item.text is not None:
                                seg = juman.analysis(item.text)
                                toks += [w.midasi for w in seg.mrph_list()]
                                if item.tag in ['event', 'TIMEX3', 
                                                'd', 'a', 'f', 'c', 'C', 't', 'r',
                                                'm-key', 'm-val', 't-test', 't-key', 't-val', 'cc']:
                                    tok_labs = ['I-%s' % (item.tag.capitalize())] * len(seg)
                                    tok_labs[0] = 'B-%s' % (item.tag.capitalize())
                                    labs += tok_labs
                                    if item.tag == 'd' and 'certainty' in item.attrib:
                                        tok_cert_labs = ['_'] * len(seg)
                                        tok_cert_labs[0] = item.attrib['certainty']
                                        cert_labs += tok_cert_labs
                                    else:
                                        cert_labs += ['_'] * len(seg)

                                    if item.tag == 'TIMEX3' and 'type' in item.attrib:
                                        tok_ttype_labs = ['_'] * len(seg)
                                        tok_ttype_labs[0] = item.attrib['type']
                                        ttype_labs += tok_ttype_labs
                                    else:
                                        ttype_labs += ['_'] * len(seg)
                                    if item.tag in ['t-test', 'r', 'cc'] and 'state' in item.attrib:
                                        tok_state_labs = ['_'] * len(seg)
                                        tok_state_labs[0] = item.attrib['state']
                                        state_labs += tok_state_labs
                                    else:
                                        state_labs += ['_'] * len(seg)
                                else:
                                    if item.tag not in ['sentence', 'p']:
                                        print(item.tag)
                                    labs += ['O'] * len(seg)
                                    cert_labs += ['_'] * len(seg)
                                    ttype_labs += ['_'] * len(seg)
                                    state_labs += ['_'] * len(seg)
                            if item.tail is not None:
                                seg_tail = juman.analysis(item.tail)
                                toks += [w.midasi for w in seg_tail.mrph_list()]
                                labs += ['O'] * len(seg_tail)
                                cert_labs += ['_'] * len(seg_tail)
                                ttype_labs += ['_'] * len(seg_tail)
                                state_labs += ['_'] * len(seg_tail)
                        assert len(toks) == len(labs) == len(cert_labs) == len(ttype_labs) == len(state_labs)

                        sent_stat.append(len(toks))
                        # replace '\u3000' to '[JASP]' 
                        toks = ['[JASP]' if t == '\u3000' else mojimoji.han_to_zen(t) for t in toks]
                        sbp_toks = tokenizer.tokenize(' '.join(toks)) if tokenizer else toks
                        deunk_toks = explore_unk(sbp_toks, toks)
                        sbp_labs = match_sbp_label(deunk_toks, labs)
                        sbp_cert_labs = match_sbp_cert_labs(deunk_toks, cert_labs)
                        sbp_ttype_labs = match_sbp_cert_labs(deunk_toks, ttype_labs)
                        sbp_state_labs = match_sbp_cert_labs(deunk_toks, state_labs)
                    else:
                        print(line)
                        seg = juman.analysis(line)
                        toks = [w.midasi for w in seg.mrph_list()]
                        print(toks)
                        print()
                        sent_stat.append(len(toks))
                        # replace '\u3000' to '[JASP]' 
                        toks = ['[JASP]' if t == '\u3000' else mojimoji.han_to_zen(t) for t in toks]
                        sbp_toks = tokenizer.tokenize(' '.join(toks)) if tokenizer else toks
                        deunk_toks = explore_unk(sbp_toks, toks)
                        sbp_labs = ['O'] * len(sbp_toks)
                        sbp_cert_labs = ['_'] * len(sbp_toks)
                        sbp_ttype_labs = ['_'] * len(sbp_toks)
                        sbp_state_labs = ['_'] * len(sbp_toks)
                    
                    assert len(sbp_toks) == len(deunk_toks) == len(sbp_labs) == len(sbp_cert_labs) == len(sbp_ttype_labs) == len(sbp_state_labs)
                    
                    for d, t, l, cl, tl, sl in zip(deunk_toks, sbp_toks, sbp_labs, sbp_cert_labs, sbp_ttype_labs, sbp_state_labs):
                        fo.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (d, t, l, cl, tl, sl))
                    fo.write('\n')
                except Exception as ex:

                    print('[error]' + clinical_file + ': ' + line)
                    print(ex)
#     return index + 1
    return sent_stat
                
            
def batch_convert_clinical_data_to_conll(
    file_list, file_out, tokenizer,
    is_separated=True,
    sent_tag=True,
    defaut_cert='_',
    is_raw=False
):
    # print(tokenizer)

    # file_list = os.listdir(data_dir)

    doc_stat = []
    # if is_separated:
    #     for file in file_list:
    #         file_ext = ".xml" if sent_tag else ".txt"
    #         if file.endswith(file_ext):
    #             file_out = os.path.join(data_dir, os.path.splitext(file)[0] + '.conll', )
    #             with open(file_out, 'w') as fo:
    #                 try:
    #                     doc_stat.append(convert_clinical_data_to_conll(
    #                         dir_file, fo, tokenizer, sent_tag=sent_tag,
    #                         defaut_cert=defaut_cert,
    #                         is_raw=is_raw
    #                     ))
    #                 except Exception as ex:
    #                     print('[error]:' + file)
    #                     print(ex)
    # else:
    with open(file_out, 'w') as fo:
        for file in file_list:
            file_ext = ".xml" if sent_tag else ".txt"
            if file.endswith(file_ext):
                try:
                    doc_stat.append(convert_clinical_data_to_conll(
                        file, fo, tokenizer, sent_tag=sent_tag,
                        defaut_cert=defaut_cert,
                        is_raw=is_raw
                    ))
                except Exception as ex:
                    print('[error]:' + file)
                    print(ex)
    return doc_stat
                
                
def read_conll(conll_file, is_merged=False):

    def merge_modality(ner_lab, cert_lab, ttype_lab, stat_lab):
        out_lab = ner_lab
        for lab in [cert_lab, ttype_lab, stat_lab]:
            if lab != '_':
                out_lab = "%s_%s" % (out_lab, lab)
        return out_lab

    deunks, toks, labs, cert_labs, ttype_labs, state_labs = [], [], [], [], [], []
    with open(conll_file) as fi:
        sent_deunks, sent_toks, sent_labs, sent_cert_labs, sent_ttype_labs, sent_state_labs = [], [], [], [], [], []
        for line in fi:
            line = line.rstrip()
            if not line:
                if sent_deunks:
                    deunks.append(sent_deunks)
                    toks.append(sent_toks)
                    labs.append(sent_labs)
                    cert_labs.append(sent_cert_labs)
                    ttype_labs.append(sent_ttype_labs)
                    state_labs.append(sent_state_labs)
                    sent_deunks, sent_toks, sent_labs, sent_cert_labs, sent_ttype_labs, sent_state_labs = [], [], [], [], [], []
                continue
            # print('line:', line)
            items = line.split('\t')
            sent_deunks.append(items[DEUNK_COL])
            sent_toks.append(items[TOK_COL])
            sent_labs.append(
                items[NER_COL] if not is_merged else merge_modality(
                    items[NER_COL],
                    items[CERT_COL],
                    items[TYPE_COL],
                    items[STAT_COL]
                )
            )
            sent_cert_labs.append(items[CERT_COL])
            sent_ttype_labs.append(items[TYPE_COL])
            sent_state_labs.append(items[STAT_COL])
    return deunks, toks, labs, cert_labs, ttype_labs, state_labs


# unfinished
def read_conll_v2(conll_file):
    doc_cols = []
    with open(conll_file) as fi:
        sent_cols = []
        for line in fi:
            line = line.rstrip()
            if not line:
                if sent_cols[0]:
                    for i in range(len(sent_cols)):
                        doc_cols[str(i)].append(sent_cols[str(i)])
                continue
            toks = line.split('\t')
            for j in range(len(toks)):
                sent_cols[str(j)].append(toks[j])
    return tuple()


def extract_ner_from_conll(conll_file, tokenizer, lab2ix, device, is_merged=False):
    deunks, toks, labs, cert_labs, ttype_labs, state_labs = read_conll(conll_file, is_merged=is_merged)
    max_len = max([len(x) for x in toks])
    pad_tok_ids, pad_masks, pad_lab_ids = [], [], []
    for tok, lab in zip(toks, labs):
        pad_tok = padding_1d(['[CLS]'] + tok, max_len + 1, pad_tok='[PAD]')
        pad_tok_id = tokenizer.convert_tokens_to_ids(pad_tok)
        pad_mask = padding_1d([1] * (len(tok) + 1), max_len + 1, pad_tok=0)
        pad_lab = padding_1d(['O'] + lab, max_len + 1, pad_tok='O')
        pad_lab_id = [lab2ix[lab] for lab in pad_lab]
        assert len(pad_tok_id) == len(pad_mask) == len(pad_lab_id)
        pad_tok_ids.append(pad_tok_id)
        pad_masks.append(pad_mask)
        pad_lab_ids.append(pad_lab_id)
    pad_tok_ids_t = torch.tensor(pad_tok_ids).to(device)
    pad_masks_t = torch.tensor(pad_masks).to(device)
    pad_lab_ids_t = torch.tensor(pad_lab_ids).to(device)
    print('ner data size:',
          pad_tok_ids_t.shape,
          pad_masks_t.shape,
          pad_lab_ids_t.shape)
    return TensorDataset(pad_tok_ids_t,
                         pad_masks_t.bool(),
                         pad_lab_ids_t), deunks


def extract_ner_from_conll_w2v(conll_file, word2ix, lab2ix, device):
    deunks, toks, labs, cert_labs, ttype_labs, state_labs = read_conll(conll_file)
    max_len = max([len(x) for x in toks])
    pad_tok_ids, pad_masks, pad_lab_ids = [], [], []
    for tok, lab in zip(toks, labs):
        pad_tok = padding_1d(tok, max_len, pad_tok='[UNK]')
        pad_tok_id = [word2ix[t] if t in word2ix else word2ix['[UNK]'] for t in pad_tok]
        pad_mask = padding_1d([1] * (len(tok)), max_len, pad_tok=0)
        pad_lab = padding_1d(lab, max_len, pad_tok='O')
        pad_lab_id = [lab2ix[lab] for lab in pad_lab]
        assert len(pad_tok_id) == len(pad_mask) == len(pad_lab_id)
        pad_tok_ids.append(pad_tok_id)
        pad_masks.append(pad_mask)
        pad_lab_ids.append(pad_lab_id)
    pad_tok_ids_t = torch.tensor(pad_tok_ids).to(device)
    pad_masks_t = torch.tensor(pad_masks).to(device)
    pad_lab_ids_t = torch.tensor(pad_lab_ids).to(device)
    print('ner data size:',
          pad_tok_ids_t.shape,
          pad_masks_t.shape,
          pad_lab_ids_t.shape)
    return TensorDataset(pad_tok_ids_t,
                         pad_masks_t.bool(),
                         pad_lab_ids_t), deunks


def mask_ner_label(ner_labels, ner_masks, ner_cert_labels, attrib_tag, cert_labels, ner_offset):

    if not cert_labels:
        prev_label = 'O'
        for i, curr_label in enumerate(ner_labels):
            if i > 0:
                prev_label = ner_labels[i - 1]
            if curr_label in ['B-' + attrib_tag]:
                if prev_label in ['B-' + attrib_tag, 'I-' + attrib_tag]:
                    ner_offset += 1
                ner_masks[ner_offset][i] = 1
            elif curr_label in ['I-' + attrib_tag]:
                if prev_label in ['B-' + attrib_tag, 'I-' + attrib_tag]:
                    ner_masks[ner_offset][i] = 1
                else:
                    ner_masks[ner_offset][i] = 1
            else:
                if prev_label in ['B-' + attrib_tag, 'I-' + attrib_tag]:
                    ner_offset += 1
    else:
        prev_label = 'O'
        prev_cert_label = '_'
        for i, (curr_label, curr_cert_label) in enumerate(zip(ner_labels, cert_labels)):
            if i > 0:
                prev_label = ner_labels[i - 1]
                prev_cert_label = cert_labels[i - 1]
            if curr_label in ['B-' + attrib_tag]:
                if prev_label in ['B-' + attrib_tag, 'I-' + attrib_tag]:
                    ner_offset += 1
                ner_masks[ner_offset][i] = 1
                ner_cert_labels[ner_offset] = curr_cert_label
            elif curr_label in ['I-' + attrib_tag]:
                if prev_label in ['B-' + attrib_tag, 'I-' + attrib_tag]:
                    ner_masks[ner_offset][i] = 1
                else:
                    ner_masks[ner_offset][i] = 1
                    ner_cert_labels[ner_offset] = curr_cert_label
            else:
                if prev_label in ['B-' + attrib_tag, 'I-' + attrib_tag]:
                    ner_offset += 1
    return ner_offset


def ner_labels_to_masks(ner_labels, max_ner_num, attrib, cert_labels):

    ner_masks = np.zeros((max_ner_num, len(ner_labels)), dtype=int)
    ner_cert_labels = ['_'] * max_ner_num

    if attrib == 'cert':
        tags = ['D']
    elif attrib == 'ttype':
        tags = ['Timex3']
    elif attrib == 'state':
        tags = ['T-test', 'R', 'Cc']
    else:
        raise Exception("[ERROR] wrong attrib...")

    ner_offset = 0
    for tag in tags:
        ner_offset = mask_ner_label(ner_labels, ner_masks, ner_cert_labels, tag, cert_labels, ner_offset)

    cert_label_masks = padding_1d([1] * len(set(ner_masks.nonzero()[0])), max_ner_num, pad_tok=0)
    
    return ner_masks, cert_label_masks, ner_cert_labels


def extract_cert_from_conll(conll_file, tokenizer, attrib_lab2ix, device, max_ner_num=8, attrib='cert', test_mode=False):
    deunks, toks, labs, clabs, tlabs, slabs = read_conll(conll_file)
    
    max_len = max([len(x) for x in toks])

    if attrib == 'cert':
        attrib_labs = clabs
    elif attrib == 'ttype':
        attrib_labs = tlabs
    elif attrib == 'state':
        attrib_labs = slabs
    else:
        raise Exception("Error: wrong task...!")
    # empirical set max_ner_num
    # max_ner_num = max([s_l.count('B-D') for s_l in labs])
    
    pad_tok_ids, pad_masks, pad_ner_masks, clab_masks, pad_cert_lab_ids = [], [], [], [], []
    for s_toks, s_labs, s_clabs in zip(toks, labs, attrib_labs):
        pad_s_toks = padding_1d(['[CLS]'] + s_toks, max_len + 1, pad_tok='[PAD]')
        pad_s_tok_ids = tokenizer.convert_tokens_to_ids(pad_s_toks)
        pad_s_masks = padding_1d([1] * (len(s_toks) + 1), max_len + 1, pad_tok=0)
        
        if test_mode:
            s_ner_masks, s_clab_masks, s_ner_clabs = ner_labels_to_masks(s_labs, max_ner_num, attrib, None)
        else:
            s_ner_masks, s_clab_masks, s_ner_clabs = ner_labels_to_masks(s_labs, max_ner_num, attrib, s_clabs)

        pad_s_ner_masks = padding_2d(np.insert(s_ner_masks, 0, 0, axis=1).tolist(), max_len + 1, pad_tok=0)

        pad_tok_ids.append(pad_s_tok_ids)
        pad_masks.append(pad_s_masks)
        pad_ner_masks.append(pad_s_ner_masks)
        clab_masks.append(s_clab_masks)
        pad_cert_lab_ids.append([attrib_lab2ix[clab] for clab in s_ner_clabs])

        assert len(pad_tok_ids) == len(pad_masks)
        assert len(pad_ner_masks) == len(clab_masks) == len(pad_cert_lab_ids)
    
    pad_tok_ids_t = torch.tensor(pad_tok_ids).to(device)
    pad_masks_t = torch.tensor(pad_masks).to(device)
    pad_ner_masks_t = torch.tensor(pad_ner_masks).to(device)
    clab_masks_t = torch.tensor(clab_masks).to(device)
    pad_clab_ids_t = torch.tensor(pad_cert_lab_ids).to(device)
    print('cert data size:', 
          pad_tok_ids_t.shape, 
          pad_masks_t.shape, 
          pad_ner_masks_t.shape, 
          clab_masks_t.shape,
          pad_clab_ids_t.shape)
    
    return TensorDataset(pad_tok_ids_t, 
                         pad_masks_t, 
                         pad_ner_masks_t, 
                         clab_masks_t, 
                         pad_clab_ids_t
                         ), deunks


def doc_kfold(data_dir, cv=5, dev_ratio=0.1, random_seed=1029):
    from sklearn.model_selection import KFold, train_test_split
    file_list, file_splits = [], []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".xml"):
            dir_file = os.path.join(data_dir, file)
            file_list.append(dir_file)
    print("[Number] %i files in '%s'" % (len(file_list), data_dir))
    gss = KFold(n_splits=cv, shuffle=True, random_state=random_seed)
    for raw_train_split, test_split in gss.split(file_list):
        if dev_ratio:
            train_split, dev_split = train_test_split(
                raw_train_split,
                test_size=dev_ratio,
                shuffle=False,
                random_state=random_seed
            )
        else:
            train_split = raw_train_split
            dev_split = []
        file_splits.append((
            [file_list[fid] for fid in train_split],
            [file_list[fid] for fid in dev_split],
            [file_list[fid] for fid in test_split]
        ))
    return file_splits


def eval_pid_seq(model, tokenizer, test_data, orig_token, label2ix, epoch):
    lines = []
    model.eval()
    with torch.no_grad():
        with open('output_ep%i.txt' % epoch, 'w') as fo:
            for (token, mask, gold), ti in zip(test_data, orig_token):
                pred_prob = model(token, attention_mask=mask)[0]
                pred = torch.argmax(pred_prob, dim=-1)

                t_masked_ix = torch.masked_select(token[:, 1:], mask[:, 1:].bool())
                pred_masked_ix = torch.masked_select(pred[:, 1:], mask[:, 1:].bool())
                gold_masked_ix = torch.masked_select(gold[:, 1:], mask[:, 1:].bool())
                
                ix2label = {v: k for k, v in label2ix.items()}
                
                bpe_tok = [tokenizer.convert_ids_to_tokens([ix])[0] for ix in t_masked_ix.tolist()]
                
                flat_ori_tok = [item for sublist in ti for item in sublist]
                   
                deunk_bpe_tok = explore_unk(bpe_tok, flat_ori_tok)
                                
                for t, g, p in zip(deunk_bpe_tok, gold_masked_ix.tolist(), pred_masked_ix.tolist()):
                    fo.write('%s\t%s\t%s\n' % (t, ix2label[g], ix2label[p]))
                fo.write('EOR\tO\tO\n')


def batch_demask(batch_tokens, batch_masks):
    # batch_tokens: batch_size x token_length
    demasked_seq_t = torch.masked_select(batch_tokens, batch_masks)
    batch_lens = batch_masks.sum(-1)
    sent_begin = 0
    demask_l = []
    for sent_len in batch_lens:
        demask_l.append(demasked_seq_t[sent_begin: sent_begin + sent_len].tolist())
        sent_begin += sent_len
    return demask_l


def split_merged(merged_tag, delimiter='_'):
    items = merged_tag.split(delimiter)
    if len(items) > 1:
        return ''.join(items[0:-1]), items[-1]
    else:
        return merged_tag, '_'


# save bert model
def save_bert(model, tokenizer, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


# Evaluate the non-crf ner model
def eval_seq(model, tokenizer, test_dataloader, deunk_toks, label2ix, file_out, is_merged):

    ix2lab = {v: k for k, v in label2ix.items()}
    dir_name, file_name = os.path.split(file_out)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    model.eval()
    with torch.no_grad():
        with open(file_out, 'w') as fe:
            for batch_deunk, (batch_token_ix, batch_mask, batch_gold) in zip(deunk_toks, test_dataloader):

                pred_prob = model(batch_token_ix, attention_mask=batch_mask)[0]
                pred_ix = torch.argmax(pred_prob, dim=-1)

                t_masked_ix = batch_demask(batch_token_ix[:, 1:], batch_mask[:, 1:].bool())
                pred_masked_ix = batch_demask(pred_ix[:, 1:], batch_mask[:, 1:].bool())
                gold_masked_ix = batch_demask(batch_gold[:, 1:], batch_mask[:, 1:].bool())

                # batch_token = [tokenizer.convert_ids_to_tokens([ix])[0] for ix in t_masked_ix]
                batch_token = [[tokenizer.convert_ids_to_tokens([ix])[0] for ix in sent_ix] for sent_ix in t_masked_ix]

                for sent_deunk, sent_token in zip(batch_deunk, batch_token):
                    assert len(sent_deunk) == len(sent_token)

                for sent_deunk, sent_token, sent_gold_ix, sent_pred_ix in zip(
                        batch_deunk,
                        batch_token,
                        gold_masked_ix,
                        pred_masked_ix
                ):
                    for tok_deunk, tok, tok_gold, tok_pred in zip(sent_deunk, sent_token, sent_gold_ix, sent_pred_ix):
                        gold_ner, gold_modality = split_merged(ix2lab[tok_gold])
                        pred_ner, pred_modality = split_merged(ix2lab[tok_pred])
                        fe.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (
                            tok_deunk,
                            tok,
                            gold_modality,
                            pred_modality,
                            gold_ner,
                            pred_ner
                        ))
                    fe.write('\n')


# Evaluate crf ner model
def eval_crf(model, tokenizer, test_dataloader, test_deunk_loader, label2ix, file_out, is_merged):

    ix2lab = {v: k for k, v in label2ix.items()}
    dir_name, file_name = os.path.split(file_out)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    model.eval()
    with torch.no_grad():
        with open(file_out, 'w') as fo:
            for batch_deunk, (batch_tok_ix, batch_mask, batch_gold) in zip(test_deunk_loader, test_dataloader):
                pred_ix = [l[1:] for l in model.decode(batch_tok_ix, batch_mask)]
                gold_masked_ix = batch_demask(batch_gold[:, 1:], batch_mask[:, 1:].bool())
                if not batch_deunk:
                    continue
                for sent_deunk, sent_gold_ix, sent_pred_ix in zip(batch_deunk, gold_masked_ix, pred_ix):
                    assert len(sent_deunk) == len(sent_gold_ix) == len(sent_pred_ix)
                tok_masked_ix = batch_demask(batch_tok_ix[:, 1:], batch_mask[:, 1:].bool())
                batch_bpe = [[tokenizer.convert_ids_to_tokens([ix])[0] for ix in sent_ix] for sent_ix in tok_masked_ix]

                for sent_deunk, sent_tok, sent_gold_ix, sent_pred_ix in zip(
                        batch_deunk,
                        batch_bpe,
                        gold_masked_ix,
                        pred_ix
                ):
                    for tok_deunk, tok, tok_gold, tok_pred in zip(sent_deunk, sent_tok, sent_gold_ix, sent_pred_ix):
                        gold_ner, gold_modality = split_merged(ix2lab[tok_gold])
                        pred_ner, pred_modality = split_merged(ix2lab[tok_pred])
                        fo.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (
                            tok_deunk,
                            tok,
                            gold_modality,
                            pred_modality,
                            gold_ner,
                            pred_ner
                        ))
                    fo.write('\n')


def measure_modality_fscore(gold_tags, pred_tags):
    from collections import defaultdict
    counts = defaultdict(lambda: {'g': 0.0, 'p': 0.0, 'c': 0.0})
    for g, p in zip(gold_tags, pred_tags):
        counts[g]['g'] += 1
        counts[p]['p'] += 1
        if g == p:
            counts[g]['c'] += 1
    for k, v in counts.items():
        if k == '_':
            continue
        precision = (v['c'] / v['p']) if v['p'] != 0.0 else 0.0
        recall = v['c'] / v['g'] if v['g'] != 0.0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) != 0.0 else 0.0
        print('Modality Tag: %s, precision: %.2f%%, recall: %.2f%%, f1: %.2f' % (k, 100*precision, 100*recall, 100*f1))
    certainty_labs = ['positive', 'negative', 'suspicious', 'general']
    type_labs = ['DATE', 'DURATION', 'AGE', 'CC', 'SET', 'TIME']
    state_labs = ['executed', 'scheduled', 'negated', 'other']
    print(sum([v['c'] for k, v in counts.items() if k in certainty_labs]) / sum(
        [v['g'] for k, v in counts.items() if k in certainty_labs]))
    print(sum([v['c'] for k, v in counts.items() if k in type_labs]) / sum(
        [v['g'] for k, v in counts.items() if k in type_labs]))
    print(sum([v['c'] for k, v in counts.items() if k in state_labs]) / sum(
        [v['g'] for k, v in counts.items() if k in state_labs]))


def eval_modality(file_out):
    gold_tags, pred_tags = [], []
    with open(file_out, 'r', encoding='utf8') as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            items = line.split()
            gold_tags.append(items[2])
            pred_tags.append(items[3])
    measure_modality_fscore(gold_tags, pred_tags)
