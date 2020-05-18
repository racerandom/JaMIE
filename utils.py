import os
from collections import defaultdict, Counter
import copy
import json
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

DEUNK_COL = 0
TOK_COL = 1
NER_COL = 2
CERT_COL = 3
TYPE_COL = 4
STAT_COL = 5


def get_label2ix(y_data, default=None, ignore_lab=None):
    label2ix = default if default is not None else {}
    for line in y_data:
        for label in line:
            if label not in label2ix and label != ignore_lab:
                label2ix[label] = len(label2ix)
    return label2ix


def padding_1d(seq_1d, max_len, pad_tok=None, direct='right'):
    tmp_seq_1d = copy.deepcopy(seq_1d)
    for i in range(0, max_len - len(tmp_seq_1d)):
        if direct in ['right']:
            tmp_seq_1d.append(pad_tok)
        else:
            tmp_seq_1d.insert(0, pad_tok)
    return tmp_seq_1d


def padding_2d(seq_2d, max_len, pad_tok=0, direct='right'):
    tmp_seq_2d = copy.deepcopy(seq_2d)
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


def return_index_of_last_one(mask):
    last_id = None
    for i in range(len(mask)):
        if mask[i] == 1:
            last_id = i
    return last_id


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


def convert_clinical_data_to_relconll(clinical_file, fo, tokenizer, sent_tag=True, defaut_cert='_', is_raw=False):
    from collections import defaultdict
    x_data, y_data, sent_stat = [], [], []
    filename, extension = os.path.splitext(clinical_file)
    rel_file = filename + '.rel'
    rel_dic = defaultdict(lambda: [[], []])
    with open(rel_file, 'r') as rfi:
        for line in rfi:
            if not line.strip():
                continue
            tail_tid, head_tid, rel = eval(line)
            rel_dic[tail_tid][0].append(head_tid)
            rel_dic[tail_tid][1].append(rel)

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
                        tag2mask = {}
                        st = ET.fromstring(line)
                        toks, labs, cert_labs, ttype_labs, state_labs = [], [], [], [], []
                        for item in st.iter():
                            if item.text is not None:
                                seg = juman.analysis(item.text)
                                toks += [w.midasi for w in seg.mrph_list()]
                                if item.tag in ['event', 'TIMEX3',
                                                'd', 'a', 'f', 'c', 'C', 't', 'r',
                                                'm-key', 'm-val', 't-test', 't-key', 't-val', 'cc']:
                                    tag2mask[item.attrib['tid']] = [0] * len(labs) + [1] * len(seg)
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

                        # calculate tag mask only in the current sentence
                        for tid, tag_mask in tag2mask.items():
                            tag2mask[tid] += [0] * (len(toks) - len(tag2mask[tid]))
                            # tag2mask[tid] = match_bpe_mask(deunk_toks, tag2mask[tid])
                        tok_tail_list = [str(i) for i in range(len(toks))]
                        tok_head_list = ["[%i]" % i for i in range(len(toks))]
                        tok_rel_list = ["['N']" for _ in toks]
                        for tail_tid, (head_tids, rels) in rel_dic.items():
                            if tail_tid in tag2mask:
                                tail_id = return_index_of_last_one(tag2mask[tail_tid])
                                head_list, rel_list = [], []
                                for head_tid, rel in zip(head_tids, rels):
                                    if head_tid in tag2mask:
                                        head_list.append(return_index_of_last_one(tag2mask[head_tid]))
                                        rel_list.append(rel)
                                if head_list and rel_list:
                                    tok_head_list[tail_id] = str(head_list)
                                    tok_rel_list[tail_id] = str(rel_list)
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

                    assert len(sbp_toks) == len(deunk_toks) == len(sbp_labs) == len(sbp_cert_labs) == len(
                        sbp_ttype_labs) == len(sbp_state_labs)

                    # for d, t, l, cl, tl, sl in zip(deunk_toks, sbp_toks, sbp_labs, sbp_cert_labs, sbp_ttype_labs,
                    #                                sbp_state_labs):
                    #     fo.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (d, t, l, cl, tl, sl))
                    # fo.write('\n')
                    fo.write('#doc\n')
                    for i, t, l, r, h in zip(tok_tail_list, toks, labs, tok_rel_list, tok_head_list):
                        fo.write('%s\t%s\t%s\t%s\t%s\n' % (i, t, l, r, h))
                    # fo.write('\n')
                except Exception as ex:

                    print('[error]' + clinical_file + ': ' + line)
                    print(ex)
    #     return index + 1
    return sent_stat


def batch_convert_clinical_data_to_relconll(
    file_list, file_out, tokenizer,
    is_separated=True,
    sent_tag=True,
    defaut_cert='_',
    is_raw=False
):
    doc_stat = []
    with open(file_out, 'w') as fo:
        for file in file_list:
            file_ext = ".xml" if sent_tag else ".txt"
            if file.endswith(file_ext):
                try:
                    doc_stat.append(convert_clinical_data_to_relconll(
                        file, fo, tokenizer, sent_tag=sent_tag,
                        defaut_cert=defaut_cert,
                        is_raw=is_raw
                    ))
                except Exception as ex:
                    print('[error]:' + file)
                    print(ex)
    return doc_stat


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
                                if item.tag in ['event', 'TIMEX3', 'timex3'
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

    doc_stat = []
    if is_separated:
        for dir_file_in in file_list:
            file_ext = ".xml" if sent_tag else ".txt"
            dir_in, file_in = os.path.split(dir_file_in)
            if dir_file_in.endswith(file_ext):
                dir_file_out = os.path.join(dir_in, os.path.splitext(file_in)[0] + '.conll')
                with open(dir_file_out, 'w') as fo:
                    try:
                        doc_stat.append(convert_clinical_data_to_conll(
                            dir_file_in, fo, tokenizer, sent_tag=sent_tag,
                            defaut_cert=defaut_cert,
                            is_raw=is_raw
                        ))
                    except Exception as ex:
                        print('[error]:' + dir_file_in)
                        print(ex)
    else:
        with open(file_out, 'w') as fo:
            for dir_file in file_list:
                file_ext = ".xml" if sent_tag else ".txt"
                if dir_file.endswith(file_ext):
                    try:
                        doc_stat.append(convert_clinical_data_to_conll(
                            dir_file, fo, tokenizer, sent_tag=sent_tag,
                            defaut_cert=defaut_cert,
                            is_raw=is_raw
                        ))
                    except Exception as ex:
                        print('[error]:' + dir_file)
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


# clinical_rel.py related
import pandas as pd
import csv
import random

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
        bio, tag = toks[BIO_COL].split('-', 1) if len(toks[BIO_COL].split('-', 1)) > 1 else ('O', 'N')
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


def extract_rel_data_from_mh_conll(conll_file, down_neg, del_neg=False):
    conll_data = read_multihead_conll(conll_file)
    conll_sents = convert_multihead_conll_2d_to_3d(conll_data)
    ner_toks = [[tok[1] for tok in sent] for sent in conll_sents]
    ner_labs = [[tok[2] for tok in sent] for sent in conll_sents]

    rel_tuples = []
    for sent_id, sent in enumerate(conll_sents):
        sent_rels = extract_rels_from_conll_sent(sent, down_neg=down_neg)
        rel_tuples.append(sent_rels)
    assert len(ner_toks) == len(ner_labs) == len(rel_tuples)
    print('number of sents:', len(ner_toks))
    print('number of ne:', len([ner for sent_ner in ner_labs for ner in sent_ner if ner.startswith('B-')]))
    print(
        'pos rels:', len([rel for sent_rel in rel_tuples for rel in sent_rel if rel[-1] != 'N']),
        'neg rels:', len([rel for sent_rel in rel_tuples for rel in sent_rel if rel[-1] == 'N'])
    )
    bio2ix = get_label2ix(ner_labs)
    ne2ix = get_label2ix([[lab.split('-', 1)[-1] for lab in labs if '-' in lab] for labs in ner_labs])
    if del_neg:
        rel2ix = get_label2ix([eval(tok[3]) for sent in conll_sents for tok in sent], ignore_lab='N')
    else:
        rel2ix = get_label2ix([eval(tok[3]) for sent in conll_sents for tok in sent])

    return ner_toks, ner_labs, rel_tuples, bio2ix, ne2ix, rel2ix


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
        sbw_sent_labs = match_sbp_label(sbw_sent_toks, sent_labs)
        sbw_sent_tok_padded = padding_1d(
            [cls_tok] + sbw_sent_toks + [sep_tok],
            max_len + 2,
            pad_tok=pad_tok)

        sbw_sent_labs_padded = padding_1d(
            [pad_lab_id] + [bio2ix[lab] for lab in sbw_sent_labs] + [pad_lab_id],
            max_len + 2,
            pad_tok=pad_lab_id
        )

        sbw_sent_attn_mask_padded = padding_1d(
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
            sbw_tail_mask_padded = padding_1d(
                [pad_mask_id] + sbw_tail_mask + [pad_mask_id],
                max_len + 2,
                pad_tok=pad_mask_id
            )

            head_mask = mask_one_entity(head_ids)
            head_mask += [pad_id] * (len(sent_toks) - len(head_mask))
            sbw_head_mask = match_bpe_mask(sbw_sent_toks, head_mask)
            sbw_head_mask_padded = padding_1d(
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

    print(doc_toks_t.shape, doc_attn_masks_t.shape, doc_labs_t.shape)
    print(doc_tail_masks_t.shape, doc_tail_labs_t.shape, doc_head_masks_t.shape, doc_head_labs_t.shape,
          doc_rel_labs_t.shape)

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


# tok_id -> sbw_tok_id
def align_sbw_ids(sbw_sent_toks):
    aligned_ids = []
    for index, token in enumerate(sbw_sent_toks):
        if token.startswith("##"):
            aligned_ids[-1].append(index)
        else:
            aligned_ids.append([index])
    return aligned_ids


# clinical_mhs.py related
def convert_rels_to_mhs(ner_toks, ner_labs, rels,
                        tokenizer, bio2ix, ne2ix, rel2ix,
                        max_len,
                        cls_tok='[CLS]',
                        sep_tok='[SEP]',
                        pad_tok='[PAD]',
                        pad_id=0,
                        pad_mask_id=0,
                        pad_lab_id=0,
                        verbose=0):
    doc_toks, doc_attn_masks, doc_labs = [], [], []
    rel_count = 0
    print((len(ner_toks), max_len + 2, max_len + 2, len(rel2ix)))
    doc_matrix_rels = np.zeros((len(ner_toks), max_len + 2, max_len + 2, len(rel2ix)))
    doc_matrix_rels[:, :, :, rel2ix['N']] = 1
    doc_num = len(ner_toks)
    print("ready to preprocess...")
    for sent_id, (sent_toks, sent_labs, sent_rels) in enumerate(zip(ner_toks, ner_labs, rels)):
        sbw_sent_toks = tokenizer.tokenize(' '.join(sent_toks))
        sbw_sent_labs = match_sbp_label(sbw_sent_toks, sent_labs)
        sbw_sent_tok_padded = padding_1d(
            [cls_tok] + sbw_sent_toks + [sep_tok],
            max_len + 2,
            pad_tok=pad_tok)
        aligned_ids = align_sbw_ids([cls_tok] + sbw_sent_toks + [sep_tok])

        assert len(aligned_ids) == (len(sent_toks) + 2)
        # print(aligned_ids)
        sbw_sent_labs_padded = padding_1d(
            [bio2ix[lab] for lab in (['O'] + sbw_sent_labs + ['O'])],
            max_len + 2,
            pad_tok=pad_lab_id
        )

        sbw_sent_attn_mask_padded = padding_1d(
            [1] * (len(sbw_sent_toks) + 2),
            max_len + 2,
            pad_tok=pad_mask_id
        )

        sbw_sent_tok_ids_padded = tokenizer.convert_tokens_to_ids(sbw_sent_tok_padded)

        if verbose:
            print("sent_id: {}/{}".format(sent_id, doc_num))
            print(["{}: {}".format(index, tok) for index, tok in enumerate([cls_tok] + sbw_sent_toks + [sep_tok])])
            print(["{}: {}".format(index, lab) for index, lab in enumerate(['O'] + sbw_sent_labs + ['O'])])

        # align entity_ids in sent_rels
        for tail_ids, tail_lab, head_ids, head_lab, rel_lab in sent_rels:
            tail_last_id = aligned_ids[int(tail_ids[-1]) + 1][-1]  # with the begining [CLS] + 1
            head_last_id = aligned_ids[int(head_ids[-1]) + 1][-1]   # with the begining [CLS] + 1
            doc_matrix_rels[sent_id][tail_last_id][head_last_id][rel2ix[rel_lab]] = 1
            doc_matrix_rels[sent_id][tail_last_id][head_last_id][rel2ix['N']] = 0
            if verbose:
                print((tail_last_id, rel_lab, head_last_id))
            rel_count += 1
        if verbose:
            print()
        assert len(sbw_sent_tok_ids_padded) == len(sbw_sent_attn_mask_padded) == len(sbw_sent_labs_padded)
        doc_toks.append(sbw_sent_tok_ids_padded)
        doc_attn_masks.append(sbw_sent_attn_mask_padded)
        doc_labs.append(sbw_sent_labs_padded)
    print("ready to tensor list/numpy")
    doc_id_t = torch.tensor(list(range(len(ner_toks))))  # add sent_id into batch data
    doc_toks_t = torch.tensor(doc_toks)
    doc_labs_t = torch.tensor(doc_labs)
    doc_attn_masks_t = torch.tensor(doc_attn_masks)
    doc_matrix_rels_t = torch.from_numpy(doc_matrix_rels)

    print(doc_toks_t.shape, doc_attn_masks_t.shape, doc_labs_t.shape, doc_matrix_rels_t.shape)
    print("positive rel count:", rel_count)
    print()
    return TensorDataset(
        doc_id_t,
        doc_toks_t,
        doc_attn_masks_t,
        doc_labs_t,
        doc_matrix_rels_t
    )


# clinical_mhs.py related
def convert_rels_to_mhs_v2(
        ner_toks, ner_labs, rels,
        tokenizer, bio2ix, rel2ix,
        max_len,
        cls_tok='[CLS]',
        sep_tok='[SEP]',
        pad_tok='[PAD]',
        pad_id=0,
        pad_mask_id=0,
        pad_lab_id=0,
        verbose=0
):
    doc_tok, doc_attn_mask, doc_lab, doc_rel, doc_spo = [], [], [], [], []
    rel_count = 0
    cls_max_len = max_len + 2  # wrap data with two additional token [CLS] and [SEP]
    print((len(ner_toks), cls_max_len, cls_max_len, len(rel2ix)))
    doc_num = len(ner_toks)
    print("ready to preprocess...")
    for sent_id, (sent_toks, sent_labs, sent_rels) in enumerate(zip(ner_toks, ner_labs, rels)):

        # wrapping data with [CLS] and [SEP]
        sbw_sent_tok = tokenizer.tokenize(' '.join(sent_toks))
        sbw_sent_lab = match_sbp_label(sbw_sent_tok, sent_labs)

        cls_sbw_sent_tok = [cls_tok] + sbw_sent_tok + [sep_tok]
        cls_sbw_sent_lab = ['O'] + sbw_sent_lab + ['O']
        cls_sbw_sent_mask = [1] * len(cls_sbw_sent_tok)

        assert len(cls_sbw_sent_tok) == len(cls_sbw_sent_lab) == len([1] * len(cls_sbw_sent_mask))

        if verbose:
            print("sent_id: {}/{}".format(sent_id, doc_num))
            print(["{}: {}".format(index, tok) for index, tok in enumerate(cls_sbw_sent_tok)])
            print(["{}: {}".format(index, lab) for index, lab in enumerate(cls_sbw_sent_lab)])

        # preparing rel data
        sent_rel, sent_spo = [], []
        # align entity_ids in sent_rels
        cls_aligned_ids = align_sbw_ids(cls_sbw_sent_tok)
        assert len(cls_aligned_ids) == (len(sent_toks) + 2)
        for tail_ids, tail_lab, head_ids, head_lab, rel_lab in sent_rels:
            tail_last_id = cls_aligned_ids[int(tail_ids[-1]) + 1][-1]  # with the begining [CLS] + 1
            head_last_id = cls_aligned_ids[int(head_ids[-1]) + 1][-1]   # with the begining [CLS] + 1
            rel_item = (tail_last_id, head_last_id, rel_lab)
            sent_rel.append(rel_item)
            sbw_tail_tok = [cls_sbw_sent_tok[a_i] for o_i in tail_ids for a_i in cls_aligned_ids[int(o_i) + 1]]
            sbw_head_tok = [cls_sbw_sent_tok[a_i] for o_i in head_ids for a_i in cls_aligned_ids[int(o_i) + 1]]
            spo_item = {'subject': sbw_tail_tok, 'predicate': rel_lab, 'object': sbw_head_tok}
            sent_spo.append(spo_item)
            if verbose:
                print(rel_item)
                print(["{}: {}".format(ix, a_i) for ix, a_i in enumerate(cls_aligned_ids)])
                print((tail_ids, rel_lab, head_ids))
                print(spo_item)
            rel_count += 1
        if verbose:
            print()
        doc_tok.append(cls_sbw_sent_tok)
        doc_attn_mask.append(cls_sbw_sent_mask)
        doc_lab.append(cls_sbw_sent_lab)
        doc_rel.append(sent_rel)
        doc_spo.append(sent_spo)

    assert len(doc_tok) == len(doc_attn_mask) == len(doc_lab) == len(doc_rel)
    print("ready to tensor list/numpy")
    doc_ix_t = torch.tensor(list(range(len(ner_toks))))  # add sent_id into batch data

    # padding data to cls_max_len
    padded_doc_tok_ix_t = torch.tensor(
        [tokenizer.convert_tokens_to_ids(padding_1d(
                sent_tok,
                cls_max_len,
                pad_tok=pad_tok
        )) for sent_tok in doc_tok]
    )
    padded_doc_lab_ix_t = torch.tensor(
        [padding_1d(
            [bio2ix[lab] for lab in sent_lab],
            cls_max_len,
            pad_tok=pad_lab_id
        ) for sent_lab in doc_lab]
    )
    padded_doc_attn_mask_t = torch.tensor(
        [padding_1d(
            sent_mask,
            cls_max_len,
            pad_tok=pad_mask_id
        ) for sent_mask in doc_attn_mask]
    )

    print(padded_doc_tok_ix_t.shape, padded_doc_attn_mask_t.shape, padded_doc_lab_ix_t.shape, len(doc_rel))
    print("positive rel count:", rel_count)
    print()
    return TensorDataset(
        doc_ix_t,
        padded_doc_tok_ix_t,
        padded_doc_attn_mask_t,
        padded_doc_lab_ix_t,
    ), doc_tok, doc_lab, doc_rel, doc_spo


def gen_relmat(doc_rel, sent_ids, max_len, rel2ix, del_neg=False):
    if del_neg:
        relmat = torch.zeros(len(sent_ids), max_len, len(rel2ix), max_len)
        for b_id, sent_id in enumerate(sent_ids):
            for (tail_last_id, head_last_id, rel) in doc_rel[sent_id]:
                if rel != 'N':
                    relmat[b_id][tail_last_id][rel2ix[rel] - 1][head_last_id] = 1
        return relmat
    else:
        relmat = torch.zeros(len(sent_ids), max_len, len(rel2ix), max_len)
        relmat[:, :, rel2ix['N'], :] = 1
        for b_id, sent_id in enumerate(sent_ids):
            for (tail_last_id, head_last_id, rel) in doc_rel[sent_id]:
                if rel != 'N':
                    relmat[b_id][tail_last_id][rel2ix[rel]][head_last_id] = 1
                    relmat[b_id][tail_last_id][rel2ix['N']][head_last_id] = 0
        return relmat


def convert_rels_to_pmhs(ner_toks, ner_labs, rels,
                         tokenizer, rel2ix, out_file, word_vocab):
    print("writing pmhs out_file: {}".format(out_file))
    with open(out_file, 'w') as fo:
        for sent_id, (sent_toks, sent_labs, sent_rels) in enumerate(zip(ner_toks, ner_labs, rels)):
            sent_out = {}
            sbw_sent_toks = tokenizer.tokenize(' '.join(sent_toks))
            sbw_sent_labs = match_sbp_label(sbw_sent_toks, sent_labs)

            assert len(sbw_sent_toks) == len(sbw_sent_labs)
            aligned_ids = align_sbw_ids(sbw_sent_toks)
            word_vocab.update(sbw_sent_toks)
            sent_out["text"] = sbw_sent_toks
            sent_out["spo_list"] = []
            sent_out["bio"] = [lab.split('-')[0] for lab in sbw_sent_labs]
            sent_out["selection"] = []

            assert len(aligned_ids) == len(sent_toks)
            # print(aligned_ids)

            # align entity_ids in sent_rels
            for tail_ids, tail_lab, head_ids, head_lab, rel_lab in sent_rels:
                tail_last_sbw_id = aligned_ids[int(tail_ids[-1])][-1]  # with the begining [CLS] + 1
                head_last_sbw_id = aligned_ids[int(head_ids[-1])][-1]  # with the begining [CLS] + 1
                tail_sbw = [sbw_sent_toks[sbw_id] for tok_id in tail_ids for sbw_id in aligned_ids[int(tok_id)]]
                head_sbw = [sbw_sent_toks[sbw_id] for tok_id in head_ids for sbw_id in aligned_ids[int(tok_id)]]
                spo = {"subject": tail_sbw, "predicate": rel_lab, "object": head_sbw}
                sent_out["spo_list"].append(spo)
                selection = {"subject": tail_last_sbw_id, "predicate": rel2ix[rel_lab], "object": head_last_sbw_id}
                sent_out["selection"].append(selection)
            fo.write("{}\n".format(json.dumps(sent_out, ensure_ascii=False)))


def gen_vocab(word_vocab, vocab_file, min_freq=1):
    result = {'<pad>': 0}
    i = 1
    for k, v in word_vocab.items():
        if v > min_freq:
            result[k] = i
            i += 1
    result['oov'] = i
    json.dump(result, open(vocab_file, 'w'), ensure_ascii=False)


def calculate_f1(tps, fps, fns):
    p = 0. if not (tps + fps) else (tps / (tps + fps))
    r = 0. if not (tps + fns) else (tps / (tps + fns))
    f1 = 0. if not (p + r) else (2 * p * r / (p + r))
    return p, r, f1


def evaluate_tuples(pred_tuples, gold_tuples, ix2rel, rel_col=-1):
    # eval_dic[rel] = [tps, fps, fns]
    tps_id = 0
    fps_id = 1
    fns_id = 2
    eval_dic = defaultdict(lambda: [0, 0, 0])
    for g_t in gold_tuples:
        g_rel = ix2rel[g_t[rel_col]] if isinstance(g_t[rel_col], int) else g_t[rel_col]
        if g_rel in ['N', 'O']:
            continue
        if g_t in pred_tuples:
            eval_dic[g_rel][tps_id] += 1
            pred_tuples.remove(g_t)
        else:
            eval_dic[g_rel][fns_id] += 1
    for p_t in pred_tuples:
        p_rel = ix2rel[p_t[rel_col]] if isinstance(p_t[rel_col], int) else p_t[rel_col]
        if p_rel in ['N', 'O']:
            continue
        eval_dic[p_rel][fps_id] += 1

    print()
    for rel, (rel_tps, rel_fps, rel_fns) in eval_dic.items():
        p, r, f1 = calculate_f1(rel_tps, rel_fps, rel_fns)
        print("\t{:>12}, p {:.6f}, r {:.6f}, f1 {:.6f}, (tps {:d}, fps {:d}, fns {:d})".format(
            rel,
            p, r, f1,
            rel_tps, rel_fps, rel_fns
        ))

    all_tps = sum([v[tps_id] for v in eval_dic.values()])
    all_fps = sum([v[fps_id] for v in eval_dic.values()])
    all_fns = sum([v[fns_id] for v in eval_dic.values()])
    all_p, all_r, all_f1 = calculate_f1(all_tps, all_fps, all_fns)
    print("overall, p %.6f, r %.6f, f1 %.6f, (tps %i, fps %i, fns %i)\n" % (
        all_p, all_r, all_f1,
        all_tps, all_fps, all_fns
    ))


def sent_ner2tuple(sent_id, sent_ner):
    ent_output = []  # [[sent_id, ent_ids, ent_lab], ...]
    ent_cache = [[], 'O']  # [ent_ids, ent_lab]

    for tok_id, tok_ner in enumerate(sent_ner):
        if tok_ner.split('-')[0] in ['B']:
            if ent_cache[0]:
                ent_output.append(ent_cache)
            ent_cache = [[tok_id], '-'.join(tok_ner.split('-')[1:])]
        elif tok_ner.split('-')[0] in ['I']:
            if '-'.join(tok_ner.split('-')[1:]) == ent_cache[-1]:
                ent_cache[0].append(tok_id)
            else:
                if ent_cache[0]:
                    ent_output.append(ent_cache)
                ent_cache = [[tok_id], '-'.join(tok_ner.split('-')[1:])]
        elif tok_ner == 'O':
            if ent_cache[0]:
                ent_output.append(ent_cache)
            ent_cache = [[tok_id], 'O']
        else:
            raise Exception("[ERROR] Unknown ner label '%i' '%s'..." % (tok_id, tok_ner))
    if ent_cache[0]:
        ent_output.append(ent_cache)
    return [[sent_id] + tuple for tuple in ent_output]


def ner2tuple(sent_ids, ners):
    ner_tuples = []
    assert len(sent_ids) == len(ners)
    for sent_id, sent_ner in zip(sent_ids, ners):
        ner_tuples += sent_ner2tuple(sent_id, sent_ner)
    return ner_tuples


def freeze_bert_layers(model, bert_name='encoder', freeze_embed=False, layer_list=None):

    layer_prefixes = ["%s.encoder.layer.%i." % (bert_name, i) for i in layer_list]

    for n, p in list(model.named_parameters()):

        if freeze_embed:
            if n.startswith("%s.embeddings" % bert_name):
                p.requires_grad = False
        else:
            if n.startswith("%s.embeddings" % bert_name):
                p.requires_grad = True

        if any(n.startswith(prefix) for prefix in layer_prefixes):
            p.requires_grad = False
        else:
            p.requires_grad = True


class TupleEvaluator(object):
    # eval_dic[rel] = [tps, fps, fns]
    def __init__(self):
        self.tps_id = 0
        self.fps_id = 1
        self.fns_id = 2
        self.eval_dic = defaultdict(lambda: [1e-10, 1e-10, 1e-10])

    def reset(self):
        self.eval_dic = defaultdict(lambda: [1e-10, 1e-10, 1e-10])

    def update(self, gold_tuples, pred_tuples, rel_col=-1):
        for g_t in gold_tuples:
            g_rel = g_t[rel_col]
            if g_rel in ['N', 'O']:
                continue
            if g_t in pred_tuples:
                self.eval_dic[g_rel][self.tps_id] += 1
                pred_tuples.remove(g_t)
            else:
                self.eval_dic[g_rel][self.fns_id] += 1
        for p_t in pred_tuples:
            p_rel = p_t[rel_col]
            if p_rel in ['N', 'O']:
                continue
            self.eval_dic[p_rel][self.fps_id] += 1

    def print_results(self, message, print_details, print_general, f1_mode):

        class_scores = {}
        for rel, (rel_tps, rel_fps, rel_fns) in self.eval_dic.items():
            p, r, f1 = calculate_f1(rel_tps, rel_fps, rel_fns)
            class_scores[rel] = (p, r, f1)
            if print_details:
                print("\t{:>12}, p {:.6f}, r {:.6f}, f1 {:.6f}, (tps {:.0f}, fps {:.0f}, fns {:.0f})".format(
                    rel,
                    p, r, f1,
                    rel_tps, rel_fps, rel_fns
                ))

        if f1_mode == 'micro':
            all_tps = sum([v[self.tps_id] for v in self.eval_dic.values()])
            all_fps = sum([v[self.fps_id] for v in self.eval_dic.values()])
            all_fns = sum([v[self.fns_id] for v in self.eval_dic.values()])
            all_p, all_r, all_f1 = calculate_f1(all_tps, all_fps, all_fns)
        elif f1_mode == 'macro':
            all_p = sum([v[0] for k, v in class_scores.items()]) / len(class_scores)
            all_r = sum([v[1] for k, v in class_scores.items()]) / len(class_scores)
            all_f1 = sum([v[2] for k, v in class_scores.items()]) / len(class_scores)
        else:
            raise Exception("Unknown f1_model: {} ...".format(f1_mode))

        if print_general:
            print("{}, {}, overall, p {:.6f}, r {:.6f}, f1 {:.6f}\n".format(
                message, f1_mode,
                all_p, all_r, all_f1
            ))

        return all_f1
