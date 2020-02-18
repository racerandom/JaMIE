#!/usr/bin/env python
# coding: utf-8
import os, sys
import mojimoji
from pyknp import Juman
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import TensorDataset
from xml.sax.saxutils import escape
from textformatting import ssplit

juman = Juman()


def get_label2ix(y_data, default=False):
    label2ix = {}
    for line in y_data:
        for label in line:
            if label not in label2ix:
                label2ix[label] = len(label2ix)
    return label2ix


def padding_1d(seq_1d, max_len, pad_tok=None, direct='right'):
    for i in range(0, max_len - len(seq_1d)):
        if direct in ['right']:
            seq_1d.append(pad_tok)
        else:
            seq_1d.insert(0, pad_tok)
    return seq_1d


def padding_2d(seq_2d, max_len, pad_tok=0, direct='right'):

    for seq_1d in seq_2d:
        for i in range(0, max_len - len(seq_1d)):
            if direct in ['right']:
                seq_1d.append(pad_tok)
            else:
                seq_1d.insert(0, pad_tok)
    return seq_2d


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
    assert len(bpe_x)==len(deunk_bpe_x)
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

                    line = line.replace('>>', '>').replace('<<', '<')

                    if is_raw:
                        line = escape(line)  
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
                        sbp_toks = tokenizer.tokenize(' '.join(toks))
                        deunk_toks = explore_unk(sbp_toks, toks)
                        sbp_labs = match_sbp_label(deunk_toks, labs)
                        sbp_cert_labs = match_sbp_cert_labs(deunk_toks, cert_labs)
                        sbp_ttype_labs = match_sbp_cert_labs(deunk_toks, ttype_labs)
                        sbp_state_labs = match_sbp_cert_labs(deunk_toks, state_labs)
                    else:
                        seg = juman.analysis(line)
                        toks = [w.midasi for w in seg.mrph_list()]
                        sent_stat.append(len(toks))
                        # replace '\u3000' to '[JASP]' 
                        toks = ['[JASP]' if t == '\u3000' else mojimoji.han_to_zen(t) for t in toks]
                        sbp_toks = tokenizer.tokenize(' '.join(toks))
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
                
            
def batch_convert_clinical_data_to_conll(data_dir, file_out, tokenizer, sent_tag=True, defaut_cert='_', is_raw=False):
    doc_stat = []
    with open(file_out, 'w') as fo:
        for file in os.listdir(data_dir):
            ext = ".sent" if sent_tag else ".txt"
            if file.endswith(ext):
                try:
                    dir_file = os.path.join(data_dir, file)
                    doc_stat.append(convert_clinical_data_to_conll(
                        dir_file, fo, tokenizer, sent_tag=sent_tag,
                        defaut_cert=defaut_cert, 
                        is_raw=is_raw
                    ))
                except Exception as ex:
                    print('[error]:' + file)
                    print(ex)
    return doc_stat
                
                
def read_conll(conll_file):
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
            deunk, tok, lab, cert_lab, ttype_lab, state_lab = line.split('\t')
            sent_deunks.append(deunk)
            sent_toks.append(tok)
            sent_labs.append(lab)
            sent_cert_labs.append(cert_lab)
            sent_ttype_labs.append(ttype_lab)
            sent_state_labs.append(state_lab)
    return deunks, toks, labs, cert_labs, ttype_labs, state_labs


def extract_ner_from_conll(conll_file, tokenizer, lab2ix, device):
    deunks, toks, labs, cert_labs, ttype_labs, state_labs = read_conll(conll_file)
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

        # if 'B-T-test' in s_labs and 'B-R' in s_labs:
        #     print(s_toks)
        #     print(s_labs)
        #     print(s_ner_masks)
        #     print(s_clab_masks)
        #     print(s_ner_clabs)
        #     print()

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

            
def eval_pid_seq(model, tokenizer, test_data, orig_token, label2ix, epoch):
    lines = []
    model.eval()
    with torch.no_grad():
        with open('output_ep%i.txt' % epoch, 'w') as fo:
            for (token, mask, gold), ti in zip(test_data, orig_token):
                pred_prob = model(token, attention_mask=mask)[0]
                pred = torch.argmax(pred_prob, dim=-1)

                t_masked_ix = torch.masked_select(token[:,1:], mask[:,1:].byte())
                pred_masked_ix = torch.masked_select(pred[:,1:], mask[:,1:].byte())
                gold_masked_ix = torch.masked_select(gold[:,1:], mask[:,1:].byte())
                
                ix2label = {v:k for k, v in label2ix.items()}
                
                bpe_tok = [tokenizer.convert_ids_to_tokens([ix])[0] for ix in t_masked_ix.tolist()]
                
                flat_ori_tok = [item for sublist in ti for item in sublist]
                   
                deunk_bpe_tok = explore_unk(bpe_tok, flat_ori_tok)
                                
                for t, g, p in zip(deunk_bpe_tok, gold_masked_ix.tolist(), pred_masked_ix.tolist()):
                    fo.write('%s\t%s\t%s\n' % (t, ix2label[g], ix2label[p]))
                fo.write('EOR\tO\tO\n')
                
#                print(sum([len(st) for st in deunk_bpe_tok]), len(flat_ori_tok), len(pred_masked_ix))

#                lines += out_xml(deunk_bpe_tok, pred_masked_ix.tolist(), ix2label)
#                
#    with open('data/seq_pred_ep%i.txt' % epoch, 'w') as fo:
#        for line in lines:
#            fo.write(line + '\n')


def eval_seq(model, tokenizer, test_data, deunk_toks, label2ix, file_out):
    model.eval()
    with torch.no_grad():
        with open('%s_eval.txt' % file_out, 'w') as fe, open('%s_out.txt' % file_out, 'w') as fo:
            for deunk_tok, (token, mask, gold) in zip(deunk_toks, test_data):
                pred_prob = model(token, attention_mask=mask)[0]
                # pred_prob = model.decode(token, mask)
                pred = torch.argmax(pred_prob, dim=-1)

                t_masked_ix = torch.masked_select(token[:, 1:], mask[:, 1:].byte())
                pred_masked_ix = torch.masked_select(pred[:, 1:], mask[:, 1:].byte())
                gold_masked_ix = torch.masked_select(gold[:, 1:], mask[:, 1:].byte())
                
                ix2label = {v: k for k, v in label2ix.items()}
                
                bpe_tok = [tokenizer.convert_ids_to_tokens([ix])[0] for ix in t_masked_ix.tolist()]
                
                assert len(bpe_tok) == len(deunk_tok)
                
                for t, g, p in zip(deunk_tok, gold_masked_ix.tolist(), pred_masked_ix.tolist()):
                    fe.write('%s\t%s\t%s\n' % (t, ix2label[g], ix2label[p]))
                fe.write('\n')
                for dt, t, g, p in zip(deunk_tok, bpe_tok, gold_masked_ix.tolist(), pred_masked_ix.tolist()):
                    fo.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (dt, t, ix2label[p], '_', '_', '_'))
                fo.write('\n')


# def eval_seq_cert(model, tokenizer, test_dataloader, test_deunks, test_labs, cert_lab2ix, file_out):
#     pred_labs, gold_labs = [], []
#     ix2clab = {v:k for k,v in cert_lab2ix.items()}
#     model.eval()
#     with torch.no_grad():
#         with open(file_out, 'w') as fo:
#             for b_deunk, b_labs, (b_toks, b_masks, b_ner_masks, b_clab_masks, b_clabs) in zip(test_deunks, test_labs, test_dataloader):
#                 pred_prob = model(b_toks, b_ner_masks, b_clab_masks, attention_mask=b_masks)
#                 active_index = b_clab_masks.view(-1) == 1
#                 if not (active_index != 0).sum().item():
#                     for t_deunk, t_lab in zip(b_deunk, b_labs):
#                         fo.write('%s\t%s\t%s\n' % (t_deunk, t_lab, '_'))
#                     fo.write('\n')
#                     continue
#                 active_pred_prob = pred_prob.view(-1, len(cert_lab2ix))[active_index]
#                 active_pred_lab = torch.argmax(active_pred_prob, dim=-1)
#                 active_gold_lab = b_clabs.view(-1)[active_index]
#                 pred_labs.append(active_pred_lab.tolist())
#                 gold_labs.append(active_gold_lab.tolist())
#
#                 active_pred_lab_list = active_pred_lab.tolist()
#                 for t_deunk, t_lab in zip(b_deunk, b_labs):
#                     fo.write('%s\t%s\t%s\n' % (t_deunk, t_lab, ix2clab[active_pred_lab_list.pop(0)] if t_lab == 'B-D' else '_'))
#                 fo.write('\n')

