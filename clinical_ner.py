#!/usr/bin/env python
# coding: utf-8
import os
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import torch
from utils import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from model import save_bert_model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

print('device', device)

#if str(device) == 'cuda':
##    BERT_URL='/larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE'
#    BERT_URL='/home/feicheng/transformers/examples/lm_finetuning/new_lm/finetuing_lm_e10'
#else:
#    BERT_URL = '/Users/fei-c/Resources/embed/L-12_H-768_A-12_E-30_BPE'

juman = Juman()

# torch.cuda.manual_seed_all(1029)

# CORPUS='goku'

import argparse
parser = argparse.ArgumentParser(description='PRISM tag recognizer')

parser.add_argument("-c", "--corpus", dest="CORPUS", default='goku', type=str,
                    help="goku (国がん), osaka (阪大), tb (BCCWJ-Timebank)")
parser.add_argument("-m", "--model", dest="MODEL_DIR", default='checkpoints/ner_full/', type=str,
                    help="save/load model dir")
parser.add_argument("-p", "--pre", dest="PRE_MODEL", default='/larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE', type=str, help="pre-trained model dir")
parser.add_argument("-b", "--batch", dest="BATCH_SIZE", default=16, type=int,
                    help="BATCH SIZE")
parser.add_argument("-e", "--epoch", dest="NUM_EPOCHS", default=3, type=int,
                    help="fine-tuning epoch number")
parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("-o", "--output", dest="OUTPUT_FILE", default='outputs/temp.ner', type=str,
                    help="output filename")
args = parser.parse_args()


# In[2]:
tokenizer = BertTokenizer.from_pretrained(args.PRE_MODEL, do_lower_case=False, do_basic_tokenize=False)

# batch_convert_clinical_data_to_conll('data/train_%s/' % CORPUS, 'data/train_%s.txt' % CORPUS, sent_tag=True,  is_raw=False)
# batch_convert_clinical_data_to_conll('data/test_%s/' % CORPUS, 'data/test_%s.txt' % CORPUS, sent_tag=True,  is_raw=False)
# batch_convert_clinical_data_to_conll('data/records/', 'data/records.txt', sent_tag=False, is_raw=True)


""" Read conll file for counting statistics, such as: [UNK] token ratio, label2ix, etc. """
train_deunks, train_toks, train_labs, train_cert_labs = read_conll('data/train_%s.txt' % args.CORPUS)
test_deunks, test_toks, test_labs, test_cert_labs = read_conll('data/test_%s.txt' % args.CORPUS)
# test_deunks, test_toks, test_labs, test_cert_labs = read_conll('data/records.txt')

whole_toks = train_toks + test_toks
max_len = max([len(x) for x in whole_toks])
print('max sequence length:', max_len)
unk_count = sum([x.count('[UNK]') for x in whole_toks])
total_count = sum([len(x) for x in whole_toks])       
print('[UNK] token: %s, total: %s, oov rate: %.2f%%' % (unk_count, total_count, unk_count * 100 / total_count))
print('[Example:]', whole_toks[0])
cert_lab2ix = {'positive':1, 'negative':2, 'suspicious':3, '[PAD]':0}
lab2ix = get_label2ix(train_labs + test_labs)
print(lab2ix)
print(cert_lab2ix)


""" 
- Generate train/test tensors including (token_ids, mask_ids, label_ids) 
- wrap them into dataloader for mini-batch cutting
"""
# train_tensors, train_deunk = extract_cert_from_conll('data/train_%s.txt' % CORPUS, tokenizer, cert_lab2ix, device)
# test_tensors, test_deunk = extract_cert_from_conll('data/test_%s.txt' % CORPUS, tokenizer, cert_lab2ix, device)
train_tensors, train_deunk = extract_ner_from_conll('data/train_%s.txt' % args.CORPUS, tokenizer, lab2ix, device)


# NUM_EPOCHS = 5
# BATCH_SIZE = 16
# test_tensors, test_deunk = extract_ner_from_conll('data/records.txt', tokenizer, lab2ix)
train_dataloader = DataLoader(train_tensors, batch_size=args.BATCH_SIZE, shuffle=True)
print('train size: %i' % len(train_tensors))

if args.do_train:
    """ Disease Tags recognition """
    model = BertForTokenClassification.from_pretrained(args.PRE_MODEL, num_labels=len(lab2ix))
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer= BertAdam(optimizer_grouped_parameters,
                        lr=5e-5,
                        warmup=0.1,
                        t_total=args.NUM_EPOCHS * len(train_dataloader))

    model.train()
    for epoch in range(1, args.NUM_EPOCHS + 1):
        for batch_feat, batch_mask, batch_lab in tqdm(train_dataloader, desc='Training'):
            loss = model(batch_feat, attention_mask=batch_mask, labels=batch_lab)

            loss.backward()
            optimizer.step()
            model.zero_grad()

    #        if step !=0 and step % 100 == 0:
    #            eval(model, test_data, label2ix)

    """ save the trained model """
#    save_bert_model(model, tokenizer, args.MODEL_DIR)

""" load the new tokenizer"""
# tokenizer = BertTokenizer.from_pretrained(args.MODEL_DIR, do_lower_case=False, do_basic_tokenize=False)
test_tensors, test_deunk = extract_ner_from_conll('data/test_%s.txt' % args.CORPUS, tokenizer, lab2ix, device)
test_dataloader = DataLoader(test_tensors, batch_size=1, shuffle=False)
print('ttest size: %i' % len(test_tensors))

""" load the new model"""
# config = BertConfig.from_json_file(os.path.join(args.MODEL_DIR, 'config.json'))
# model = BertForTokenClassification(config, num_labels=len(lab2ix))
# state_dict = torch.load(os.path.join(args.MODEL_DIR, 'pytorch_model.bin'))
# model.load_state_dict(state_dict)
# model.to(device)

""" predict test out """
#output_file = 'outputs/ner_%s_ep%i' % (args.CORPUS, args.NUM_EPOCHS)
eval_seq(model, tokenizer, test_dataloader, test_deunks, lab2ix, args.OUTPUT_FILE)
