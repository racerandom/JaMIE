#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import torch
from utils import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule


from model import SeqCertClassifier, save_bert_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

print('device', device)

if str(device) == 'cuda':
    BERT_URL='/larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE'
else:
    BERT_URL = '/Users/fei-c/Resources/embed/L-12_H-768_A-12_E-30_BPE'

juman = Juman()
tokenizer = BertTokenizer.from_pretrained(BERT_URL, do_lower_case=False, do_basic_tokenize=False)

# CORPUS='goku'
# MODEL_DIR = 'outputs/'

import argparse
parser = argparse.ArgumentParser(description='PRISM certainty classification')

parser.add_argument("-c", "--corpus", dest="CORPUS", default='goku', type=str,
                    help="goku (国がん), osaka (阪大), tb (BCCWJ-Timebank)")
parser.add_argument("-m", "--model", dest="MODEL_DIR", default='checkpoints/cert/', type=str,
                    help="save/load model dir")
parser.add_argument("-b", "--batch", dest="BATCH_SIZE", default=16, type=int,
                    help="BATCH SIZE")
parser.add_argument("-e", "--epoch", dest="NUM_EPOCHS", default=3, type=int,
                    help="fine-tuning epoch number")
parser.add_argument("-n", "--ner_out", dest="NER_OUT", type=str,
                    help="tag recognition results")
parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")

args = parser.parse_args()

# batch_convert_clinical_data_to_conll('data/train_%s/' % CORPUS, 'data/train_%s.txt' % CORPUS, sent_tag=True,  is_raw=False)
# batch_convert_clinical_data_to_conll('data/test_%s/' % CORPUS, 'data/test_%s.txt' % CORPUS, sent_tag=True,  is_raw=False)
# batch_convert_clinical_data_to_conll('data/records/', 'data/records.txt', sent_tag=False, is_raw=True)

train_deunks, train_toks, train_labs, train_cert_labs = read_conll('data/train_%s.txt' % args.CORPUS)
# test_deunks, test_toks, test_labs, test_cert_labs = read_conll('data/test_%s.txt' % CORPUS)
test_deunks, test_toks, test_labs, test_cert_labs = read_conll(args.NER_OUT)
# test_deunks, test_toks, test_labs, test_cert_labs = read_conll('data/records.txt')


# In[3]:


whole_toks = train_toks + test_toks
max_len = max([len(x) for x in whole_toks])
print('max sequence length:', max_len)
unk_count = sum([x.count('[UNK]') for x in whole_toks])
total_count = sum([len(x) for x in whole_toks])
print('[UNK] token: %s, total: %s, oov rate: %.2f%%' % (unk_count, total_count, unk_count * 100 / total_count))
print('[Example:]', whole_toks[0])
cert_lab2ix = {'positive':1, 'negative':2, 'suspicious':3, '[PAD]':0}
lab2ix = get_label2ix(train_labs)
print(lab2ix)
print(cert_lab2ix)


train_tensors, train_deunk = extract_cert_from_conll('data/train_%s.txt' % args.CORPUS,
                                                     tokenizer,
                                                     cert_lab2ix,
                                                     device)

# test_tensors, test_deunk = extract_ner_from_conll('data/records.txt', tokenizer, lab2ix)
train_dataloader = DataLoader(train_tensors, batch_size=args.BATCH_SIZE, shuffle=True)
print('train size: %i' % len(train_tensors))

def eval_seq_cert(model, tokenizer, test_dataloader, test_deunks, test_labs, cert_lab2ix, file_out):
    pred_labs, gold_labs = [], []
    ix2clab = {v:k for k,v in cert_lab2ix.items()}
    model.eval()
    with torch.no_grad():
        with open(file_out, 'w') as fo:
            for b_deunk, b_labs, (b_toks, b_masks, b_ner_masks, b_clab_masks, b_clabs) in zip(test_deunks, test_labs, test_dataloader):
                pred_prob = model(b_toks, b_ner_masks, b_clab_masks, attention_mask=b_masks)
                active_index = b_clab_masks.view(-1) == 1
                if not (active_index != 0).sum().item():
                    for t_deunk, t_lab in zip(b_deunk, b_labs):
                        fo.write('%s\t%s\t%s\n' % (t_deunk, t_lab, '_'))
                    fo.write('\n')
                    continue
                active_pred_prob = pred_prob.view(-1, len(cert_lab2ix))[active_index]
                active_pred_lab = torch.argmax(active_pred_prob, dim=-1)
                active_gold_lab = b_clabs.view(-1)[active_index]
                pred_labs.append(active_pred_lab.tolist())
                gold_labs.append(active_gold_lab.tolist())

                active_pred_lab_list = active_pred_lab.tolist()
                for t_deunk, t_lab in zip(b_deunk, b_labs):
                    fo.write('%s\t%s\t%s\n' % (t_deunk, t_lab, ix2clab[active_pred_lab_list.pop(0)] if t_lab == 'B-D' else '_'))
                fo.write('\n')


if args.do_train:
    model = SeqCertClassifier.from_pretrained(BERT_URL, num_labels=len(cert_lab2ix))
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

    test_tensors, test_deunk = extract_cert_from_conll(args.NER_OUT,
                                                       tokenizer,
                                                       cert_lab2ix,
                                                       device,
                                                       test_mode=True)
    test_dataloader = DataLoader(test_tensors, batch_size=1, shuffle=False)
    print('test size: %i' % len(test_tensors))
    output_file = 'outputs/cert_%s_ep%i_out.txt' % (args.CORPUS, args.NUM_EPOCHS)
    eval_seq_cert(model, tokenizer, test_dataloader, test_deunk, test_labs, cert_lab2ix, output_file)

    model.train()
    for epoch in range(1, args.NUM_EPOCHS + 1):
        for (b_toks, b_masks, b_ner_masks, b_clab_masks, b_clabs) in tqdm(train_dataloader, desc='Training'):
            loss = model(b_toks, b_ner_masks, b_clab_masks, attention_mask=b_masks, labels=b_clabs)

            loss.backward()
            optimizer.step()
            model.zero_grad()

    """ save the trained model """
    save_bert_model(model, tokenizer, args.MODEL_DIR)

""" load the new tokenizer """
tokenizer = BertTokenizer.from_pretrained(args.MODEL_DIR, do_lower_case=False, do_basic_tokenize=False)


""" load the new model"""
config = BertConfig.from_json_file(os.path.join(args.MODEL_DIR, 'config.json'))
model = BertForTokenClassification(config, num_labels=len(cert_lab2ix))
state_dict = torch.load(os.path.join(args.MODEL_DIR, 'pytorch_model.bin'))
model.load_state_dict(state_dict)
model.to(device)

output_file = 'outputs/cert_%s_ep%i_out.txt' % (args.CORPUS, args.NUM_EPOCHS)

eval_seq_cert(model, tokenizer, test_dataloader, test_deunk, test_labs, cert_lab2ix, output_file)






