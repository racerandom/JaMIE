#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import argparse
from transformers import *

# local libraries
from utils import *
from model import SeqCertClassifier, save_bert_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

print('device', device)

juman = Juman()

parser = argparse.ArgumentParser(description='PRISM certainty classification')

parser.add_argument("-c", "--corpus", dest="CORPUS", default='goku', type=str,
                    help="goku (国がん), osaka (阪大), tb (BCCWJ-Timebank)")

parser.add_argument("--train_file", dest="TRAIN_FILE", type=str,
                    help="train file, BIO format.")

parser.add_argument("-m", "--model", dest="MODEL_DIR", default='checkpoints/cert/', type=str,
                    help="save/load model dir")

parser.add_argument("-p", "--pre", dest="PRE_MODEL",
                    default='/home/feicheng/Tools/Japanese_L-12_H-768_A-12_E-30_BPE',
                    type=str, help="pre-trained model dir")

parser.add_argument("-b", "--batch", dest="BATCH_SIZE", default=16, type=int,
                    help="BATCH SIZE")

parser.add_argument("-e", "--epoch", dest="NUM_EPOCHS", default=5, type=int,
                    help="fine-tuning epoch number")

parser.add_argument("-a", "--attrib", dest="ATTRIB", default='cert', type=str,
                    help="the attrib name to recognize, value: 'cert', 'ttype' or 'state'")

parser.add_argument("-n", "--ner_out", dest="NER_OUT", type=str,
                    help="tag recognition results")

parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")

parser.add_argument("-o", "--output",
                    dest="OUTPUT_FILE",
                    # default='outputs/temp_cert.txt',
                    type=str,
                    help="output filename")

args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(args.PRE_MODEL, do_lower_case=False, do_basic_tokenize=False)

TRAIN_FILE = args.TRAIN_FILE
TEST_FILE = args.NER_OUT

# batch_convert_clinical_data_to_conll('data/train_%s/' % CORPUS, 'data/train_%s.txt' % CORPUS, sent_tag=True,  is_raw=False)
# batch_convert_clinical_data_to_conll('data/test_%s/' % CORPUS, 'data/test_%s.txt' % CORPUS, sent_tag=True,  is_raw=False)
# batch_convert_clinical_data_to_conll('data/records/', 'data/records.txt', sent_tag=False, is_raw=True)

train_deunks, train_toks, train_labs, train_cert_labs, train_ttype_labs, train_state_labs = read_conll(TRAIN_FILE)
# test_deunks, test_toks, test_labs, test_cert_labs = read_conll('data/test_%s.txt' % CORPUS)
test_deunks, test_toks, test_labs, test_cert_labs, test_ttype_labs, test_state_labs = read_conll(TEST_FILE)
# test_deunks, test_toks, test_labs, test_cert_labs = read_conll('data/records.txt')


whole_toks = train_toks + test_toks
max_len = max([len(x) for x in whole_toks])
unk_count = sum([x.count('[UNK]') for x in whole_toks])
total_count = sum([len(x) for x in whole_toks])
lab2ix = get_label2ix(train_labs)

if args.ATTRIB == 'cert':
    attrib_lab2ix = get_label2ix(train_cert_labs + test_cert_labs)
elif args.ATTRIB == 'ttype':
    attrib_lab2ix = get_label2ix(train_ttype_labs + test_ttype_labs)
elif args.ATTRIB == 'state':
    attrib_lab2ix = get_label2ix(train_state_labs + test_state_labs)
else:
    raise Exception("Error: wrong task...")

print(attrib_lab2ix)

def eval_seq_cert(model, tokenizer, test_dataloader, test_deunks, test_labs, attrib_lab2ix, file_out, attrib='cert'):
    pred_labs, gold_labs = [], []
    ix2clab = {v: k for k, v in attrib_lab2ix.items()}
    model.eval()
    with torch.no_grad():
        with open(file_out, 'w') as fo:
            for b_deunk, b_labs, (b_toks, b_masks, b_ner_masks, b_clab_masks, b_clabs) in zip(test_deunks, test_labs, test_dataloader):
                # import pdb
                # pdb.set_trace()
                pred_prob = model(b_toks, b_ner_masks, b_clab_masks, attention_mask=b_masks)
                active_index = b_clab_masks.view(-1) == 1
                if not (active_index != 0).sum().item():
                    for t_deunk, t_lab in zip(b_deunk, b_labs):
                        fo.write('%s\t%s\t%s\n' % (t_deunk, t_lab, '_'))
                    fo.write('\n')
                    continue
                active_pred_prob = pred_prob.view(-1, len(attrib_lab2ix))[active_index]
                active_pred_lab = torch.argmax(active_pred_prob, dim=-1)
                active_gold_lab = b_clabs.view(-1)[active_index]
                pred_labs.append(active_pred_lab.tolist())
                gold_labs.append(active_gold_lab.tolist())

                active_pred_lab_list = active_pred_lab.tolist()
                for t_deunk, t_lab in zip(b_deunk, b_labs):
                    if attrib == 'cert':
                        attrib_lab = ix2clab[active_pred_lab_list.pop(0)] if t_lab in ['B-D'] else '_'
                    elif attrib == 'ttype':
                        attrib_lab = ix2clab[active_pred_lab_list.pop(0)] if t_lab in ['B-Timex3'] else '_'
                    elif attrib == 'state':
                        attrib_lab = ix2clab[active_pred_lab_list.pop(0)] if t_lab in ['B-T-test', 'B-R', 'B-Cc'] and active_pred_lab_list else '_'
                    else:
                        raise Exception("Error: wrong task...")
                    fo.write('%s\t%s\t%s\n' % (t_deunk, t_lab, attrib_lab))
                fo.write('\n')


if args.do_train:

    print('max sequence length:', max_len)
    print('[UNK] token: %s, total: %s, oov rate: %.2f%%' % (unk_count, total_count, unk_count * 100 / total_count))
    print('[Example:]', whole_toks[0])

    train_tensors, train_deunk = extract_cert_from_conll(
        TRAIN_FILE,
        tokenizer,
        attrib_lab2ix,
        device,
        max_ner_num=16,
        attrib=args.ATTRIB
    )

    # test_tensors, test_deunk = extract_ner_from_conll('data/records.txt', tokenizer, lab2ix)
    train_dataloader = DataLoader(train_tensors, batch_size=args.BATCH_SIZE, shuffle=True)
    print('train size: %i' % len(train_tensors))

    model = SeqCertClassifier.from_pretrained(
        args.PRE_MODEL,
        num_labels=len(attrib_lab2ix)
    )
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_training_steps = args.NUM_EPOCHS * len(train_dataloader)
    warmup_ratio = 0.1
    max_grad_norm = 1.0

    # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW(
        # model.parameters(),
        optimizer_grouped_parameters,
        lr=5e-5,
        correct_bias=False
    )

    # PyTorch scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps * warmup_ratio,
                                                num_training_steps=num_training_steps)

    model.train()
    for epoch in range(1, args.NUM_EPOCHS + 1):
        for (b_toks, b_masks, b_ner_masks, b_clab_masks, b_clabs) in tqdm(train_dataloader, desc='Training'):

            loss = model(b_toks, b_ner_masks, b_clab_masks, attention_mask=b_masks, labels=b_clabs)

            # temporal solution for the loss of a empty tensor
            if not loss:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        """ save the trained model per epoch """
        model_dir = "checkpoints/attrib/%s_ep%i" % (args.ATTRIB, epoch)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

""" load the new tokenizer """
model_dir = "checkpoints/attrib/%s_ep%i" % (args.ATTRIB, 1)
tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False, do_basic_tokenize=False)
test_tensors, test_deunk = extract_cert_from_conll(
    args.NER_OUT,
    tokenizer,
    attrib_lab2ix,
    device,
    max_ner_num=12,
    attrib=args.ATTRIB,
    test_mode=True
)
test_dataloader = DataLoader(test_tensors, batch_size=1, shuffle=False)
print('test size: %i' % len(test_tensors))

""" load the new model"""
model = SeqCertClassifier.from_pretrained(model_dir)
model.to(device)

eval_seq_cert(model, tokenizer, test_dataloader, test_deunk, test_labs, attrib_lab2ix, args.OUTPUT_FILE, attrib=args.ATTRIB)






