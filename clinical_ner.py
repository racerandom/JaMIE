#!/usr/bin/env python
# coding: utf-8
import warnings
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import *
import argparse

from model import *

warnings.filterwarnings("ignore")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

print('device', device)

juman = Juman()

torch.cuda.manual_seed_all(1234)


parser = argparse.ArgumentParser(description='PRISM tag recognizer')

parser.add_argument("-c", "--corpus", dest="CORPUS", default='goku', type=str,
                    help="goku (国がん), osaka (阪大), tb (BCCWJ-Timebank)")

parser.add_argument("-m", "--model", dest="MODEL_DIR", default='checkpoints/ner', type=str,
                    help="save/load model dir")

parser.add_argument("--train_file", dest="TRAIN_FILE", type=str,
                    help="train file, BIO format.")

parser.add_argument("--test_file", dest="TEST_FILE", type=str,
                    help="test file, BIO format.")

parser.add_argument("-p", "--pre", dest="PRE_MODEL",
                    default='/home/feicheng/Tools/Japanese_L-12_H-768_A-12_E-30_BPE',
                    type=str,
                    help="pre-trained model dir")

parser.add_argument("-b", "--batch", dest="BATCH_SIZE", default=16, type=int,
                    help="BATCH SIZE")

parser.add_argument("-e", "--epoch", dest="NUM_EPOCHS", default=3, type=int,
                    help="epoch number")

parser.add_argument("--fine_epoch", dest="NUM_FINE_EPOCHS", default=3, type=int,
                    help="fine-tuning epoch number")

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")

parser.add_argument("--do_crf",
                    action='store_true',
                    help="Whether to use CRF.")

parser.add_argument("-o", "--output", dest="OUTPUT_FILE", default='outputs/temp.ner', type=str,
                    help="output filename")

args = parser.parse_args()


# In[2]:
tokenizer = BertTokenizer.from_pretrained(args.PRE_MODEL, do_lower_case=False, do_basic_tokenize=False)

# batch_convert_clinical_data_to_conll('data/train_%s/' % CORPUS, 'data/train_%s.txt' % CORPUS, sent_tag=True,  is_raw=False)
# batch_convert_clinical_data_to_conll('data/test_%s/' % CORPUS, 'data/test_%s.txt' % CORPUS, sent_tag=True,  is_raw=False)
# batch_convert_clinical_data_to_conll('data/records/', 'data/records.txt', sent_tag=False, is_raw=True)

# TRAIN_FILE = 'data/train_%s.txt' % args.CORPUS
# TEST_FILE = 'data/test_%s.txt' % args.CORPUS

TRAIN_FILE = args.TRAIN_FILE
TEST_FILE = args.TEST_FILE

""" Read conll file for counting statistics, such as: [UNK] token ratio, label2ix, etc. """
train_deunks, train_toks, train_labs, train_cert_labs, train_ttype_labs, train_state_labs = read_conll(TRAIN_FILE)
test_deunks, test_toks, test_labs, test_cert_labs, test_ttype_labs, test_state_labs = read_conll(TEST_FILE)
# test_deunks, test_toks, test_labs, test_cert_labs = read_conll('data/records.txt')

whole_toks = train_toks + test_toks
max_len = max([len(x) for x in whole_toks])
print('max sequence length:', max_len)
unk_count = sum([x.count('[UNK]') for x in whole_toks])
total_count = sum([len(x) for x in whole_toks])       
print('[UNK] token: %s, total: %s, oov rate: %.2f%%' % (unk_count, total_count, unk_count * 100 / total_count))
print('[Example:]', whole_toks[0])
# cert_lab2ix = {'positive': 1, 'negative': 2, 'suspicious': 3, '[PAD]': 0}
lab2ix = get_label2ix(train_labs + test_labs)
print(lab2ix)
cert_lab2ix = get_label2ix(train_cert_labs + test_cert_labs, default=True)
ttype_lab2ix = get_label2ix(train_ttype_labs + test_ttype_labs, default=True)
state_lab2ix = get_label2ix(train_state_labs + test_state_labs, default=True)
print(cert_lab2ix)
print(ttype_lab2ix)
print(state_lab2ix)

""" 
- Generate train/test tensors including (token_ids, mask_ids, label_ids) 
- wrap them into dataloader for mini-batch cutting
"""
# train_tensors, train_deunk = extract_cert_from_conll('data/train_%s.txt' % CORPUS, tokenizer, cert_lab2ix, device)
# test_tensors, test_deunk = extract_cert_from_conll('data/test_%s.txt' % CORPUS, tokenizer, cert_lab2ix, device)
train_tensors, train_deunk = extract_ner_from_conll(TRAIN_FILE, tokenizer, lab2ix, device)

# test_tensors, test_deunk = extract_ner_from_conll('data/records.txt', tokenizer, lab2ix)
train_dataloader = DataLoader(train_tensors, batch_size=args.BATCH_SIZE, shuffle=True)
print('train size: %i' % len(train_tensors))
model_dir = ""

if args.do_train:
    """ Disease Tags recognition """

    if args.do_crf:
        model = BertCRF.from_pretrained(args.PRE_MODEL, num_labels=len(lab2ix))
    else:
        model = BertForTokenClassification.from_pretrained(args.PRE_MODEL, num_labels=len(lab2ix))
    model.to(device)

    param_optimizer = list(model.named_parameters())
    bert_name_list = ['bert']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in bert_name_list)], 'lr': 5e-5},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in bert_name_list)], 'lr': 1e-2}
    ]

    num_epoch_steps = len(train_dataloader)
    num_finetuning_steps = args.NUM_FINE_EPOCHS * num_epoch_steps
    num_training_steps = args.NUM_EPOCHS * num_epoch_steps
    warmup_ratio = 0.1
    max_grad_norm = 1.0

    # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW(
        optimizer_grouped_parameters,
        correct_bias=False
    )

    # PyTorch scheduler

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_finetuning_steps * warmup_ratio,
        num_training_steps=num_training_steps
    )

    model.train()
    for epoch in range(1, args.NUM_EPOCHS + 1):

        print("BERT lr:%.8f, Non-BERT lr:%.8f" % (
            optimizer.state_dict()['param_groups'][0]['lr'],
            optimizer.state_dict()['param_groups'][1]['lr']
        ))

        # if epoch > 5:
        #     for param in model.bert.parameters():
        #         param.requires_grad = False

        for batch_feat, batch_mask, batch_lab in tqdm(train_dataloader, desc='Training'):

            # BERT loss, logits: (batch_size, seq_len, tag_num)
            if args.do_crf:
                # transformers return tuple
                loss = -model.crf_forward(batch_feat, attention_mask=batch_mask, labels=batch_lab)
            else:
                loss = model(batch_feat, attention_mask=batch_mask, labels=batch_lab)[0]
            # print(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        """ save the trained model per epoch """
        if args.do_crf:
            model_dir = "%s/crf_ep%i" % (args.MODEL_DIR, epoch)
        else:
            model_dir = "%s/seq_ep%i" % (args.MODEL_DIR, epoch)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
else:
    model_dir = args.MODEL_DIR
    
""" load the new tokenizer"""
tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False, do_basic_tokenize=False)
test_tensors, test_deunk = extract_ner_from_conll(TEST_FILE, tokenizer, lab2ix, device)
test_dataloader = DataLoader(test_tensors, batch_size=1, shuffle=False)
print('test size: %i' % len(test_tensors))

""" load the new model"""
if args.do_crf:
    model = BertCRF.from_pretrained(model_dir)
else:
    model = BertForTokenClassification.from_pretrained(model_dir)
model.to(device)

""" predict test out """
# output_file = 'outputs/ner_%s_ep%i' % (args.CORPUS, args.NUM_EPOCHS)

if not args.do_crf:
    eval_seq(model, tokenizer, test_dataloader, test_deunks, lab2ix, args.OUTPUT_FILE)
else:
    ix2lab = {v: k for k, v in lab2ix.items()}

    model.eval()
    with torch.no_grad():
        with open(args.OUTPUT_FILE + '_eval.txt', 'w', encoding='utf8') as fo:
            for sent_deunk, (sent_tok_ids, sent_mask, sent_gold) in zip(test_deunks, test_dataloader):
                pred_ix = model.decode(sent_tok_ids, sent_mask)[0][1:]
                gold_masked_ix = torch.masked_select(sent_gold[:, 1:], sent_mask[:, 1:].byte()).tolist()
                if not sent_deunk:
                    continue
                assert len(sent_deunk) == len(gold_masked_ix) == len(pred_ix)
                for tok_deunk, tok_gold, tok_pred in zip(sent_deunk, gold_masked_ix, pred_ix):
                    fo.write("%s\t%s\t%s\n" % (tok_deunk, ix2lab[tok_gold], ix2lab[tok_pred]))
                fo.write("\n")



