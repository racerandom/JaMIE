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

def freeze_bert_layers(model, bert_name='bert', freeze_embed=True, layer_list=None):
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

""" 
python input arguments 
"""

parser = argparse.ArgumentParser(description='PRISM tag recognizer')

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

parser.add_argument("--freeze", dest="EPOCH_FREEZE", default=10, type=int,
                    help="freeze the BERT encoder after N epoches")

parser.add_argument("--gradual_freeze",
                    action='store_true',
                    help="gradually freeze the BERT encoder from bottom to top")

parser.add_argument("--fine_epoch", dest="NUM_FINE_EPOCHS", default=5, type=int,
                    help="fine-tuning epoch number")

parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")

parser.add_argument("--do_crf",
                    action='store_true',
                    help="Whether to use CRF.")

parser.add_argument("-o", "--output", dest="OUTPUT_FILE", default='outputs/temp.ner', type=str,
                    help="output filename")

parser.add_argument("--epoch_eval",
                    action='store_true',
                    help="Whether eval model every epoch.")

parser.add_argument("--fp16",
                    action='store_true',
                    help="fp16")

parser.add_argument("--fp16_opt_level", type=str, default="O1",
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                    "See details at https://nvidia.github.io/apex/amp.html")

args = parser.parse_args()


tokenizer = BertTokenizer.from_pretrained(args.PRE_MODEL, do_lower_case=False, do_basic_tokenize=False)


TRAIN_FILE = args.TRAIN_FILE
TEST_FILE = args.TEST_FILE

""" Read conll file for counting statistics, such as: [UNK] token ratio, label2ix, etc. """
train_deunks, train_toks, train_labs, train_cert_labs, train_ttype_labs, train_state_labs = read_conll(TRAIN_FILE)
test_deunks, test_toks, test_labs, test_cert_labs, test_ttype_labs, test_state_labs = read_conll(TEST_FILE)
# test_deunks, test_toks, test_labs, test_cert_labs = read_conll('data/records.txt')

whole_toks = train_toks + test_toks
max_len = max([len(x) for x in whole_toks])
unk_count = sum([x.count('[UNK]') for x in whole_toks])
total_count = sum([len(x) for x in whole_toks])

lab2ix = get_label2ix(train_labs + test_labs)
cert_lab2ix = get_label2ix(train_cert_labs + test_cert_labs)
ttype_lab2ix = get_label2ix(train_ttype_labs + test_ttype_labs)
state_lab2ix = get_label2ix(train_state_labs + test_state_labs)


if args.do_train:


    print('max sequence length:', max_len)

    print('[UNK] token: %s, total: %s, oov rate: %.2f%%' % (unk_count, total_count, unk_count * 100 / total_count))
    print('[Example:]', whole_toks[0])

    print(lab2ix)
    print(cert_lab2ix)
    print(ttype_lab2ix)
    print(state_lab2ix)

    """ 
    - Generate train/test tensors including (token_ids, mask_ids, label_ids) 
    - wrap them into dataloader for mini-batch cutting
    """
    train_tensors, train_deunk = extract_ner_from_conll(TRAIN_FILE, tokenizer, lab2ix, device)
    train_dataloader = DataLoader(train_tensors, batch_size=args.BATCH_SIZE, shuffle=True)
    print('train size: %i' % len(train_tensors))

    test_tensors, test_deunk = extract_ner_from_conll(TEST_FILE, tokenizer, lab2ix, device)
    test_dataloader = DataLoader(test_tensors, batch_size=args.BATCH_SIZE, shuffle=False)
    test_deunk_loader = [test_deunk[i: i + args.BATCH_SIZE] for i in range(0, len(test_deunk), args.BATCH_SIZE)]
    print('test size: %i' % len(test_tensors))

    model_dir = ""


    """ Disease Tags recognition """

    if args.do_crf:
        model = BertCRF.from_pretrained(args.PRE_MODEL, num_labels=len(lab2ix))

        # specify different lr
        param_optimizer = list(model.named_parameters())
        crf_name_list = ['crf_layer']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in crf_name_list)], 'lr': 1e-3},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in crf_name_list)], 'lr': 5e-5}
        ]
        # To reproduce BertAdam specific behavior set correct_bias=False
        optimizer = AdamW(
            optimizer_grouped_parameters,
            correct_bias=False,
            weight_decay=1e-2,
        )
    else:
        model = BertForTokenClassification.from_pretrained(args.PRE_MODEL, num_labels=len(lab2ix))

        # To reproduce BertAdam specific behavior set correct_bias=False
        optimizer = AdamW(
            model.parameters(),
            lr=5e-5,
            correct_bias=False
        )
    model.to(device)

    # PyTorch scheduler
    num_epoch_steps = len(train_dataloader)
    num_finetuning_steps = args.NUM_FINE_EPOCHS * num_epoch_steps
    num_training_steps = args.NUM_EPOCHS * num_epoch_steps
    warmup_ratio = 0.1

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_finetuning_steps,
        num_training_steps=num_training_steps
    )


    # support fp16
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    for epoch in range(1, args.NUM_EPOCHS + 1):

        model.train()

        if args.gradual_freeze:
            freeze_bert_layers(model, freeze_embed=False, layer_list=list(range(0, epoch - 1)))
        else:
            if epoch > args.EPOCH_FREEZE:
                freeze_bert_layers(model, layer_list=list(range(0, 11)))


        epoch_loss = 0.0

        for batch_feat, batch_mask, batch_lab in tqdm(train_dataloader, desc='Training'):

            # BERT loss, logits: (batch_size, seq_len, tag_num)
            if args.do_crf:
                # transformers return tuple
                loss = model(batch_feat, attention_mask=batch_mask, labels=batch_lab)
            else:
                loss = model(batch_feat, attention_mask=batch_mask, labels=batch_lab)[0]
            # print(loss)
            epoch_loss += loss.item()

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if args.epoch_eval:
            if not args.do_crf:
                eval_seq(model, tokenizer, test_dataloader, test_deunk_loader, lab2ix, args.OUTPUT_FILE)
            else:
                eval_crf(model, tokenizer, test_dataloader, test_deunk_loader, lab2ix, args.OUTPUT_FILE)
            import subprocess
            eval_out = subprocess.check_output(
                ['./ner_eval.sh', args.OUTPUT_FILE]
            ).decode("utf-8")
            print("epoch loss: %.6f; " % (epoch_loss/len(train_dataloader)), eval_out.split('\n')[2])

        """ save the trained model per epoch """
        if args.do_crf:
            model_dir = "%s/crf" % args.MODEL_DIR
        else:
            model_dir = "%s/seq" % args.MODEL_DIR
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
else:
    model_dir = args.MODEL_DIR
    """ load the new tokenizer"""
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False, do_basic_tokenize=False)
    test_tensors, test_deunk = extract_ner_from_conll(TEST_FILE, tokenizer, lab2ix, device)
    test_dataloader = DataLoader(test_tensors, batch_size=args.BATCH_SIZE, shuffle=False)
    test_deunk_loader = [test_deunk[i: i + args.BATCH_SIZE] for i in range(0, len(test_deunk), args.BATCH_SIZE)]
    print('test size: %i' % len(test_tensors))


""" load the new model"""
if args.do_crf:
    model = BertCRF.from_pretrained(model_dir)
else:
    model = BertForTokenClassification.from_pretrained(model_dir)
model.to(device)

""" predict test out """

if not args.do_crf:
    eval_seq(model, tokenizer, test_dataloader, test_deunk_loader, lab2ix, args.OUTPUT_FILE)
else:
    eval_crf(model, tokenizer, test_dataloader, test_deunk_loader, lab2ix, args.OUTPUT_FILE)


