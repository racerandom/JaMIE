#!/usr/bin/env python
# coding: utf-8
import warnings
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
from transformers import *
import argparse
import random
import math

from model import *

warnings.filterwarnings("ignore")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

print('device', device)

juman = Juman()

torch.cuda.manual_seed_all(1234)


def pulse_freeze_bert(model, pulse_delta, bert_name='bert', freeze_embed=False):

    for n, p in list(model.named_parameters()):
        for i in list(range(0, 12)):

            if freeze_embed:
                if n.startswith("%s.embeddings" % bert_name):
                    p.requires_grad = False
            else:
                if n.startswith("%s.embeddings" % bert_name):
                    p.requires_grad = True

            if n.startswith("%s.encoder.layer.%i." % (bert_name, i)):

                if random.random() < pulse_delta:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

""" 
python input arguments 
"""

parser = argparse.ArgumentParser(description='PRISM tag recognizer')

parser.add_argument("-m", "--model", dest="MODEL_DIR", default='checkpoints/Dokuei2019', type=str,
                    help="save/load model dir")

parser.add_argument("--train_file", dest="TRAIN_FILE", type=str,
                    help="train file, BIO format.")

parser.add_argument("--test_file", dest="TEST_FILE", type=str,
                    help="test file, BIO format.")

parser.add_argument("--dev_file", dest="DEV_FILE", type=str,
                    help="dev file, BIO format.")

parser.add_argument("-p", "--pre", dest="PRE_MODEL",
                    default='/home/feicheng/Tools/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers',
                    type=str,
                    help="pre-trained model dir")

parser.add_argument("-b", "--batch", dest="BATCH_SIZE", default=16, type=int,
                    help="BATCH SIZE")

parser.add_argument("-e", "--epoch", dest="NUM_EPOCHS", default=3, type=int,
                    help="epoch number")

parser.add_argument("--freeze", dest="EPOCH_FREEZE", default=6, type=int,
                    help="freeze the BERT encoder after N epoches")

parser.add_argument("--bottomup_freeze",
                    action='store_true',
                    help="freeze the BERT layers from bottom to top")

parser.add_argument("--pulse_freeze",
                    action='store_true',
                    help="pulsely freeze all BERT layers")

parser.add_argument("--pulse_bottomup_freeze",
                    action='store_true',
                    help="pulse freeze the BERT layers from bottom to top")

parser.add_argument("--freeze_embed",
                    action='store_true',
                    help="whether freeze the embedding layer")

parser.add_argument("--fine_epoch", dest="NUM_FINE_EPOCHS", default=2, type=int,
                    help="fine-tuning epoch number")

parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")

parser.add_argument("--do_crf",
                    action='store_true',
                    help="Whether to use CRF.")

parser.add_argument("--test_output", default='outputs/temp_test.ner', type=str,
                    help="test output filename")

parser.add_argument("--dev_output", default='outputs/temp_dev.ner', type=str,
                    help="dev output filename")

parser.add_argument("--later_eval",
                    action='store_true',
                    help="Whether eval model every epoch.")

parser.add_argument("--save_best", action='store', type=str, default='f1',
                    help="save the best model, given dev scores (f1 or loss)")

parser.add_argument("--save_step_interval", default=80, type=int,
                    help="save best model given a step interval")

parser.add_argument("--fp16",
                    action='store_true',
                    help="fp16")

parser.add_argument("--fp16_opt_level", type=str, default="O2",
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                    "See details at https://nvidia.github.io/apex/amp.html")

parser.add_argument("--scheduled_lr",
                    action='store_true',
                    help="learning rate schedule")

parser.add_argument("--joint",
                    action='store_true',
                    help="merge ner and modality jointly")

args = parser.parse_args()


tokenizer = BertTokenizer.from_pretrained(args.PRE_MODEL, do_lower_case=False, do_basic_tokenize=False)


TRAIN_FILE = args.TRAIN_FILE
TEST_FILE = args.TEST_FILE
DEV_FILE = args.DEV_FILE

""" Read conll file for counting statistics, such as: [UNK] token ratio, label2ix, etc. """
train_deunks, train_toks, train_labs, train_cert_labs, train_ttype_labs, train_state_labs = read_conll(
    TRAIN_FILE,
    is_merged=args.joint
)
test_deunks, test_toks, test_labs, test_cert_labs, test_ttype_labs, test_state_labs = read_conll(
    TEST_FILE,
    is_merged=args.joint
)
dev_deunks, dev_toks, dev_labs, dev_cert_labs, dev_ttype_labs, dev_state_labs = read_conll(
    DEV_FILE,
    is_merged=args.joint
)

whole_toks = train_toks + dev_toks + test_toks
max_len = max([len(x) for x in whole_toks])
unk_count = sum([x.count('[UNK]') for x in whole_toks])
total_count = sum([len(x) for x in whole_toks])

lab2ix = get_label2ix(train_labs + dev_labs + test_labs)
cert_lab2ix = get_label2ix(train_cert_labs + dev_cert_labs + test_cert_labs)
ttype_lab2ix = get_label2ix(train_ttype_labs + dev_ttype_labs + test_ttype_labs)
state_lab2ix = get_label2ix(train_state_labs + dev_state_labs + test_state_labs)

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
train_tensors, train_deunk = extract_ner_from_conll(TRAIN_FILE, tokenizer, lab2ix, device, is_merged=args.joint)
train_sampler = RandomSampler(train_tensors)
train_dataloader = DataLoader(train_tensors, sampler=train_sampler, batch_size=args.BATCH_SIZE)
print('train size: %i' % len(train_tensors))

dev_tensors, dev_deunk = extract_ner_from_conll(DEV_FILE, tokenizer, lab2ix, device, is_merged=args.joint)
dev_dataloader = DataLoader(dev_tensors, batch_size=args.BATCH_SIZE, shuffle=False)
dev_deunk_loader = [dev_deunk[i: i + args.BATCH_SIZE] for i in range(0, len(dev_deunk), args.BATCH_SIZE)]
print('dev size: %i' % len(dev_tensors))

test_tensors, test_deunk = extract_ner_from_conll(TEST_FILE, tokenizer, lab2ix, device, is_merged=args.joint)
test_dataloader = DataLoader(test_tensors, batch_size=args.BATCH_SIZE, shuffle=False)
test_deunk_loader = [test_deunk[i: i + args.BATCH_SIZE] for i in range(0, len(test_deunk), args.BATCH_SIZE)]
print('test size: %i' % len(test_tensors))

if args.do_crf:
    model_dir = "%s/crf" % args.MODEL_DIR
else:
    model_dir = "%s/seq" % args.MODEL_DIR




if args.do_train:

    """ Disease Tags recognition """

    if args.do_crf:
        model = BertCRF.from_pretrained(args.PRE_MODEL, num_labels=len(lab2ix))

        # specify different lr
        param_optimizer = list(model.named_parameters())
        crf_name_list = ['crf_layer']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in crf_name_list)], 'lr': 1e-2},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in crf_name_list)], 'lr': 5e-5}
        ]
        # To reproduce BertAdam specific behavior set correct_bias=False
        optimizer = AdamW(
            optimizer_grouped_parameters,
            eps=1e-8,
            correct_bias=False,
            # weight_decay=1e-2,
        )
    else:
        model = BertForTokenClassification.from_pretrained(args.PRE_MODEL, num_labels=len(lab2ix))

        # To reproduce BertAdam specific behavior set correct_bias=False
        optimizer = AdamW(
            model.parameters(),
            lr=5e-5,
            eps=1e-8,
            correct_bias=False
        )
    model.to(device)

    # PyTorch scheduler
    num_epoch_steps = len(train_dataloader)
    num_finetuning_steps = args.NUM_FINE_EPOCHS * num_epoch_steps
    num_training_steps = args.NUM_EPOCHS * num_epoch_steps
    warmup_ratio = 0.1

    if args.scheduled_lr:
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

    pulse_delta = 1.0 / args.NUM_EPOCHS

    best_dev_score = float('-inf') if args.save_best == 'f1' else float('inf')

    save_step_interval = math.ceil(num_epoch_steps / 4)

    for param_group in optimizer.param_groups:
        print(param_group['lr'])
        print(len(param_group['params']))
        print()
    print("freeze embedding:", args.freeze_embed)
    for epoch in range(1, args.NUM_EPOCHS + 1):

        if args.bottomup_freeze:
            freeze_bert_layers(model, bert_name='bert', freeze_embed=args.freeze_embed, layer_list=list(range(0, epoch - 1)))

        if args.EPOCH_FREEZE != 0 and epoch > args.EPOCH_FREEZE:
            freeze_bert_layers(model, bert_name='bert', freeze_embed=args.freeze_embed, layer_list=list(range(0, 11)))


        epoch_loss = 0.0
        pbar = tqdm(enumerate(BackgroundGenerator(train_dataloader)), total=len(train_dataloader))

        for step, (batch_feat, batch_mask, batch_lab) in pbar:

        # for step, (batch_feat, batch_mask, batch_lab) in enumerate(tqdm(train_dataloader, desc='Training'), start=1):

            model.train()

            # BERT loss, logits: (batch_size, seq_len, tag_num)
            if args.pulse_freeze:
                pulse_freeze_bert(model, pulse_delta, freeze_embed=args.freeze_embed)

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

            if args.scheduled_lr:
                scheduler.step()
            model.zero_grad()

            pbar.set_description("Epoch: {}/{} | Training Loss: {:.6f}".format(
                epoch, args.NUM_EPOCHS, loss.item()
            ))

            if ((step + 1) % save_step_interval == 0) or ((step + 1) == num_epoch_steps):
                if args.save_best == 'loss':
                    dev_loss = 0.0

                    model.eval()
                    with torch.no_grad():
                        for dev_feat, dev_mask, dev_lab in dev_dataloader:
                            if args.do_crf:
                                dev_loss += model(dev_feat, attention_mask=dev_mask, labels=dev_lab)
                            else:
                                dev_loss += model(dev_feat, attention_mask=dev_mask, labels=dev_lab)[0]

                    if best_dev_score > (dev_loss / len(dev_dataloader)):

                        print("-> best dev loss %.4f; current loss %.4f; best model saved '%s'" % (
                            best_dev_score,
                            dev_loss / len(dev_dataloader),
                            model_dir
                        ))
                        best_dev_score = dev_loss / len(dev_dataloader)

                        """ save the best model """
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)
                        model.save_pretrained(model_dir)
                        tokenizer.save_pretrained(model_dir)
                elif args.save_best == 'f1':
                    if args.do_crf:
                        eval_crf(model, tokenizer, dev_dataloader, dev_deunk_loader, lab2ix, args.dev_output, args.joint)
                        eval_crf(model, tokenizer, test_dataloader, test_deunk_loader, lab2ix, args.test_output,
                                 args.joint)
                    else:
                        eval_seq(model, tokenizer, dev_dataloader, dev_deunk_loader, lab2ix, args.dev_output, args.joint)
                    import subprocess
                    eval_out = subprocess.check_output(
                        ['./ner_eval.sh', args.dev_output]
                    ).decode("utf-8").split('\n')[2]
                    dev_f1 = float(eval_out.split()[-1])
                    print("current dev f1: {:2f}".format(dev_f1))

                    test_out = subprocess.check_output(
                        ['./ner_eval.sh', args.test_output]
                    ).decode("utf-8").split('\n')[2]
                    test_f1 = float(test_out.split()[-1])
                    print("current test f1: {:2f}".format(test_f1))

                    if best_dev_score < dev_f1:
                        print("-> best dev f1 %.4f; current f1 %.4f; best model saved '%s'" % (
                            best_dev_score,
                            dev_f1,
                            model_dir
                        ))
                        best_dev_score = dev_f1

                        """ save the best model """
                        save_bert(model, tokenizer, model_dir)

    if not args.save_best:
        save_bert(model, tokenizer, model_dir)

    if args.later_eval:
        if args.do_crf:
            model = BertCRF.from_pretrained(model_dir)
            model.to(device)
            eval_crf(model, tokenizer, test_dataloader, test_deunk_loader, lab2ix, args.test_output, args.joint)
        else:
            model = BertForTokenClassification.from_pretrained(model_dir)
            model.to(device)
            eval_seq(model, tokenizer, test_dataloader, test_deunk_loader, lab2ix, args.test_output, args.joint)
        import subprocess
        eval_out = subprocess.check_output(
            ['./ner_eval.sh', args.test_output]
        ).decode("utf-8")
        print("epoch loss: %.6f; " % (epoch_loss/len(train_dataloader)))
        # print(eval_out.split('\n')[2])
        print(eval_out)
        eval_modality(args.test_output)


else:
    """ load the new tokenizer"""
    print("test_mode:", model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False, do_basic_tokenize=False)
    # test_tensors, test_deunk = extract_ner_from_conll(TEST_FILE, tokenizer, lab2ix, device)
    # test_dataloader = DataLoader(test_tensors, batch_size=args.BATCH_SIZE, shuffle=False)
    # test_deunk_loader = [test_deunk[i: i + args.BATCH_SIZE] for i in range(0, len(test_deunk), args.BATCH_SIZE)]
    # print('test size: %i' % len(test_tensors))
    #
    # dev_tensors, dev_deunk = extract_ner_from_conll(DEV_FILE, tokenizer, lab2ix, device, is_merged=args.joint)
    # dev_dataloader = DataLoader(dev_tensors, batch_size=args.BATCH_SIZE, shuffle=False)
    # dev_deunk_loader = [dev_deunk[i: i + args.BATCH_SIZE] for i in range(0, len(dev_deunk), args.BATCH_SIZE)]
    # print('dev size: %i' % len(dev_tensors))

    """ load the new model"""
    if args.do_crf:
        model = BertCRF.from_pretrained(model_dir)
    else:
        model = BertForTokenClassification.from_pretrained(model_dir)
    model.to(device)

    """ predict test out """
    # if not args.do_crf:
    #     eval_seq(model, tokenizer, dev_dataloader, dev_deunk_loader, lab2ix, args.dev_output, args.joint)
    # else:
    #     eval_crf(model, tokenizer, dev_dataloader, dev_deunk_loader, lab2ix, args.dev_output, args.joint)

    if not args.do_crf:
        eval_seq(model, tokenizer, test_dataloader, test_deunk_loader, lab2ix, args.test_output, args.joint)
    else:
        eval_crf(model, tokenizer, test_dataloader, test_deunk_loader, lab2ix, args.test_output, args.joint)

    # import subprocess
    #
    # dev_score = subprocess.check_output(
    #     ['./ner_eval.sh', args.dev_output]
    # ).decode("utf-8")
    # # print(eval_out.split('\n')[2])
    # print(dev_score)
    # eval_modality(args.dev_output)
    #
    # import subprocess
    #
    # test_score = subprocess.check_output(
    #     ['./ner_eval.sh', args.test_output]
    # ).decode("utf-8")
    # # print(eval_out.split('\n')[2])
    # print(test_score)
    # eval_modality(args.test_output)
