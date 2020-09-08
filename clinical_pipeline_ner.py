#!/usr/bin/env python
# coding: utf-8
import warnings
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
from transformers import *
import argparse
from model import *
warnings.filterwarnings("ignore")


def output_ner(model, eval_dataloader, eval_tok, eval_ner, ner2ix, ner_outfile, device):
    ix2ner = {v: k for k, v in ner2ix.items()}
    model.eval()
    with torch.no_grad(), open(ner_outfile, 'w') as fo:
        for dev_step, dev_batch in enumerate(eval_dataloader):
            b_toks, b_attn_mask, b_ner, b_mod = tuple(
                t.to(device) for t in dev_batch[1:]
            )
            b_sent_ids = dev_batch[0].tolist()
            b_text_list = [utils.padding_1d(
                eval_tok[sent_id],
                cls_max_len,
                pad_tok='[PAD]') for sent_id in b_sent_ids]

            b_gold_ner = [eval_ner[sent_id] for sent_id in b_sent_ids]
            pred_ix = [tags[1:-1] for tags in model.decode(b_toks, attention_mask=b_attn_mask.bool())]

            for sent_id, toks, mask, p in zip(b_sent_ids, b_toks.cpu().tolist(), b_attn_mask.cpu().tolist(), pred_ix):
                print(len(eval_tok[sent_id]), eval_tok[sent_id])
                print(len(eval_ner[sent_id]), eval_ner[sent_id])
                print(toks)
                print(sum(mask))
                print(len(p), p)
                print()


""" 
python input arguments 
"""
parser = argparse.ArgumentParser(description='Clinical IE pipeline NER')

parser.add_argument("--pretrained_model",
                    default="/home/feicheng/Tools/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12",
                    type=str,
                    help="pre-trained model dir")

parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="tokenizer: do_lower_case")

parser.add_argument("--saved_model", default='checkpoints/tmp/pipeline/ner', type=str,
                    help="save/load model dir")

parser.add_argument("--train_file", default="data/i2b2/i2b2_training.conll", type=str,
                    help="train file, multihead conll format.")

parser.add_argument("--dev_file", default="data/i2b2/i2b2_dev.conll", type=str,
                    help="dev file, multihead conll format.")

parser.add_argument("--test_file", default="data/i2b2/i2b2_test.conll", type=str,
                    help="test file, multihead conll format.")

parser.add_argument("--batch_size", default=8, type=int,
                    help="BATCH SIZE")

parser.add_argument("--num_epoch", default=15, type=int,
                    help="fine-tuning epoch number")

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")

parser.add_argument("--do_crf",
                    action='store_true',
                    help="Whether to use CRF.")

parser.add_argument("--enc_lr", default=5e-5, type=float,
                    help="encoder lr")

parser.add_argument("--crf_lr", default=1e-2, type=float,
                    help="crf layer lr")

parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")

parser.add_argument("--test_output", default='tmp/test.ner', type=str,
                    help="test output filename")

parser.add_argument("--dev_output", default='tmp/dev.ner', type=str,
                    help="dev output filename")

parser.add_argument("--later_eval",
                    action='store_true',
                    help="Whether eval model every epoch.")

parser.add_argument("--save_best", action='store', type=str, default='f1',
                    help="save the best model, given dev scores (f1 or loss)")

parser.add_argument("--save_step_portion", default=4, type=int,
                        help="save best model given a portion of steps")

parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="warmup ratio")

parser.add_argument("--fp16",
                    action='store_true',
                    help="fp16")

parser.add_argument("--fp16_opt_level", type=str, default="O1",
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                    "See details at https://nvidia.github.io/apex/amp.html")

parser.add_argument("--scheduled_lr",
                    action='store_true',
                    help="learning rate schedule")

parser.add_argument("--joint",
                    action='store_true',
                    help="merge ner and modality jointly")

args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', args.device)
args.n_gpu = torch.cuda.device_count()

if args.do_train:
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, do_lower_case=args.do_lower_case, do_basic_tokenize=False)

    """ Read conll file for counting statistics, such as: [UNK] token ratio, label2ix, etc. """
    train_comments, train_toks, train_ners, train_mods, train_rels, bio2ix, ne2ix, mod2ix, rel2ix = utils.extract_rel_data_from_mh_conll_v2(
        args.train_file,
        down_neg=0.0
    )
    max_len_train = utils.max_sents_len(train_toks, tokenizer)
    print(bio2ix)
    print()
    print('max training sent len:', max_len_train)
    print()

    dev_comments, dev_toks, dev_ners, dev_mods, dev_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(
        args.dev_file,
        down_neg=0.0
    )
    max_len_dev = utils.max_sents_len(dev_toks, tokenizer)
    print('max dev sent len:', )
    print()

    max_len = max(max_len_train, max_len_dev)
    cls_max_len = max_len + 2
    print(f"max seq len: {max_len}, max seq len with [CLS] and [SEP]: {cls_max_len}")

    example_id = 15

    print(f"Random example: id {example_id}, len: {len(train_toks[example_id])}")
    for tok_id in range(len(train_toks[example_id])):
        print(f"{tok_id}\t{train_toks[example_id][tok_id]}\t{train_ners[example_id][tok_id]}")
    print(train_rels[example_id])
    print()

    """ 
    - Generate train/test tensors including (token_ids, mask_ids, label_ids) 
    - wrap them into dataloader for mini-batch cutting
    """
    train_dataset, train_tok, train_ner, train_mod, train_rel, train_spo = utils.convert_rels_to_mhs_v3(
        train_toks, train_ners, train_mods, train_rels,
        tokenizer, bio2ix, mod2ix, rel2ix, max_len, verbose=0)

    dev_dataset, dev_tok, dev_ner, dev_mod, dev_rel, dev_spo = utils.convert_rels_to_mhs_v3(
        dev_toks, dev_ners, dev_mods, dev_rels,
        tokenizer, bio2ix, mod2ix, rel2ix, max_len, verbose=0)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    """
    Model
    """
    model = BertCRF.from_pretrained(args.pretrained_model, num_labels=len(bio2ix))

    # specify different lr
    param_optimizer = list(model.named_parameters())
    crf_name_list = ['crf_layer']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in crf_name_list)], 'lr': args.crf_lr},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in crf_name_list)], 'lr': args.enc_lr}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        correct_bias=False
        # weight_decay=1e-2,
    )
    model.to(args.device)

    # PyTorch scheduler
    num_epoch_steps = len(train_dataloader)
    num_training_steps = args.num_epoch * num_epoch_steps
    save_step_interval = math.ceil(num_epoch_steps / args.save_step_portion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps * args.warmup_ratio,
        num_training_steps=num_training_steps
    )

    # support fp16
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    best_dev_score = float('-inf') if args.save_best == 'f1' else float('inf')

    save_step_interval = math.ceil(num_epoch_steps / 4)

    for epoch in range(1, args.num_epoch + 1):

        epoch_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", total=len(train_dataloader))
        for step, batch in enumerate(epoch_iterator):

            model.train()

            b_toks, b_attn_mask, b_ner, b_mod = tuple(
                t.to(args.device) for t in batch[1:]
            )

            # BERT loss, logits: (batch_size, seq_len, tag_num)
            loss = model(b_toks, attention_mask=b_attn_mask.bool(), labels=b_ner)

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

            epoch_iterator.set_description(
                f"L_NER: {epoch_loss / (step + 1):.6f} | epoch: {epoch}/{args.num_epoch}:"
            )

            if ((step + 1) % save_step_interval == 0) or ((step + 1) == num_epoch_steps):
                output_ner(model, dev_dataloader, dev_tok, dev_ner, bio2ix, args.dev_output, args.device)
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
                        args.args.saved_model
                    ))
                    best_dev_score = dev_f1

                    """ save the best model """
                    if not os.path.exists(args.saved_model):
                        os.makedirs(args.saved_model)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), os.path.join(args.saved_model, 'best.pt'))
                    tokenizer.save_pretrained(args.saved_model)
                    with open(os.path.join(args.saved_model, 'ner2ix.json'), 'w') as fp:
                        json.dump(bio2ix, fp)

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
