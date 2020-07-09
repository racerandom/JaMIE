#!/usr/bin/env python
# coding: utf-8
import warnings

import os
import argparse
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss
from transformers import *
from model import *

import utils
warnings.filterwarnings("ignore")


def eval_mod(model, eval_dataloader, eval_tok, eval_ner, eval_mod, ix2mod,
             cls_max_len, gpu_id, message, print_details, print_general, out_file='tmp.conll',
             f1_mode='micro'):

    mod_evaluator = utils.TupleEvaluator()
    model.eval()
    with torch.no_grad(), open(out_file, 'w') as fo:
        for dev_step, dev_batch in enumerate(eval_dataloader):
            b_toks, b_attn_mask, b_ner, b_mod = tuple(
                t.cuda(gpu_id) for t in dev_batch[1:]
            )
            b_sent_ids = dev_batch[0].tolist()
            b_text_list = [utils.padding_1d(
                eval_tok[sent_id],
                cls_max_len,
                pad_tok='[PAD]') for sent_id in b_sent_ids]

            b_ner_text = [eval_ner[sent_id] for sent_id in b_sent_ids]
            b_ner_tuple = utils.ner2tuple(b_sent_ids, b_ner_text)

            b_gold_mod = [eval_mod[sent_id] for sent_id in b_sent_ids]

            mod_logits = model(b_toks, attention_mask=b_attn_mask.bool())[0]
            b_pred_mod_ix = utils.decode_tensor_prediction(mod_logits.argmax(-1), b_attn_mask.bool())
            b_pred_mod = [[ix2mod[tag] for tag in sent_mod] for sent_mod in b_pred_mod_ix]

            b_pred_mod_tuple = [p + [b_pred_mod[b_sent_ids.index(p[0])][p[1][-1]]]
                                for p in b_ner_tuple if p[-1] != 'O']
            b_gold_mod_tuple = [g + [b_gold_mod[b_sent_ids.index(g[0])][g[1][-1]]]
                                for g in b_ner_tuple if g[-1] != 'O']
            mod_evaluator.update(b_gold_mod_tuple, b_pred_mod_tuple)

        mod_f1 = mod_evaluator.print_results(message + ' mod', print_details=print_details, print_general=print_general,
                                             f1_mode=f1_mode)
        return (mod_f1,)


def main():

    parser = argparse.ArgumentParser(description='PRISM joint recognizer')

    # parser.add_argument("--train_file", default="data/NCC1K20200601/cv1_train_juman.conll", type=str,
    #                     help="train file, multihead conll format.")
    #
    # parser.add_argument("--dev_file", default="data/NCC1K20200601/cv1_dev_juman.conll", type=str,
    #                     help="dev file, multihead conll format.")
    #
    # parser.add_argument("--test_file", default="ncc.ner.conll", type=str,
    #                     help="test file, multihead conll format.")
    #
    # parser.add_argument("--pred_file", default="ncc.mod.conll", type=str,
    #                     help="test prediction, multihead conll format.")
    #
    # parser.add_argument("--save_model", default='checkpoints/ncc/mod', type=str,
    #                     help="save/load model dir")

    parser.add_argument("--train_file", default="data/clinical20200605/cv2_train_juman.conll", type=str,
                        help="train file, multihead conll format.")

    parser.add_argument("--dev_file", default="data/clinical20200605/cv2_dev_juman.conll", type=str,
                        help="dev file, multihead conll format.")

    parser.add_argument("--test_file", default="data/clinical20200605/cv2_test_juman.conll", type=str,
                        help="test file, multihead conll format.")

    parser.add_argument("--pred_file", default="mr.mod.conll", type=str,
                        help="test prediction, multihead conll format.")

    parser.add_argument("--save_model", default='checkpoints/mr/mod', type=str,
                        help="save/load model dir")

    parser.add_argument("--pretrained_model",
                        default="/home/feicheng/Tools/Japanese_L-12_H-768_A-12_E-30_BPE",
                        type=str,
                        help="pre-trained model dir")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="tokenizer: do_lower_case")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="BATCH SIZE")

    parser.add_argument("--num_epoch", default=15, type=int,
                        help="fine-tuning epoch number")

    parser.add_argument("--embed_size", default='[32, 32, 384]', type=str,
                        help="ner, mod, rel embedding size")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--freeze_after_epoch", default=15, type=int,
                        help="freeze encoder after N epochs")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--lr", default=5e-5, type=float,
                        help="learning rate")

    parser.add_argument("--reduction", default='token_mean', type=str,
                        help="loss reduction: `token_mean` or `sum`")

    parser.add_argument("--save_best", default='f1', type=str,
                        help="save the best model, given dev scores (f1 or loss)")

    parser.add_argument("--save_step_portion", default=4, type=int,
                        help="save best model given a portion of steps")

    parser.add_argument("--neg_ratio", default=1.0, type=float,
                        help="negative sample ratio")

    parser.add_argument("--scheduled_lr",
                        default=True,
                        type=bool,
                        help="learning rate schedule")

    parser.add_argument("--epoch_eval",
                        action='store_true',
                        help="eval each epoch")

    parser.add_argument("--fp16",
                        action='store_true',
                        help="fp16")

    parser.add_argument("--fp16_opt_level", type=str, default="O2",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--gpu_id", default=0, type=int,
                        help="gpu id: default 0")

    args = parser.parse_args()

    print(args)

    bio_emb_size, mod_emb_size, rel_emb_size = eval(args.embed_size)

    if args.do_train:
        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model,
            do_lower_case=args.do_lower_case,
            do_basic_tokenize=False
        )
        tokenizer.add_tokens(['[JASP]'])
    else:
        tokenizer = BertTokenizer.from_pretrained(
            args.save_model,
            do_lower_case=args.do_lower_case,
            do_basic_tokenize=False
        )

    train_toks, train_ners, train_mods, train_rels, bio2ix, ne2ix, mod2ix, rel2ix = utils.extract_rel_data_from_mh_conll_v2(
        args.train_file,
        down_neg=0.0
    )

    print(bio2ix)
    print(rel2ix)
    print(mod2ix)
    print()

    print('max sent len:', utils.max_sents_len(train_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in train_rels]), max([len(sent_rels) for sent_rels in train_rels]))
    print()

    dev_toks, dev_ners, dev_mods, dev_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(args.dev_file, down_neg=0.0)
    print('max sent len:', utils.max_sents_len(dev_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in dev_rels]), max([len(sent_rels) for sent_rels in dev_rels]))
    print()

    ix2mod = {v: k for k, v in mod2ix.items()}

    from collections import defaultdict

    rel_count = defaultdict(lambda: 0)

    for sent_rels in train_rels:
        for rel in sent_rels:
            rel_count[rel[-1]] += 1

    for sent_rels in dev_rels:
        for rel in sent_rels:
            rel_count[rel[-1]] += 1

    example_id = 15
    print('Random example: id %i, len: %i' % (example_id, len(train_toks[example_id])))
    for tok_id in range(len(train_toks[example_id])):
        print("%i\t%10s\t%s" % (tok_id, train_toks[example_id][tok_id], train_ners[example_id][tok_id]))
    print(train_rels[example_id])
    print()

    max_len = max(
        utils.max_sents_len(train_toks, tokenizer),
        utils.max_sents_len(dev_toks, tokenizer)
    )
    cls_max_len = max_len + 2

    train_dataset, train_tok, train_ner, train_mod, train_rel, train_spo = utils.convert_rels_to_mhs_v3(
        train_toks, train_ners, train_mods, train_rels,
        tokenizer, bio2ix, mod2ix, rel2ix, max_len, verbose=0)

    dev_dataset, dev_tok, dev_ner, dev_mod, dev_rel, dev_spo = utils.convert_rels_to_mhs_v3(
        dev_toks, dev_ners, dev_mods, dev_rels,
        tokenizer, bio2ix, mod2ix, rel2ix, max_len, verbose=0)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    if args.do_train:
        num_epoch_steps = len(train_dataloader)
        num_training_steps = args.num_epoch * num_epoch_steps
        warmup_ratio = 0.1
        save_step_interval = math.ceil(num_epoch_steps / args.save_step_portion)

        model = BertForTokenClassification.from_pretrained(args.pretrained_model, num_labels=len(mod2ix))

        model.bert.resize_token_embeddings(len(tokenizer))

        param_optimizer = list(model.named_parameters())
        encoder_name_list = ['bert']
        crf_name_list = ['crf_layer']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in encoder_name_list)],
                'lr': args.lr
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in crf_name_list)],
                'lr': 1e-2
            },
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in encoder_name_list + crf_name_list)],
                'lr': 1e-3
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=1e-8,
            correct_bias=False
        )
        if args.scheduled_lr:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_training_steps * warmup_ratio,
                num_training_steps=num_training_steps
            )

        model.cuda(args.gpu_id)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # (F1, NER_F1, MOD_F1, REL_F1, epoch, step)
        best_dev_f1 = (float('-inf'), 0, 0)

        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            print(len(param_group['params']))
            print()

        for epoch in range(1, args.num_epoch + 1):

            train_loss = .0
            pbar = tqdm(enumerate(BackgroundGenerator(train_dataloader)), total=len(train_dataloader))
            for step, batch in pbar:
                model.train()

                b_toks, b_attn_mask, b_ner, b_mod = tuple(
                    t.cuda(args.gpu_id) for t in batch[1:]
                )
                b_sent_ids = batch[0].tolist()

                b_text_list = [utils.padding_1d(
                    train_tok[sent_id],
                    cls_max_len,
                    pad_tok='[PAD]') for sent_id in b_sent_ids]

                loss = model(b_toks, attention_mask=b_attn_mask.bool(), labels=b_mod)
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

                train_loss += loss.item()
                pbar.set_description("L_NER: {:.6f} | epoch: {}/{}:".format(
                    loss.item(), epoch, args.num_epoch
                ))

                if epoch > 4:
                    if ((step + 1) % save_step_interval == 0) or (step == num_epoch_steps - 1):
                        dev_f1 = eval_mod(model, dev_dataloader, dev_tok, dev_ner, dev_mod, ix2mod, cls_max_len, args.gpu_id,
                                          "dev dataset",
                                          print_details=False,
                                          print_general=False)
                        dev_f1 += (epoch,)
                        dev_f1 += (step,)
                        if best_dev_f1[0] < dev_f1[0]:
                            print(
                                " -> Previous best dev f1 {:.6f}; epoch {:d} / step {:d} \n"
                                " >> Current f1 {:.6f}; best model saved '{}'".format(
                                    best_dev_f1[0],
                                    best_dev_f1[1],
                                    best_dev_f1[2],
                                    dev_f1[0],
                                    args.save_model
                                )
                            )
                            best_dev_f1 = dev_f1

                            """ save the best model """
                            if not os.path.exists(args.save_model):
                                os.makedirs(args.save_model)
                            torch.save(model.state_dict(), os.path.join(args.save_model, 'best.pt'))
                            tokenizer.save_pretrained(args.save_model)

            print('Epoch %i, train loss: %.6f\n' % (
                epoch,
                train_loss / num_epoch_steps,
            ))

        print("""Best dev f1 {:.6f}; epoch {:d} / step {:d}\n""".format(
            best_dev_f1[0],
            best_dev_f1[1],
            best_dev_f1[2],
        ))
    else:
        model = BertForTokenClassification.from_pretrained(args.pretrained_model, num_labels=len(bio2ix))

        model.bert.resize_token_embeddings(len(tokenizer))
        model.cuda(args.gpu_id)
        model.load_state_dict(torch.load(os.path.join(args.save_model, 'best.pt')))

        test_toks, test_ners, test_mods, test_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(args.test_file,
                                                                                                         down_neg=0.0)
        print('max sent len:', utils.max_sents_len(test_toks, tokenizer))
        print(min([len(sent_rels) for sent_rels in test_rels]), max([len(sent_rels) for sent_rels in test_rels]))
        print()
        max_len = max(
            max_len,
            utils.max_sents_len(test_toks, tokenizer)
        )

        test_dataset, test_tok, test_ner, test_mod, test_rel, test_spo = utils.convert_rels_to_mhs_v3(
            test_toks, test_ners, test_mods, test_rels,
            tokenizer, bio2ix, mod2ix, rel2ix, max_len, verbose=0)

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        dev_f1 = eval_mod(model, test_dataloader, test_tok, test_ner, test_mod, ix2mod, cls_max_len, args.gpu_id,
                          "test dataset",
                          print_details=True,
                          print_general=True)


if __name__ == '__main__':
    main()

