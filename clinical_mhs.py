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


def main():
    parser = argparse.ArgumentParser(description='PRISM tag recognizer')

    parser.add_argument("--train_file", default="data/CoNLL04/train.txt", type=str,
                        help="train file, multihead conll format.")

    parser.add_argument("--dev_file", default="data/CoNLL04/dev.txt", type=str,
                        help="dev file, multihead conll format.")

    parser.add_argument("--test_file", default="data/CoNLL04/test.txt", type=str,
                        help="test file, multihead conll format.")

    parser.add_argument("--pretrained_model",
                        default='bert-base-uncased',
                        type=str,
                        help="pre-trained model dir")

    parser.add_argument("--do_lower_case",
                        # action='store_True',
                        default=True,
                        type=bool,
                        help="tokenizer: do_lower_case")

    # parser.add_argument("--train_file", default="data/clinical2020Q1/cv4_train.conll", type=str,
    #                     help="train file, multihead conll format.")
    #
    # parser.add_argument("--dev_file", default="data/clinical2020Q1/cv4_dev.conll", type=str,
    #                     help="dev file, multihead conll format.")
    #
    # parser.add_argument("--test_file", default="data/clinical2020Q1/cv4_test.conll", type=str,
    #                     help="test file, multihead conll format.")
    #
    # parser.add_argument("--pretrained_model",
    #                     default="/home/feicheng/Tools/Japanese_L-12_H-768_A-12_E-30_BPE",
    #                     type=str,
    #                     help="pre-trained model dir")
    #
    # parser.add_argument("--do_lower_case",
    #                     # action='store_True',
    #                     default=False,
    #                     type=bool,
    #                     help="tokenizer: do_lower_case")

    parser.add_argument("--save_model", default='checkpoints/rel', type=str,
                        help="save/load model dir")

    parser.add_argument("--batch_size", default=8, type=int,
                        help="BATCH SIZE")

    parser.add_argument("--num_epoch", default=50, type=int,
                        help="fine-tuning epoch number")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--lr", default=5e-5, type=float,
                        help="learning rate")

    parser.add_argument("--ne_size", default=128, type=int,
                        help="size of name entity embedding")

    parser.add_argument("--save_best", default='f1', type=str,
                        help="save the best model, given dev scores (f1 or loss)")

    parser.add_argument("--save_step_interval", default=200, type=int,
                        help="save best model given a step interval")

    parser.add_argument("--neg_ratio", default=1.0, type=float,
                        help="negative sample ratio")

    parser.add_argument("--scheduled_lr",
                        # action='store_True',
                        default=False,
                        type=bool,
                        help="learning rate schedule")

    parser.add_argument("--epoch_eval",
                        # action='store_True',
                        default=True,
                        type=bool,
                        help="eval each epoch")

    args = parser.parse_args()

    print(args)

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model,
        do_lower_case=args.do_lower_case,
        do_basic_tokenize=False
    )

    tokenizer.add_tokens(['[JASP]'])

    train_toks, train_labs, train_rels, bio2ix, ne2ix, rel2ix = utils.extract_rel_data_from_mh_conll(args.train_file,
                                                                                                     0.0)
    print(bio2ix)
    print(ne2ix)
    print(rel2ix)
    print('max sent len:', utils.max_sents_len(train_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in train_rels]), max([len(sent_rels) for sent_rels in train_rels]))
    print()

    dev_toks, dev_labs, dev_rels, _, _, _ = utils.extract_rel_data_from_mh_conll(args.dev_file, 0.0)
    print('max sent len:', utils.max_sents_len(dev_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in dev_rels]), max([len(sent_rels) for sent_rels in dev_rels]))
    print()

    test_toks, test_labs, test_rels, _, _, _ = utils.extract_rel_data_from_mh_conll(args.test_file, 0.0)
    print('max sent len:', utils.max_sents_len(test_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in test_rels]), max([len(sent_rels) for sent_rels in test_rels]))
    print()

    ix2rel = {v: k for k, v in rel2ix.items()}
    ix2bio = {v: k for k, v in bio2ix.items()}

    from collections import defaultdict

    rel_count = defaultdict(lambda: 0)

    for sent_rels in train_rels:
        for rel in sent_rels:
            rel_count[rel[-1]] += 1

    for sent_rels in dev_rels:
        for rel in sent_rels:
            rel_count[rel[-1]] += 1

    for sent_rels in test_rels:
        for rel in sent_rels:
            rel_count[rel[-1]] += 1

    print(rel_count)

    example_id = 15
    print('Random example: id %i, len: %i' % (example_id, len(train_toks[example_id])))
    for tok_id in range(len(train_toks[example_id])):
        print("%i\t%10s\t%s" % (tok_id, train_toks[example_id][tok_id], train_labs[example_id][tok_id]))
    print(train_rels[example_id])
    print()

    max_len = max(
        utils.max_sents_len(train_toks, tokenizer),
        utils.max_sents_len(dev_toks, tokenizer),
        utils.max_sents_len(test_toks, tokenizer)
    )

    train_dataset = utils.convert_rels_to_mhs(train_toks, train_labs, train_rels,
                                              tokenizer, bio2ix, ne2ix, rel2ix, max_len, verbose=0)
    dev_dataset = utils.convert_rels_to_mhs(dev_toks, dev_labs, dev_rels,
                                            tokenizer, bio2ix, ne2ix, rel2ix, max_len, verbose=1)
    test_dataset = utils.convert_rels_to_mhs(test_toks, test_labs, test_rels,
                                             tokenizer, bio2ix, ne2ix, rel2ix, max_len, verbose=0)

    # from collections import Counter
    # import json
    # word_vocab = Counter()
    #
    # utils.convert_rels_to_pmhs(train_toks, train_labs, train_rels,
    #                            tokenizer, rel2ix, "tmp/train_cv4.txt", word_vocab)
    # utils.convert_rels_to_pmhs(dev_toks, dev_labs, dev_rels,
    #                            tokenizer, rel2ix, "tmp/dev_cv4.txt", word_vocab)
    # utils.convert_rels_to_pmhs(test_toks, test_labs, test_rels,
    #                            tokenizer, rel2ix, "tmp/test_cv4.txt", word_vocab)
    # utils.gen_vocab(word_vocab, "tmp/word_vocab.json")
    # json.dump(rel2ix, open("tmp/relation_vocab.json", 'w'), ensure_ascii=False)
    #
    # print("pmhs data generated")

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.batch_size)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    num_epoch_steps = len(train_dataloader)
    num_training_steps = args.num_epoch * num_epoch_steps
    warmup_ratio = 0.1

    # rel2ix = {'Located_In': 0, 'Work_For': 1, 'Live_In': 2, 'OrgBased_In': 3, 'Kill': 4}

    model = HeadSelectModel.from_pretrained(
        args.pretrained_model,
        ner_emb_dim=50,
        rel_emb_dim=100,
        ner_num_labels=len(bio2ix),
        rel_num_labels=len(rel2ix),
        rel_prob_threshold=0.5
    )

    # model = BertCRF.from_pretrained(args.PRE_MODEL, num_labels=len(bio2ix))

    param_optimizer = list(model.named_parameters())
    bert_name_list = ['bert']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in bert_name_list)], 'lr': 5e-5},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in bert_name_list)], 'lr': 1e-3}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        correct_bias=False
    )
    if args.scheduled_lr:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps * warmup_ratio,
            num_training_steps=num_training_steps
        )

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    best_dev_f1 = float('-inf')

    for epoch in range(1, args.num_epoch + 1):
        # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        train_loss, train_ner_loss, train_rel_loss = .0, .0, .0
        pbar = tqdm(enumerate(BackgroundGenerator(train_dataloader)), total=len(train_dataloader))
        for step, batch in pbar:
            model.train()

            if epoch > 15:
                utils.freeze_bert_layers(model, freeze_embed=True, layer_list=list(range(0, 12)))

            b_toks, b_attn_mask, b_ner, b_matrix_rel = tuple(
                t.to(device) for t in batch[1:]
            )
            b_sent_ids = batch[0]
            print(b_sent_ids)
            # print(b_sent_ids.tolist())
            ner_loss, rel_loss = model(b_toks, b_attn_mask.bool(), ner_labels=b_ner, rel_labels=b_matrix_rel)
            loss = ner_loss + rel_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if args.scheduled_lr:
                scheduler.step()
            model.zero_grad()
            train_loss += loss.item()
            train_ner_loss += ner_loss.item()
            train_rel_loss += rel_loss.item()
            pbar.set_description("L {:.6f}, L_CRF: {:.6f}, L_REL: {:.6f}".format(loss.item(), ner_loss.item(), rel_loss.item()))
        print('Epoch %i, train loss: %.6f, training ner_loss: %.6f, rel_loss: %.6f\n' % (
            epoch,
            train_loss / num_epoch_steps,
            train_ner_loss / num_epoch_steps,
            train_rel_loss / num_epoch_steps
        ))

        if args.epoch_eval:
            pred_rels, gold_rels = [], []
            pred_ners, gold_ners = [], []
            model.eval()
            with torch.no_grad():
                for dev_step, dev_batch in enumerate(dev_dataloader):
                    b_toks, b_attn_mask, b_ner, b_gold_relmat = tuple(
                        t.to(device) for t in dev_batch[1:]
                    )
                    b_sent_ids = dev_batch[0].tolist()
                    b_pred_ner, b_pred_relmat = model(b_toks, b_attn_mask.bool())
                    # tuples: [[step_id, batch_id, tail_id, head_id, rel], ...]
                    b_pred_ner_tuples = utils.ner2tuple(b_sent_ids, b_pred_ner, ix2bio)
                    pred_ners += b_pred_ner_tuples
                    b_gold_ner = utils.batch_demask(b_ner, b_attn_mask.bool())
                    b_gold_ner_tuples = utils.ner2tuple(b_sent_ids, b_gold_ner, ix2bio)
                    gold_ners += b_gold_ner_tuples
                    for b_id in range(len(b_sent_ids)):

                        sent_pred_rels = [[b_sent_ids[b_id]] + sent_pred_relmat
                                          for sent_pred_relmat in torch.nonzero(b_pred_relmat[b_id]).tolist()
                                          if sent_pred_relmat[-1] != rel2ix['N']]
                        pred_rels += sent_pred_rels

                        sent_gold_rels = [[b_sent_ids[b_id]] + sent_gold_relmat
                                          for sent_gold_relmat in torch.nonzero(b_gold_relmat[b_id]).tolist()
                                          if sent_gold_relmat[-1] != rel2ix['N']]
                        gold_rels += sent_gold_rels
            print(len(gold_rels), len(pred_rels))
            utils.evaluate_tuples(
                pred_ners,
                gold_ners,
                ix2bio
            )
            utils.evaluate_tuples(
                pred_ners,
                gold_ners,
                ix2bio
            )

        # utils.evaluate_ner(
        #     pred_ner,
        #     gold_ner,
        #     bio2ix
        # )
        # utils.evaluate_tuples(
        #     pred_rels,
        #     gold_rels,
        #     ix2rel
        # )


if __name__ == '__main__':
    main()

