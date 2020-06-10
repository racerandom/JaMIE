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


def eval_mhs(model, eval_dataloader, eval_tok, eval_lab, eval_rel, eval_spo, bio2ix, rel2ix, cls_max_len, device,
             message, ner_details, rel_details, print_general, f1_mode='micro', verbose=0):
    ix2bio = {v: k for k, v in bio2ix.items()}
    ix2rel = {v: k for k, v in rel2ix.items()}
    ner_evaluator = utils.TupleEvaluator()
    rel_evaluator = utils.TupleEvaluator()
    model.eval()
    with torch.no_grad():
        for dev_step, dev_batch in enumerate(eval_dataloader):
            b_toks, b_attn_mask, b_ner = tuple(
                t.to(device) for t in dev_batch[1:]
            )
            b_sent_ids = dev_batch[0].tolist()
            b_text_list = [utils.padding_1d(
                eval_tok[sent_id],
                cls_max_len,
                pad_tok='[PAD]') for sent_id in b_sent_ids]

            b_bio_text = [eval_lab[sent_id] for sent_id in b_sent_ids]

            b_spo_gold = tuple([eval_spo[sent_id] for sent_id in b_sent_ids])
            if verbose:
                for sent_id in b_sent_ids:
                    print(["{}: {}".format(ix, tok) for ix, tok in enumerate(eval_tok[sent_id])])
                    print(["{}: {}".format(ix, lab) for ix, lab in enumerate(eval_lab[sent_id])])
                    print(eval_rel[sent_id])
                    print()

            b_gold_relmat = utils.gen_relmat(eval_rel, b_sent_ids, cls_max_len, rel2ix, del_neg=False).to(device)
            output = model(b_toks, b_attn_mask.bool(), b_ner, b_gold_relmat, b_text_list, b_bio_text, b_spo_gold, is_train=False)

            # ner tuple -> [sent_id, [ids], ner_lab]
            b_pred_ner, b_gold_ner = output['decoded_tag'], output['gold_tags']
            ner_evaluator.update(utils.ner2tuple(b_sent_ids, b_gold_ner), utils.ner2tuple(b_sent_ids, b_pred_ner))

            # rel: {'subject': [toks], 'predicate': rel, 'object': [toks]}
            b_pred_rel, b_gold_rel = output['selection_triplets'], output['spo_gold']
            b_pred_rel_tuples = [[sent_id, rel['subject'], rel['object'], rel['predicate']]
                                 for sent_id, sent_rel in zip(b_sent_ids, b_pred_rel) for rel in sent_rel]
            b_gold_rel_tuples = [[sent_id, rel['subject'], rel['object'], rel['predicate']]
                                 for sent_id, sent_rel in zip(b_sent_ids, b_gold_rel) for rel in sent_rel]
            rel_evaluator.update(b_gold_rel_tuples, b_pred_rel_tuples)

        ner_f1 = ner_evaluator.print_results(message, print_details=ner_details, print_general=print_general, f1_mode=f1_mode)
        rel_f1 = rel_evaluator.print_results(message, print_details=rel_details, print_general=print_general, f1_mode=f1_mode)
        f1 = (ner_f1 + rel_f1) / 2
        return f1, ner_f1, rel_f1


def main():

    parser = argparse.ArgumentParser(description='PRISM mhs recognizer')

    parser.add_argument("--train_file", default="data/i2b2/i2b2_training.conll", type=str,
                        help="train file, multihead conll format.")

    parser.add_argument("--dev_file", default="data/i2b2/i2b2_dev.conll", type=str,
                        help="dev file, multihead conll format.")

    parser.add_argument("--test_file", default="data/i2b2/i2b2_test.conll", type=str,
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

    # parser.add_argument("--train_file", default="data/clinical2020Q1/cv0_train.conll", type=str,
    #                     help="train file, multihead conll format.")
    #
    # parser.add_argument("--dev_file", default="data/clinical2020Q1/cv0_dev.conll", type=str,
    #                     help="dev file, multihead conll format.")
    #
    # parser.add_argument("--test_file", default="data/clinical2020Q1/cv0_test.conll", type=str,
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

    parser.add_argument("--save_model", default='checkpoints/mhs/', type=str,
                        help="save/load model dir")

    parser.add_argument("--batch_size", default=8, type=int,
                        help="BATCH SIZE")

    parser.add_argument("--num_epoch", default=50, type=int,
                        help="fine-tuning epoch number")

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
                        # action='store_True',
                        default=True,
                        type=bool,
                        help="learning rate schedule")

    parser.add_argument("--epoch_eval",
                        # action='store_True',
                        default=True,
                        type=bool,
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

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model,
        do_lower_case=args.do_lower_case,
        do_basic_tokenize=False
    )

    tokenizer.add_tokens(['[JASP]'])

    train_toks, train_labs, train_mods, train_rels, bio2ix, ne2ix, mod2ix, rel2ix = utils.extract_rel_data_from_mh_conll_v2(
        args.train_file,
        down_neg=0.0
    )
    print(bio2ix)
    print(ne2ix)
    print(rel2ix)
    print(mod2ix)
    print()

    print('max sent len:', utils.max_sents_len(train_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in train_rels]), max([len(sent_rels) for sent_rels in train_rels]))
    print()

    dev_toks, dev_labs, dev_mods, dev_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(args.dev_file, down_neg=0.0)
    print('max sent len:', utils.max_sents_len(dev_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in dev_rels]), max([len(sent_rels) for sent_rels in dev_rels]))
    print()

    test_toks, test_labs, test_mods, test_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(args.test_file, down_neg=0.0)
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
    cls_max_len = max_len + 2

    train_dataset, train_tok, train_lab, train_rel, train_spo = utils.convert_rels_to_mhs_v2(
        train_toks, train_labs, train_rels,
        tokenizer, bio2ix, rel2ix, max_len, verbose=0)
    dev_dataset, dev_tok, dev_lab, dev_rel, dev_spo = utils.convert_rels_to_mhs_v2(
        dev_toks, dev_labs, dev_rels,
        tokenizer, bio2ix, rel2ix, max_len, verbose=0)
    test_dataset, test_tok, test_lab, test_rel, test_spo = utils.convert_rels_to_mhs_v2(
        test_toks, test_labs, test_rels,
        tokenizer, bio2ix, rel2ix, max_len, verbose=0)

    # train_sampler = RandomSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    # dev_sampler = SequentialSampler(dev_dataset)
    # dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.batch_size)
    # test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
    # print(train_sampler)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    num_epoch_steps = len(train_dataloader)
    num_training_steps = args.num_epoch * num_epoch_steps
    warmup_ratio = 0.1
    save_step_interval = math.ceil(num_epoch_steps / args.save_step_portion)

    model = MultiHeadSelection(
        bert_url=args.pretrained_model,
        bio_emb_size=64,
        bio_vocab=bio2ix,
        rel_emb_size=416,
        relation_vocab=rel2ix,
        gpu_id=args.gpu_id
    )
    model.encoder.resize_token_embeddings(len(tokenizer))

    param_optimizer = list(model.named_parameters())
    encoder_name_list = ['encoder']
    crf_name_list = ['crf_tagger']
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

    model.to(device)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # (f1, NER_F1, REL_F1, epoch, step)
    best_dev_f1 = (float('-inf'), float('-inf'), float('-inf'), 0, 0)

    for param_group in optimizer.param_groups:
        print(param_group['lr'])
        print(len(param_group['params']))
        print()

    for epoch in range(1, args.num_epoch + 1):

        train_loss, train_ner_loss, train_rel_loss = .0, .0, .0
        pbar = tqdm(enumerate(BackgroundGenerator(train_dataloader)), total=len(train_dataloader))
        for step, batch in pbar:
            model.train()

            if epoch > args.freeze_after_epoch:
                utils.freeze_bert_layers(model, bert_name='encoder', freeze_embed=True, layer_list=list(range(0, 11)))

            b_toks, b_attn_mask, b_ner = tuple(
                t.to(device) for t in batch[1:]
            )
            b_sent_ids = batch[0].tolist()
            b_gold_relmat = utils.gen_relmat(train_rel, b_sent_ids, cls_max_len, rel2ix, del_neg=False).to(device)

            b_text_list = [utils.padding_1d(
                train_tok[sent_id],
                cls_max_len,
                pad_tok='[PAD]') for sent_id in b_sent_ids]

            b_bio_text = [train_lab[sent_id] for sent_id in b_sent_ids]

            b_spo_gold = tuple([train_spo[sent_id] for sent_id in b_sent_ids])

            # print(b_toks.shape)
            # print(b_ner.shape)
            # print(b_gold_relmat.shape)
            # print([len(t) for t in b_text_list])
            # print([len(b) for b in b_bio_text])
            # print([len(s) for s in b_spo_gold])
            # for ix, sent_id in enumerate(b_sent_ids):
            #     print('sent_id', sent_id)
            #     print(["{}: {}".format(ix, tok) for ix, tok in enumerate(train_tok[sent_id])])
            #     print(["{}: {}".format(ix, lab) for ix, lab in enumerate(train_lab[sent_id])])
            #     print(train_rel[sent_id])
            #     print('-' * 20)
            #     print(b_text_list[ix])
            #     print(b_bio_text[ix])
            #     print(b_spo_gold[ix])
            #     print()

            # ner_loss, rel_loss = model(b_toks, b_attn_mask.bool(), ner_labels=b_ner, rel_labels=b_gold_relmat)
            output = model(b_toks, b_attn_mask.bool(), b_ner, b_gold_relmat, b_text_list, b_bio_text, b_spo_gold,
                           is_train=True, reduction=args.reduction)
            ner_loss = output['crf_loss']
            rel_loss = output['selection_loss']
            loss = output['loss']

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
            train_ner_loss += ner_loss.item()
            train_rel_loss += rel_loss.item()
            pbar.set_description("L {:.6f}, L_NER: {:.6f}, L_REL: {:.6f} | epoch: {}/{}:".format(
                loss.item(), ner_loss.item(), rel_loss.item(), epoch, args.num_epoch
            ))

            if epoch > 5:
                if ((step + 1) % save_step_interval == 0) or (step == num_epoch_steps - 1):
                    dev_f1 = eval_mhs(model, dev_dataloader, dev_tok, dev_lab, dev_rel, dev_spo, bio2ix,
                                      rel2ix, cls_max_len, device, "dev dataset",
                                      ner_details=False, rel_details=False, print_general=False, verbose=0)
                    dev_f1 += (epoch,)
                    dev_f1 += (step,)
                    if best_dev_f1[0] < dev_f1[0]:
                        print(
                            " -> Previous best dev f1 {:.6f} (ner: {:.6f}, rel: {:.6f}; epoch {:d} / step {:d} \n"
                            " >> Current f1 {:.6f} (ner: {:.6f}, rel: {:.6f}; best model saved '{}'".format(
                                best_dev_f1[0],
                                best_dev_f1[1],
                                best_dev_f1[2],
                                best_dev_f1[3],
                                best_dev_f1[4],
                                dev_f1[0],
                                dev_f1[1],
                                dev_f1[2],
                                args.save_model
                            )
                        )
                        best_dev_f1 = dev_f1

                        """ save the best model """
                        if not os.path.exists(args.save_model):
                            os.makedirs(args.save_model)
                        torch.save(model.state_dict(), os.path.join(args.save_model, 'best.pt'))
                        tokenizer.save_pretrained(args.save_model)

        print('Epoch %i, train loss: %.6f, training ner_loss: %.6f, rel_loss: %.6f\n' % (
            epoch,
            train_loss / num_epoch_steps,
            train_ner_loss / num_epoch_steps,
            train_rel_loss / num_epoch_steps
        ))

        if args.epoch_eval and epoch > 0:
            eval_mhs(model, test_dataloader, test_tok, test_lab, test_rel, test_spo, bio2ix, rel2ix, cls_max_len, device,
                     "test dataset", ner_details=True, rel_details=True, print_general=True, verbose=0)

    print("""Best dev f1 {:.6f} (ner: {:.6f}, rel: {:.6f}; epoch {:d} / step {:d}\n
                 """.format(
        best_dev_f1[0],
        best_dev_f1[1],
        best_dev_f1[2],
        best_dev_f1[3],
        best_dev_f1[4],
    ))
    model.load_state_dict(torch.load(os.path.join(args.save_model, 'best.pt')))
    eval_mhs(model, test_dataloader, test_tok, test_lab, test_rel, test_spo, bio2ix, rel2ix, cls_max_len, device,
             "Final test dataset",
             ner_details=True, rel_details=True, print_general=True, verbose=0)


if __name__ == '__main__':
    main()

