#!/usr/bin/env python
# coding: utf-8
import warnings
import os
import argparse
import json
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from model import *

import clinical_eval
import utils
warnings.filterwarnings("ignore")


def eval_joint(model, eval_dataloader, eval_comments, eval_tok, eval_lab, eval_mod, eval_rel, eval_spo, ner2ix, mod2ix, rel2ix,
               cls_max_len, device, message, print_levels=(0, 0, 0),
               orig_tok=None, out_file='tmp.conll',
               f1_mode='micro', verbose=0):

    ner_evaluator = clinical_eval.TupleEvaluator()
    mod_evaluator = clinical_eval.TupleEvaluator()
    rel_evaluator = clinical_eval.TupleEvaluator()
    model.eval()
    with torch.no_grad(), open(out_file, 'w') as fo:
        for dev_step, dev_batch in enumerate(eval_dataloader):
            b_toks, b_attn_mask, b_sent_mask, b_ner, b_mod = tuple(
                t.to(device) for t in dev_batch[1:]
            )
            b_sent_ids = dev_batch[0].tolist()
            b_text_list = [utils.padding_1d(
                eval_tok[sent_id],
                cls_max_len,
                pad_tok='[PAD]') for sent_id in b_sent_ids]

            b_gold_ner = [eval_lab[sent_id] for sent_id in b_sent_ids]
            b_gold_mod = [eval_mod[sent_id] for sent_id in b_sent_ids]
            b_gold_rel = tuple([eval_spo[sent_id] for sent_id in b_sent_ids])

            if verbose:
                for sent_id in b_sent_ids:
                    print([f"{ix}: {tok}" for ix, tok in enumerate(eval_tok[sent_id])])
                    print([f"{ix}: {lab}" for ix, lab in enumerate(eval_lab[sent_id])])
                    print(eval_rel[sent_id])
                    print()

            b_pred_ner, b_pred_mod, b_pred_rel_ix = model(
                b_toks, b_attn_mask.bool(), b_sent_mask.long()
            )

            # ner tuple -> [sent_id, [ids], ner_lab]
            b_gold_ner_tuple = utils.ner2tuple(b_sent_ids, b_gold_ner)
            b_pred_ner_tuple = utils.ner2tuple(b_sent_ids, b_pred_ner)
            ner_evaluator.update(b_gold_ner_tuple, b_pred_ner_tuple)

            # mod tuple -> [sent_id, [ids], ner_lab, mod_lab]
            b_pred_mod_tuple = [p + [b_pred_mod[b_sent_ids.index(p[0])][p[1][-1]]]
                                for p in b_pred_ner_tuple if p[-1] != 'O']
            b_gold_mod_tuple = [g + [b_gold_mod[b_sent_ids.index(g[0])][g[1][-1]]]
                                for g in b_gold_ner_tuple if g[-1] != 'O']
            mod_evaluator.update(b_gold_mod_tuple, b_pred_mod_tuple)

            b_pred_rel = [[{
                'subject': [b_text_list[b_id][tok_id] for tok_id in rel['subject']],
                'predicate': rel['predicate'],
                'object': [b_text_list[b_id][tok_id] for tok_id in rel['object']],
            }
                for rel in sent_rel_ix] for b_id, sent_rel_ix in enumerate(b_pred_rel_ix) ]

            b_pred_rel_tuples = [[sent_id, rel['subject'], rel['object'], rel['predicate']]
                                 for sent_id, sent_rel in zip(b_sent_ids, b_pred_rel) for rel in sent_rel]
            b_gold_rel_tuples = [[sent_id, rel['subject'], rel['object'], rel['predicate']]
                                 for sent_id, sent_rel in zip(b_sent_ids, b_gold_rel) for rel in sent_rel]

            for sid, sbw_ner, sbw_mod, sbw_rel, index_sbw_rel in zip(b_sent_ids, b_pred_ner, b_pred_mod, b_pred_rel, b_pred_rel_ix):
                w_tok, aligned_ids = utils.sbwtok2tok_alignment(eval_tok[sid])
                w_ner = utils.sbwner2ner(sbw_ner, aligned_ids)
                w_mod = utils.sbwmod2mod(sbw_mod, aligned_ids)
                w_rel, w_head = utils.sbwrel2head(index_sbw_rel, aligned_ids)
                w_tok = w_tok[1:-1]
                w_ner = w_ner[1:-1]
                w_mod = w_mod[1:-1]
                assert len(w_tok) == len(w_ner) == len(w_mod) == len(w_rel) == len(w_head)

                if orig_tok:
                    assert len(orig_tok[sid]) == len(w_tok)

                fo.write(f'{eval_comments[sid]}\n')
                for index, (tok, ner, mod, rel, head) in enumerate(zip(orig_tok[sid] if orig_tok else w_tok, w_ner, w_mod, w_rel, w_head)):
                    fo.write(f"{index}\t{tok}\t{ner}\t{mod}\t{rel}\t{head}\n")

            rel_evaluator.update(b_gold_rel_tuples, b_pred_rel_tuples)

        ner_f1 = ner_evaluator.print_results(message + ' ner', f1_mode=f1_mode, print_level=print_levels[0])
        mod_f1 = mod_evaluator.print_results(message + ' mod', f1_mode=f1_mode, print_level=print_levels[1])
        rel_f1 = rel_evaluator.print_results(message + ' rel', f1_mode=f1_mode, print_level=print_levels[2])
        f1 = (ner_f1 + mod_f1 + rel_f1) / 3
        return f1, ner_f1, mod_f1, rel_f1


def main():

    parser = argparse.ArgumentParser(description='PRISM joint recognizer')

    # parser.add_argument("--train_file", default="data/i2b2/i2b2_training.conll", type=str,
    #                     help="train file, multihead conll format.")
    #
    # parser.add_argument("--dev_file", default="data/i2b2/i2b2_dev.conll", type=str,
    #                     help="dev file, multihead conll format.")
    #
    # parser.add_argument("--test_file", default="data/i2b2/i2b2_test.conll", type=str,
    #                     help="test file, multihead conll format.")
    #
    # parser.add_argument("--pretrained_model",
    #                     default="/home/feicheng/Tools/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12",
    #                     type=str,
    #                     help="pre-trained model dir")

    parser.add_argument("--train_file",
                        default="data/2020Q2/mr20200605_rev/cv0_train.conll",
                        type=str,
                        help="train file, multihead conll format.")

    parser.add_argument("--dev_file",
                        default="data/2020Q2/mr20200605_rev/cv0_dev.conll",
                        type=str,
                        help="dev file, multihead conll format.")

    parser.add_argument("--test_file",
                        default="data/2020Q2/mr20200605_rev/cv0_test.conll",
                        type=str,
                        help="test file, multihead conll format.")

    parser.add_argument("--pretrained_model",
                        default="/home/feicheng/Tools/NICT_BERT-base_JapaneseWikipedia_32K_BPE",
                        type=str,
                        help="pre-trained model dir")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="tokenizer: do_lower_case")

    parser.add_argument("--test_output", default='tmp/mr_rev.test.conll', type=str,
                        help="test output filename")

    parser.add_argument("--dev_output", default='tmp/mr_rev.dev.conll', type=str,
                        help="dev output filename")

    parser.add_argument("--test_dir", default="tmp/", type=str,
                        help="test dir, multihead conll format.")

    parser.add_argument("--pred_dir", default="tmp/", type=str,
                        help="prediction dir, multihead conll format.")

    parser.add_argument("--saved_model", default='checkpoints/tmp/joint_mr_rev', type=str,
                        help="save/load model dir")

    parser.add_argument("--batch_test",
                        action='store_true',
                        help="test batch files")

    parser.add_argument("--batch_size", default=8, type=int,
                        help="BATCH SIZE")

    parser.add_argument("--num_epoch", default=20, type=int,
                        help="fine-tuning epoch number")

    parser.add_argument("--embed_size", default='[64, 64, 512]', type=str,
                        help="ner, mod, rel embedding size")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--freeze_after_epoch", default=50, type=int,
                        help="freeze encoder after N epochs")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--encoder_lr", default=2e-5, type=float,
                        help="learning rate")

    parser.add_argument("--decoder_lr", default=1e-2, type=float,
                        help="learning rate")

    parser.add_argument("--other_lr", default=1e-3, type=float,
                        help="learning rate")

    parser.add_argument("--reduction", default='token_mean', type=str,
                        help="loss reduction: `token_mean` or `sum`")

    parser.add_argument("--save_best", default='f1', type=str,
                        help="save the best model, given dev scores (f1 or loss)")

    parser.add_argument("--save_step_portion", default=2, type=int,
                        help="save best model given a portion of steps")

    parser.add_argument("--neg_ratio", default=1.0, type=float,
                        help="negative sample ratio")

    parser.add_argument("--warmup_epoch", default=3, type=float,
                        help="warmup epoch")

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

    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    print(args)

    bert_max_len = 512

    bio_emb_size, mod_emb_size, rel_emb_size = eval(args.embed_size)

    if args.do_train:

        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model,
            do_lower_case=args.do_lower_case,
            do_basic_tokenize=False,
            tokenize_chinese_chars=False
        )
        tokenizer.add_tokens(['[JASP]'])

        train_comments, train_toks, train_ners, train_mods, train_rels, bio2ix, ne2ix, mod2ix, rel2ix = utils.extract_rel_data_from_mh_conll_v2(
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

        dev_comments, dev_toks, dev_ners, dev_mods, dev_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(
            args.dev_file, down_neg=0.0)
        print('max sent len:', utils.max_sents_len(dev_toks, tokenizer))
        print(min([len(sent_rels) for sent_rels in dev_rels]), max([len(sent_rels) for sent_rels in dev_rels]))
        print()

        rel_count = defaultdict(lambda: 0)

        for sent_rels in train_rels:
            for rel in sent_rels:
                rel_count[rel[-1]] += 1

        for sent_rels in dev_rels:
            for rel in sent_rels:
                rel_count[rel[-1]] += 1

        print(rel_count)

        example_id = 15
        print('Random example: id %i, len: %i' % (example_id, len(train_toks[example_id])))
        for tok_id in range(len(train_toks[example_id])):
            print("%i\t%10s\t%s" % (tok_id, train_toks[example_id][tok_id], train_ners[example_id][tok_id]))
        print(train_rels[example_id])
        print()

        max_len = max(
            utils.max_sents_len(train_toks, tokenizer),
            utils.max_sents_len(dev_toks, tokenizer),
            # utils.max_sents_len(test_toks, tokenizer)
        )
        cls_max_len = max_len + 2

        train_dataset, train_comment, train_tok, train_ner, train_mod, train_rel, train_spo = utils.convert_rels_to_mhs_v3(
            train_comments, train_toks, train_ners, train_mods, train_rels,
            tokenizer, bio2ix, mod2ix, rel2ix, cls_max_len, verbose=0)

        dev_dataset, dev_comment, dev_tok, dev_ner, dev_mod, dev_rel, dev_spo = utils.convert_rels_to_mhs_v3(
            dev_comments, dev_toks, dev_ners, dev_mods, dev_rels,
            tokenizer, bio2ix, mod2ix, rel2ix, cls_max_len, verbose=0)

        cls_max_len = min(cls_max_len, bert_max_len)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

        num_epoch_steps = len(train_dataloader)
        num_training_steps = args.num_epoch * num_epoch_steps
        save_step_interval = math.ceil(num_epoch_steps / args.save_step_portion)

        model = JointNerModReExtractor(
            bert_url=args.pretrained_model,
            ner_emb_size=bio_emb_size, ner_vocab=bio2ix,
            mod_emb_size=mod_emb_size, mod_vocab=mod2ix,
            rel_emb_size=rel_emb_size, rel_vocab=rel2ix,
            device=args.device
        )
        model.encoder.resize_token_embeddings(len(tokenizer))
        model.to(args.device)

        param_optimizer = list(model.named_parameters())
        encoder_name_list = ['encoder']
        decoder_name_list = ['crf_tagger']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in encoder_name_list)],
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in decoder_name_list)],
                'lr': args.decoder_lr
            },
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in encoder_name_list + decoder_name_list)],
                'lr': args.other_lr
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            eps=1e-8,
            correct_bias=False
        )
        if args.scheduled_lr:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_epoch_steps * args.warmup_epoch,
                num_training_steps=num_training_steps
            )

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # (F1, NER_F1, MOD_F1, REL_F1, epoch, step)
        best_dev_f1 = (float('-inf'), float('-inf'),  float('-inf'), float('-inf'), 0, 0)

        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            print(len(param_group['params']))
            print()

        for epoch in range(1, args.num_epoch + 1):

            train_loss, train_ner_loss, train_mod_loss, train_rel_loss = .0, .0, .0, .0

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", total=len(train_dataloader))
            for step, batch in enumerate(epoch_iterator):
                model.train()

                if epoch > args.freeze_after_epoch:
                    utils.freeze_bert_layers(model, bert_name='encoder', freeze_embed=True, layer_list=list(range(0, 11)))

                # input processing
                b_toks, b_attn_mask, b_sent_mask, b_ner, b_mod = tuple(
                    t.to(args.device) for t in batch[1:]
                )
                b_sent_ids = batch[0].tolist()
                b_gold_relmat = utils.gen_relmat(train_rel, b_sent_ids, cls_max_len, rel2ix, del_neg=False).to(args.device)

                b_text_list = [utils.padding_1d(
                    train_tok[sent_id],
                    cls_max_len,
                    pad_tok='[PAD]') for sent_id in b_sent_ids]

                ner_loss, mod_loss, rel_loss = model(
                    b_toks, b_attn_mask.bool(), b_sent_mask.long(),
                    ner_gold=b_ner, mod_gold=b_mod, rel_gold=b_gold_relmat, reduction=args.reduction
                )
                loss = ner_loss + mod_loss + rel_loss

                if args.n_gpu > 1:
                    loss = loss.mean()
                    ner_loss = ner_loss.mean()
                    mod_loss = mod_loss.mean()
                    rel_loss = rel_loss.mean()

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if args.scheduled_lr:
                    scheduler.step()
                model.zero_grad()

                train_loss += loss.item()
                train_ner_loss += ner_loss.item()
                train_mod_loss += mod_loss.item()
                train_rel_loss += rel_loss.item()
                epoch_iterator.set_description(
                    f"L {train_loss/(step+1):.6f}, L_NER: {train_ner_loss/(step+1):.6f}, L_MOD: {train_mod_loss/(step+1):.6f}"
                    f" L_REL: {train_rel_loss/(step+1):.6f} | epoch: {epoch}/{args.num_epoch}:"
                )

                if epoch > 5:
                    if ((step + 1) % save_step_interval == 0) or (step == num_epoch_steps - 1):
                        dev_f1 = eval_joint(model, dev_dataloader, dev_comment, dev_tok, dev_ner, dev_mod, dev_rel, dev_spo, bio2ix,
                                            mod2ix, rel2ix, cls_max_len, args.device, "dev dataset",
                                            print_levels=(0, 0, 0), out_file=args.dev_output, verbose=0)
                        dev_f1 += (epoch,)
                        dev_f1 += (step,)
                        if best_dev_f1[0] < dev_f1[0]:
                            print(
                                f" -> Previous best dev f1 {best_dev_f1[0]:.6f} (ner: {best_dev_f1[1]:.6f}, "
                                f"mod: {best_dev_f1[2]:.6f}, rel: {best_dev_f1[3]:.6f}; "
                                f"epoch {best_dev_f1[4]:d} / step {best_dev_f1[5]:d} \n "
                                f">> Current f1 {dev_f1[0]:.6f} (ner: {dev_f1[1]:.6f}, mod: {dev_f1[2]:.6f}, "
                                f"rel: {dev_f1[3]:.6f}; best model saved '{args.saved_model}'"
                            )
                            best_dev_f1 = dev_f1

                            """ save the best model """
                            if not os.path.exists(args.saved_model):
                                os.makedirs(args.saved_model)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            torch.save(model_to_save.state_dict(), os.path.join(args.saved_model, 'model.pt'))
                            tokenizer.save_pretrained(args.saved_model)
                            with open(os.path.join(args.saved_model, 'ner2ix.json'), 'w') as fp:
                                json.dump(bio2ix, fp)
                            with open(os.path.join(args.saved_model, 'mod2ix.json'), 'w') as fp:
                                json.dump(mod2ix, fp)
                            with open(os.path.join(args.saved_model, 'rel2ix.json'), 'w') as fp:
                                json.dump(rel2ix, fp)

            eval_joint(model, dev_dataloader, dev_comment, dev_tok, dev_ner, dev_mod, dev_rel, dev_spo, bio2ix,
                       mod2ix, rel2ix, cls_max_len, args.device, "dev dataset", print_levels=(1, 1, 1), verbose=0)

        print(f"Best dev f1 {best_dev_f1[0]:.6f} (ner: {best_dev_f1[1]:.6f}, mod: {best_dev_f1[2]:.6f}, "
              f"rel: {best_dev_f1[3]:.6f}; epoch {best_dev_f1[4]:d} / step {best_dev_f1[5]:d}\n")
        model.load_state_dict(torch.load(os.path.join(args.saved_model, 'model.pt')))
        torch.save(model, os.path.join(args.saved_model, 'model.pt'))
    else:
        '''Load tokenizer and tag2ix'''
        tokenizer = BertTokenizer.from_pretrained(
            args.saved_model,
            do_lower_case=args.do_lower_case,
            do_basic_tokenize=False,
            tokenize_chinese_chars=False
        )
        with open(os.path.join(args.saved_model, 'ner2ix.json')) as json_fi:
            bio2ix = json.load(json_fi)
        with open(os.path.join(args.saved_model, 'mod2ix.json')) as json_fi:
            mod2ix = json.load(json_fi)
        with open(os.path.join(args.saved_model, 'rel2ix.json')) as json_fi:
            rel2ix = json.load(json_fi)

        # '''Initialize model and load state'''
        # model = JointNerModReExtractor(
        #     bert_url=args.pretrained_model,
        #     ner_emb_size=bio_emb_size, ner_vocab=bio2ix,
        #     mod_emb_size=mod_emb_size, mod_vocab=mod2ix,
        #     rel_emb_size=rel_emb_size, rel_vocab=rel2ix,
        #     device=args.device
        # )
        # model.encoder.resize_token_embeddings(len(tokenizer))
        # model.to(args.device)
        # model.load_state_dict(torch.load(os.path.join(args.saved_model, 'best.pt')))

        '''Load full model'''
        model = torch.load(os.path.join(args.saved_model, 'model.pt'))
        model.to(args.device)

        if args.batch_test:
            for file_name in sorted(os.listdir(args.test_dir)):
                if file_name.endswith(".conll"):
                    file_in = os.path.join(args.test_dir, file_name)
                    file_out = os.path.join(args.pred_dir, file_name)

                    test_comments, test_toks, test_ners, test_mods, test_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(file_in,
                                                                                                                     down_neg=0.0)
                    print(f"max sent len: {utils.max_sents_len(test_toks, tokenizer)}")
                    print(min([len(sent_rels) for sent_rels in test_rels]), max([len(sent_rels) for sent_rels in test_rels]))
                    print()
                    max_len = utils.max_sents_len(test_toks, tokenizer)
                    cls_max_len = max_len + 2

                    test_dataset, test_comment, test_tok, test_ner, test_mod, test_rel, test_spo = utils.convert_rels_to_mhs_v3(
                        test_comments, test_toks, test_ners, test_mods, test_rels,
                        tokenizer, bio2ix, mod2ix, rel2ix, cls_max_len, verbose=0)

                    cls_max_len = min(cls_max_len, bert_max_len)

                    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

                    eval_joint(model, test_dataloader, test_comment, test_tok, test_ner, test_mod, test_rel, test_spo,
                               bio2ix, mod2ix, rel2ix, cls_max_len, args.device, "Final test dataset",
                               print_levels=(2, 2, 2), out_file=file_out, verbose=0)
        else:

            test_comments, test_toks, test_ners, test_mods, test_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(
                args.test_file,
                down_neg=0.0)
            print(f"max sent len: {utils.max_sents_len(test_toks, tokenizer)}")
            print(min([len(sent_rels) for sent_rels in test_rels]), max([len(sent_rels) for sent_rels in test_rels]))
            print()

            max_len = utils.max_sents_len(test_toks, tokenizer)
            cls_max_len = max_len + 2

            test_dataset, test_comment, test_tok, test_ner, test_mod, test_rel, test_spo = utils.convert_rels_to_mhs_v3(
                test_comments, test_toks, test_ners, test_mods, test_rels,
                tokenizer, bio2ix, mod2ix, rel2ix, cls_max_len, verbose=0)

            cls_max_len = min(cls_max_len, bert_max_len)

            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            eval_joint(model, test_dataloader, test_comment, test_tok, test_ner, test_mod, test_rel, test_spo,
                       bio2ix, mod2ix, rel2ix, cls_max_len, args.device, "Final test dataset",
                       print_levels=(2, 2, 2), out_file=args.test_output, verbose=0)

if __name__ == '__main__':
    main()

