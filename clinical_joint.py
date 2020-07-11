#!/usr/bin/env python
# coding: utf-8
import warnings
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from model import *

import utils
warnings.filterwarnings("ignore")


def eval_joint(model, eval_dataloader, eval_comments, eval_tok, eval_lab, eval_mod, eval_rel, eval_spo, ner2ix, mod2ix, rel2ix,
               cls_max_len, device, message, print_details=(False, False, False, False),
               orig_tok=None, out_file='tmp.conll',
               f1_mode='micro', verbose=0):

    ner_evaluator = utils.TupleEvaluator()
    mod_evaluator = utils.TupleEvaluator()
    rel_evaluator = utils.TupleEvaluator()
    model.eval()
    with torch.no_grad(), open(out_file, 'w') as fo:
        for dev_step, dev_batch in enumerate(eval_dataloader):
            b_toks, b_attn_mask, b_ner, b_mod = tuple(
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
                b_toks, b_attn_mask.bool()
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
                    fo.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        index, tok, ner, mod, rel, head
                    ))

            rel_evaluator.update(b_gold_rel_tuples, b_pred_rel_tuples)

        ner_f1 = ner_evaluator.print_results(message + ' ner', print_details=print_details[1], print_general=print_details[0],
                                             f1_mode=f1_mode)
        mod_f1 = mod_evaluator.print_results(message + ' mod', print_details=print_details[2], print_general=print_details[0],
                                             f1_mode=f1_mode)
        rel_f1 = rel_evaluator.print_results(message + ' rel', print_details=print_details[3], print_general=print_details[0],
                                             f1_mode=f1_mode)
        f1 = (ner_f1 + mod_f1 + rel_f1) / 3
        return f1, ner_f1, mod_f1, rel_f1


def main():

    parser = argparse.ArgumentParser(description='PRISM joint recognizer')

    parser.add_argument("--train_file", default="data/clinical20200605/cv5/cv0_train_juman.conll", type=str,
                        help="train file, multihead conll format.")

    parser.add_argument("--dev_file", default="data/clinical20200605/cv5/cv0_dev_juman.conll", type=str,
                        help="dev file, multihead conll format.")

    parser.add_argument("--test_file", default="data/clinical20200605/cv5/cv0_test_juman.conll", type=str,
                        help="test file, multihead conll format.")

    parser.add_argument("--pred_file", default="mr_cv0_test_pred.conll", type=str,
                        help="test prediction, multihead conll format.")

    parser.add_argument("--test_dir", default="data/clinicalreport_part2/conll", type=str,
                        help="test dir, multihead conll format.")

    parser.add_argument("--pred_dir", default="data/clinicalreport_part2/pred/conll", type=str,
                        help="prediction dir, multihead conll format.")

    parser.add_argument("--pretrained_model",
                        default="/home/feicheng/Tools/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers",
                        type=str,
                        help="pre-trained model dir")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="tokenizer: do_lower_case")

    parser.add_argument("--batch_test",
                        action='store_true',
                        help="test batch files")

    parser.add_argument("--save_model", default='checkpoints/tmp/', type=str,
                        help="save/load model dir")

    parser.add_argument("--batch_size", default=8, type=int,
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

    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="warmup ratio")

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

    bio_emb_size, mod_emb_size, rel_emb_size = eval(args.embed_size)

    if args.do_train:
        tokenizer = BertTokenizer.from_pretrained(
            args.pretrained_model,
            do_lower_case=args.do_lower_case,
            do_basic_tokenize=False,
            tokenize_chinese_chars=False
        )
        tokenizer.add_tokens(['[JASP]'])
    else:
        tokenizer = BertTokenizer.from_pretrained(
            args.save_model,
            do_lower_case=args.do_lower_case,
            do_basic_tokenize=False,
            tokenize_chinese_chars=False
        )

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

    dev_comments, dev_toks, dev_ners, dev_mods, dev_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(args.dev_file, down_neg=0.0)
    print('max sent len:', utils.max_sents_len(dev_toks, tokenizer))
    print(min([len(sent_rels) for sent_rels in dev_rels]), max([len(sent_rels) for sent_rels in dev_rels]))
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
        crf_name_list = ['crf_tagger', 'mod_h2o']
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
                num_warmup_steps=num_training_steps * args.warmup_ratio,
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
                b_toks, b_attn_mask, b_ner, b_mod = tuple(
                    t.to(args.device) for t in batch[1:]
                )
                b_sent_ids = batch[0].tolist()
                b_gold_relmat = utils.gen_relmat(train_rel, b_sent_ids, cls_max_len, rel2ix, del_neg=False).to(args.device)

                b_text_list = [utils.padding_1d(
                    train_tok[sent_id],
                    cls_max_len,
                    pad_tok='[PAD]') for sent_id in b_sent_ids]

                ner_loss, mod_loss, rel_loss = model(
                    b_toks, b_attn_mask.bool(),
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
                        dev_f1 = eval_joint(model, dev_dataloader, dev_comments, dev_tok, dev_ner, dev_mod, dev_rel, dev_spo, bio2ix,
                                            mod2ix, rel2ix, cls_max_len, args.device, "dev dataset", verbose=0)
                        dev_f1 += (epoch,)
                        dev_f1 += (step,)
                        if best_dev_f1[0] < dev_f1[0]:
                            print(
                                f" -> Previous best dev f1 {best_dev_f1[0]:.6f} (ner: {best_dev_f1[1]:.6f}, "
                                f"mod: {best_dev_f1[2]:.6f}, rel: {best_dev_f1[3]:.6f}; "
                                f"epoch {best_dev_f1[4]:d} / step {best_dev_f1[5]:d} \n "
                                f">> Current f1 {dev_f1[0]:.6f} (ner: {dev_f1[1]:.6f}, mod: {dev_f1[2]:.6f}, "
                                f"rel: {dev_f1[3]:.6f}; best model saved '{args.save_model}'"
                            )
                            best_dev_f1 = dev_f1

                            """ save the best model """
                            if not os.path.exists(args.save_model):
                                os.makedirs(args.save_model)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            torch.save(model_to_save.state_dict(), os.path.join(args.save_model, 'best.pt'))
                            tokenizer.save_pretrained(args.save_model)

            eval_joint(model, dev_dataloader, dev_comments, dev_tok, dev_ner, dev_mod, dev_rel, dev_spo, bio2ix,
                       mod2ix, rel2ix, cls_max_len, args.device, "dev dataset", orig_tok=dev_toks,
                       print_details=(True, True, True, True), verbose=0)

            print('Epoch %i, train loss: %.6f, training ner_loss: %.6f, training mod_loss: %.6f, rel_loss: %.6f\n' % (
                epoch,
                train_loss / num_epoch_steps,
                train_ner_loss / num_epoch_steps,
                train_mod_loss / num_epoch_steps,
                train_rel_loss / num_epoch_steps
            ))

        print(f"Best dev f1 {best_dev_f1[0]:.6f} (ner: {best_dev_f1[1]:.6f}, mod: {best_dev_f1[2]:.6f}, "
              f"rel: {best_dev_f1[3]:.6f}; epoch {best_dev_f1[4]:d} / step {best_dev_f1[5]:d}\n")
    else:
        model = JointNerModReExtractor(
            bert_url=args.pretrained_model,
            ner_emb_size=bio_emb_size, ner_vocab=bio2ix,
            mod_emb_size=mod_emb_size, mod_vocab=mod2ix,
            rel_emb_size=rel_emb_size, rel_vocab=rel2ix,
            device=args.device
        )
        model.encoder.resize_token_embeddings(len(tokenizer))
        model.to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.save_model, 'best.pt')))

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
                    max_len = max(
                        max_len,
                        utils.max_sents_len(test_toks, tokenizer)
                    )

                    test_dataset, test_tok, test_ner, test_mod, test_rel, test_spo = utils.convert_rels_to_mhs_v3(
                        test_toks, test_ners, test_mods, test_rels,
                        tokenizer, bio2ix, mod2ix, rel2ix, max_len, verbose=0)

                    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

                    eval_joint(model, test_dataloader, test_comments, test_tok, test_ner, test_mod, test_rel, test_spo,
                               bio2ix, mod2ix, rel2ix, cls_max_len, args.device, "Final test dataset",
                               print_details=(True, True, True, True),
                               orig_tok=test_toks, out_file=file_out, verbose=0)
        else:

            test_comments, test_toks, test_ners, test_mods, test_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(
                args.test_file,
                down_neg=0.0)
            print(f"max sent len: {utils.max_sents_len(test_toks, tokenizer)}")
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

            eval_joint(model, test_dataloader, test_comments, test_tok, test_ner, test_mod, test_rel, test_spo,
                       bio2ix, mod2ix, rel2ix, cls_max_len, args.device, "Final test dataset",
                       print_details=(True, True, True, True), out_file=args.pred_file, verbose=0)


if __name__ == '__main__':
    main()

