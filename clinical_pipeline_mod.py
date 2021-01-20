#!/usr/bin/env python
# coding: utf-8
import warnings
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
from transformers import *
import argparse
from model import *
from clinical_eval import MhsEvaluator
from data_utils import bio_to_spans
warnings.filterwarnings("ignore")


def output_mod(trained_model, eval_dataloader, eval_comment, eval_tok, eval_ner, mod2ix, mod_outfile, non_bert, device):
    ix2mod = {v: k for k, v in mod2ix.items()}
    trained_model.eval()
    with torch.no_grad(), open(mod_outfile, 'w') as fo:
        for dev_step, dev_batch in enumerate(eval_dataloader):
            b_e_toks, b_e_attn_mask, b_e_sent_mask, b_e_ner, b_e_ner_mask, b_e_mod = tuple(
                t.to(device) for t in dev_batch[1:]
            )
            # print([eval_tok[sent_id] for sent_id in dev_batch[0].tolist()])
            # print()
            _, _, l = b_e_ner_mask.shape
            b_sent_ids = dev_batch[0].tolist()
            b_text_list = [utils.padding_1d(
                eval_tok[sent_id],
                cls_max_len,
                pad_tok='[PAD]') for sent_id in b_sent_ids]

            pred_logit = trained_model(b_e_toks, b_e_ner_mask.float(), attention_mask=b_e_attn_mask.bool())
            pred_tag_ix = pred_logit.argmax(-1).view(-1).cpu()
            tag_mask = torch.tensor([True if m != [0] * l else False for m in b_e_ner_mask.view(-1, l).tolist()])
            pred_tag = pred_tag_ix.masked_select(tag_mask).tolist()

            for sid in b_sent_ids:
                w_tok, aligned_ids = utils.sbwtok2tok_alignment(eval_tok[sid])
                w_ner = utils.sbwner2ner(eval_ner[sid], aligned_ids)
                if not non_bert:
                    w_tok = w_tok[1:-1]
                    w_ner = w_ner[1:-1]
                assert len(w_tok) == len(w_ner)

                sent_spans = bio_to_spans(w_ner)
                last_tid2mod = {}
                for (ner, start, end) in sent_spans:
                    last_tid2mod[end - 1] = ix2mod[pred_tag.pop(0)]

                fo.write(f'{eval_comment[sid]}\n')
                for index, (tok, ner) in enumerate(zip(w_tok, w_ner)):
                    mod = last_tid2mod[index] if index in last_tid2mod else '_'
                    fo.write(f"{index}\t{tok}\t{ner}\t{mod}\t['N']\t[{index}]\n")
            if len(pred_tag) != 0:
                print(f"pred_tag left: {len(pred_tag)}")

""" 
python input arguments 
"""
parser = argparse.ArgumentParser(description='Clinical IE pipeline Modality classifier')

# parser.add_argument("--pretrained_model",
#                     default="/home/feicheng/Tools/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12",
#                     type=str,
#                     help="pre-trained model dir")
#
# parser.add_argument("--do_lower_case",
#                     action='store_true',
#                     help="tokenizer: do_lower_case")
#
# parser.add_argument("--saved_model", default='checkpoints/tmp/pipeline/mod', type=str,
#                     help="save/load model dir")
#
# parser.add_argument("--train_file", default="data/i2b2/i2b2_training.conll", type=str,
#                     help="train file, multihead conll format.")
#
# parser.add_argument("--dev_file", default="data/i2b2/i2b2_dev.conll", type=str,
#                     help="dev file, multihead conll format.")
#
# parser.add_argument("--test_file", default="data/i2b2/i2b2_test.conll", type=str,
#                     help="test file, multihead conll format.")

parser.add_argument("--pretrained_model",
                    default="/home/feicheng/Tools/NICT_BERT-base_JapaneseWikipedia_32K_BPE",
                    type=str,
                    help="pre-trained model dir")

parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="tokenizer: do_lower_case")

parser.add_argument("--saved_model", default='checkpoints/mr20200605_rev/lstm_pipeline/ner', type=str,
                    help="save/load model dir")

parser.add_argument("--train_file", default="data/2020Q2/mr20200605_rev/sent_conll/cv0_train.conll", type=str,
                    help="train file, multihead conll format.")

parser.add_argument("--dev_file", default="data/2020Q2/mr20200605_rev/sent_conll/cv0_dev.conll", type=str,
                    help="dev file, multihead conll format.")

parser.add_argument("--test_file", default="data/2020Q2/mr20200605_rev/sent_conll/cv0_test.conll", type=str,
                    help="test file, multihead conll format.")

parser.add_argument("--batch_size", default=16, type=int,
                    help="BATCH SIZE")

parser.add_argument("--num_epoch", default=10, type=int,
                    help="fine-tuning epoch number")

parser.add_argument("--epoch_start_eval", default=3, type=int,
                    help="epoch num starting eval with validation data")

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")

parser.add_argument("--non_bert",
                    action='store_true',
                    help="use lstm + word embedding")

parser.add_argument("--word_embedding",
                    # default="/home/feicheng/Resources/Embedding/keyed-6B-300.bin.gz",
                    default="/home/feicheng/Resources/Embedding/w2v.midasi.256.100M.bin",
                    type=str,
                    help="pre-trained word embedding")

# parser.add_argument("--encoder_hidden_size", default=768, type=int,
#                     help="encoder hidden size")

parser.add_argument("--enc_lr", default=2e-5, type=float,
                    help="encoder lr")

parser.add_argument("--dec_lr", default=1e-3, type=float,
                    help="decoder layer lr")

parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")

parser.add_argument("--test_output", default='tmp/test.mod', type=str,
                    help="test output filename")

parser.add_argument("--dev_output", default='tmp/dev.mod', type=str,
                    help="dev output filename")

parser.add_argument("--later_eval",
                    action='store_true',
                    help="Whether eval model every epoch.")

parser.add_argument("--save_best", action='store', type=str, default='f1',
                    help="save the best model, given dev scores (f1 or loss)")

parser.add_argument("--save_step_interval", default=1, type=int,
                    help="save best model given a portion of steps")

parser.add_argument("--warmup_epoch", default=2, type=float,
                    help="warmup epoch")

parser.add_argument("--fp16",
                    action='store_true',
                    help="fp16")

parser.add_argument("--fp16_opt_level", type=str, default="O1",
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                    "See details at https://nvidia.github.io/apex/amp.html")

parser.add_argument("--non_scheduled_lr",
                    action='store_true',
                    help="learning rate schedule")

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
    print(bio2ix)
    print(mod2ix)
    print(rel2ix)
    print()

    dev_comments, dev_toks, dev_ners, dev_mods, dev_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(
        args.dev_file,
        down_neg=0.0
    )

    # non_bert for word embedding
    if args.non_bert:
        word2ix, weights = retrieve_w2v(args.word_embedding)
        max_len_train = max([len(sent_tok) for sent_tok in train_toks])
        max_len_dev = max([len(sent_tok) for sent_tok in dev_toks])
        _, hidden_size = weights.shape
    else:
        word2ix, weights = None, None
        max_len_train = utils.max_sents_len(train_toks, tokenizer)
        max_len_dev = utils.max_sents_len(dev_toks, tokenizer)
        hidden_size = 768
        print('max training sent len:', max_len_train)
        print('max dev sent len:', )
        print()

    max_len = max(max_len_train, max_len_dev)
    cls_max_len = max_len + 2
    print(f"max seq len: {max_len}, max seq len with [CLS] and [SEP]: {cls_max_len}")

    example_id = 25

    print(f"Random example: id {example_id}, len: {len(train_toks[example_id])}")
    for tok_id in range(len(train_toks[example_id])):
        print(f"{tok_id}\t{train_toks[example_id][tok_id]}\t{train_ners[example_id][tok_id]}")
    print(train_rels[example_id])
    print()

    """ 
    - Generate train/test tensors including (token_ids, mask_ids, label_ids) 
    - wrap them into dataloader for mini-batch cutting
    """

    train_dataset, train_comment, train_tok, train_ner, train_mod, \
    train_pair_mask, train_pair_tag, train_rel, train_rel_tup, train_spo = utils.extract_pipeline_data_from_mhs_conll(
        train_comments, train_toks, train_ners, train_mods, train_rels,
        tokenizer, bio2ix, mod2ix, rel2ix, cls_max_len,
        non_bert=args.non_bert, word2ix=word2ix, verbose=0)

    dev_dataset, dev_comment, dev_tok, dev_ner, dev_mod, \
    dev_pair_mask, dev_pair_tag, dev_rel, dev_rel_tup, dev_spo = utils.extract_pipeline_data_from_mhs_conll(
        dev_comments, dev_toks, dev_ners, dev_mods, dev_rels,
        tokenizer, bio2ix, mod2ix, rel2ix, cls_max_len,
        non_bert=args.non_bert, word2ix=word2ix, verbose=0)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    """
    Model
    """
    model = ModalityClassifier(
        args.pretrained_model, len(mod2ix),
        hidden_size=hidden_size, pretrain_embed=weights
    )

    # specify different lr
    param_optimizer = list(model.named_parameters())
    encoder_name_list = ['encoder']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in encoder_name_list)], 'lr': args.dec_lr},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in encoder_name_list)], 'lr': args.enc_lr}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        correct_bias=False
    )
    if not args.non_bert:
        model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    # PyTorch scheduler
    num_epoch_steps = len(train_dataloader)
    num_training_steps = args.num_epoch * num_epoch_steps
    save_step_interval = math.ceil(num_epoch_steps / args.save_step_interval)

    if not args.non_scheduled_lr:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_epoch_steps * args.warmup_epoch,
            num_training_steps=num_training_steps
        )

    # support fp16
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    best_dev_f1 = (float('-inf'), 0, 0)

    for epoch in range(1, args.num_epoch + 1):

        epoch_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", total=len(train_dataloader))
        for step, batch in enumerate(epoch_iterator):

            model.train()

            b_tok, b_attn_mask, b_sent_mask, b_ner, b_ner_mask, b_mod = tuple(
                t.to(args.device) for t in batch[1:]
            )

            # BERT loss, logits: (batch_size, seq_len, tag_num)
            loss = model(b_tok, b_ner_mask.float(), attention_mask=b_attn_mask.bool(), labels=b_mod)

            epoch_loss += loss.item()

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            if not args.non_scheduled_lr:
                scheduler.step()
            model.zero_grad()

            epoch_iterator.set_description(
                f"L_MOD: {epoch_loss / (step + 1):.6f} | epoch: {epoch}/{args.num_epoch}:"
            )

            if epoch >= args.epoch_start_eval:
                if ((step + 1) % save_step_interval == 0) or ((step + 1) == num_epoch_steps):
                    output_mod(model, dev_dataloader, dev_comments, dev_tok, dev_ner, mod2ix, args.dev_output, args.non_bert, args.device)
                    dev_evaluator = MhsEvaluator(args.dev_file, args.dev_output)
                    dev_f1 = (dev_evaluator.eval_mod(print_level=0), epoch, step)
                    if best_dev_f1[0] < dev_f1[0]:
                        print(
                            f" -> Previous best dev f1 {best_dev_f1[0]:.6f}; "
                            f"epoch {best_dev_f1[1]:d} / step {best_dev_f1[2]:d} \n "
                            f">> Current f1 {dev_f1[0]:.6f}; best model saved '{args.saved_model}'"
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
    print(f"Best dev f1 {best_dev_f1[0]:.6f}; epoch {best_dev_f1[1]:d} / step {best_dev_f1[2]:d}\n")
    model.load_state_dict(torch.load(os.path.join(args.saved_model, 'model.pt')))
    torch.save(model, os.path.join(args.saved_model, 'model.pt'))
else:
    """ load the new tokenizer"""
    print("test_mode:", args.saved_model)
    tokenizer = BertTokenizer.from_pretrained(
        args.saved_model,
        do_lower_case=args.do_lower_case,
        do_basic_tokenize=False
    )
    with open(os.path.join(args.saved_model, 'ner2ix.json')) as json_fi:
        bio2ix = json.load(json_fi)
    with open(os.path.join(args.saved_model, 'mod2ix.json')) as json_fi:
        mod2ix = json.load(json_fi)
    with open(os.path.join(args.saved_model, 'rel2ix.json')) as json_fi:
        rel2ix = json.load(json_fi)

    """ load test data """
    test_comments, test_toks, test_ners, test_mods, test_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(
        args.test_file,
        down_neg=0.0)
    if args.non_bert:
        word2ix, weights = retrieve_w2v(args.word_embedding)
        max_len = max([len(sent_tok) for sent_tok in test_toks])
    else:
        word2ix, weights = None, None
        max_len = utils.max_sents_len(test_toks, tokenizer)
    print(f"max sent len: {max_len}")
    print(min([len(sent_rels) for sent_rels in test_rels]), max([len(sent_rels) for sent_rels in test_rels]))
    print()

    cls_max_len = max_len + 2

    test_dataset, test_comment, test_tok, test_ner, test_mod, \
    test_pair_mask, test_pair_tag, test_rel, test_rel_tup, test_spo = utils.extract_pipeline_data_from_mhs_conll(
        test_comments, test_toks, test_ners, test_mods, test_rels,
        tokenizer, bio2ix, mod2ix, rel2ix, cls_max_len,
        non_bert=args.non_bert, word2ix=word2ix, verbose=0)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    """ load the saved model"""
    model = torch.load(os.path.join(args.saved_model, 'model.pt'))
    model.to(args.device)

    """ predict test out """
    output_mod(model, test_dataloader, test_comments, test_tok, test_ner, mod2ix, args.test_output, args.non_bert, args.device)
    test_evaluator = MhsEvaluator(args.test_file, args.test_output)
    test_evaluator.eval_mod(print_level=2)
