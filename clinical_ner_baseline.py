#!/usr/bin/env python
# coding: utf-8
import warnings
from tqdm import tqdm
from utils import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import *
import argparse
import gensim

from model import *

warnings.filterwarnings("ignore")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

print('device', device)

juman = Juman()

torch.cuda.manual_seed_all(1234)

""" 
python input arguments 
"""

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

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")

parser.add_argument("-o", "--output", dest="OUTPUT_FILE", default='outputs/temp.ner', type=str,
                    help="output filename")

args = parser.parse_args()


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

whole_toks = train_toks + test_toks
max_len = max([len(x) for x in whole_toks])
unk_count = sum([x.count('[UNK]') for x in whole_toks])
total_count = sum([len(x) for x in whole_toks])

lab2ix = get_label2ix(train_labs + test_labs)
cert_lab2ix = get_label2ix(train_cert_labs + test_cert_labs)
ttype_lab2ix = get_label2ix(train_ttype_labs + test_ttype_labs)
state_lab2ix = get_label2ix(train_state_labs + test_state_labs)

word2ix, weights = retrieve_w2v("/home/feicheng/Resources/Embedding/w2v.midasi.256.100M.bin")
vocab_size, embed_dim = weights.shape

# tok_list = [item for sublist in (train_deunks + test_deunks) for item in sublist]
# word2ix = get_label2ix(tok_list, default={'[UNK]': 0})

ix2lab = {v: k for k, v in lab2ix.items()}
""" load the new tokenizer"""
BATCH_SIZE = args.BATCH_SIZE
test_tensors, test_deunk = extract_ner_from_conll_w2v(TEST_FILE, word2ix, lab2ix, device)
test_dataloader = DataLoader(test_tensors, batch_size=BATCH_SIZE, shuffle=False)
test_deunk_loader = [test_deunk[i: i + BATCH_SIZE] for i in range(0, len(test_deunk), BATCH_SIZE)]
print('test size: %i' % len(test_tensors))


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
    train_tensors, train_deunk = extract_ner_from_conll_w2v(TRAIN_FILE, word2ix, lab2ix, device)
    train_dataloader = DataLoader(train_tensors, batch_size=args.BATCH_SIZE, shuffle=True)
    print('train size: %i' % len(train_tensors))

    model_dir = ""

    """ Disease Tags recognition """
    model = LSTMCRF(embed_dim, embed_dim, len(word2ix), len(lab2ix), pretrain_embed=weights)
    model.to(device)

    num_epoch_steps = len(train_dataloader)
    num_training_steps = args.NUM_EPOCHS * num_epoch_steps
    max_grad_norm = 1.0

    # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW(
        model.parameters(),
        correct_bias=False
    )

    pulse_count = 3

    for epoch in range(1, args.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = .0
        for batch_feat, batch_mask, batch_lab in tqdm(train_dataloader, desc='Training'):

            model.zero_grad()

            batch_size, max_len = batch_feat.shape

            loss = model(batch_feat, batch_mask, batch_lab)

            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        print("Epoch loss: %.6f" % (epoch_loss/(len(train_dataloader)*args.BATCH_SIZE)))

        """ predict test out """
        # output_file = 'outputs/ner_%s_ep%i' % (args.CORPUS, args.NUM_EPOCHS)
        model.eval()
        with torch.no_grad():
            EVAL_FILE = args.OUTPUT_FILE + '.eval.conll'
            with open(EVAL_FILE, 'w') as fo:
                for batch_deunk, (batch_tok_ix, batch_mask, batch_gold) in zip(test_deunk_loader, test_dataloader):
                    pred_ix = model(batch_tok_ix, batch_mask)
                    gold_masked_ix = batch_demask(batch_gold, batch_mask.bool())
                    for sent_deunk, sent_mask, sent_gold_ix, sent_pred_ix in zip(batch_deunk, batch_mask,
                                                                                 gold_masked_ix, pred_ix):
                        # print(len(sent_deunk), len(sent_gold_ix), len(sent_pred_ix))
                        assert len(sent_deunk) == len(sent_gold_ix) == len(sent_pred_ix)
                    tok_masked_ix = batch_demask(batch_tok_ix, batch_mask.bool())

                    for sent_deunk, sent_gold_ix, sent_pred_ix in zip(batch_deunk, gold_masked_ix, pred_ix):
                        for tok_deunk, tok_gold, tok_pred in zip(sent_deunk, sent_gold_ix, sent_pred_ix):
                            fo.write('%s\t%s\t%s\n' % (tok_deunk, ix2lab[tok_gold], ix2lab[tok_pred]))
                        fo.write('\n')
            import subprocess
            print(subprocess.check_output(
                ['./eval_ner.sh', 'outputs/pred_goku_1225.eval.conll']
            ).decode("utf-8"))
else:
    model_dir = args.MODEL_DIR






