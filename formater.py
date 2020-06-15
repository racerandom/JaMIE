import os
import re
import random

quoted = re.compile('"[^"]*"')


class MultiheadConllConvertor(object):

    def __init__(self):
        self.tok_3d = []   # (doc x sent x token)
        self.ner_3d = []   # (doc x sent x ner)
        self.ast_3d = []
        self.head_3d = []
        self.rel_3d = []
        self.doc_info = []

    def filter_by_length(self, len_thres):
        filted_history = []
        for doc_id in range(len(self.tok_3d)):
            for s_id in range(len(self.tok_3d[doc_id])):
                if len(self.tok_3d[doc_id][s_id]) >= len_thres:
                    filted_history.append((doc_id, s_id))
        filted_history.reverse()
        for f_d_id, f_s_id in filted_history:
            del self.tok_3d[f_d_id][f_s_id]
            del self.ner_3d[f_d_id][f_s_id]
            del self.ast_3d[f_d_id][f_s_id]
            del self.head_3d[f_d_id][f_s_id]
            del self.rel_3d[f_d_id][f_s_id]

    def filter_by_empty(self):
        filted_history = []
        for doc_id in range(len(self.ner_3d)):
            for s_id in range(len(self.ner_3d[doc_id])):
                if set(self.ner_3d[doc_id][s_id]) == {'O'}:
                    filted_history.append((doc_id, s_id))
        filted_history.reverse()
        for f_d_id, f_s_id in filted_history:
            del self.tok_3d[f_d_id][f_s_id]
            del self.ner_3d[f_d_id][f_s_id]
            del self.ast_3d[f_d_id][f_s_id]
            del self.head_3d[f_d_id][f_s_id]
            del self.rel_3d[f_d_id][f_s_id]

    def output_conll(self, out_file):
        with open(out_file, 'w', encoding='utf8') as fo:
            for tok_2d, ner_2d, ast_2d, head_2d, rel_2d, docinfo in zip(
                    self.tok_3d,
                    self.ner_3d,
                    self.ast_3d,
                    self.head_3d,
                    self.rel_3d,
                    self.doc_info
            ):
                for sent_id, (tok_1d, ner_1, ast_1d, head_1d, rel_1d) in enumerate(zip(tok_2d, ner_2d, ast_2d, head_2d, rel_2d)):
                    fo.write("#doc_{}_sent_{}\n".format(docinfo, sent_id))
                    for tok_id, (tok, ner, ast, head, rel) in enumerate(zip(tok_1d, ner_1, ast_1d, head_1d, rel_1d)):
                        fo.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(tok_id, tok, ner, ast, rel, head))

    def split_train_dev(self, train_file, dev_file, dev_ratio=0.1):
        with open(train_file, 'w', encoding='utf8') as tfo, open(dev_file, 'w', encoding='utf8') as dfo:
            for tok_2d, ner_2d, ast_2d, head_2d, rel_2d, docinfo in zip(
                    self.tok_3d,
                    self.ner_3d,
                    self.ast_3d,
                    self.head_3d,
                    self.rel_3d,
                    self.doc_info
            ):
                out_writter = tfo if random.random() > dev_ratio else dfo
                for sent_id, (tok_1d, ner_1, ast_1d, head_1d, rel_1d) in enumerate(zip(tok_2d, ner_2d, ast_2d, head_2d, rel_2d)):
                    out_writter.write("#doc_{}_sent_{}\n".format(docinfo, sent_id))
                    for tok_id, (tok, ner, ast, head, rel) in enumerate(zip(tok_1d, ner_1, ast_1d, head_1d, rel_1d)):
                        out_writter.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(tok_id, tok, ner, ast, rel, head))

    @staticmethod
    def read_con(con_line):
        items = con_line.split()
        cg = ' '.join(items[:-2]).lstrip("c=")

        sent_id, tid_begin = [int(s) for s in items[-2].split(':')]
        sent_id, tid_end = [int(s) for s in items[-1].split(':')]

        return cg[1:-1], sent_id - 1, tid_begin, tid_end

    def load_batch_from_i2b2(self, data_dir, disease):
        txt_dir = os.listdir(os.path.join(data_dir, disease, 'txt'))
        for file_name in sorted(txt_dir):
            if file_name.endswith(".txt"):
                self.load_single_from_i2b2(data_dir, disease, os.path.splitext(file_name)[0])

    def load_single_from_i2b2(self, data_dir, disease, file_name):

        in_txt = os.path.join(data_dir, disease, "txt", "{}.txt".format(file_name))
        in_con = os.path.join(data_dir, disease, "concept", "{}.con".format(file_name))
        in_ast = os.path.join(data_dir, disease, "ast", "{}.ast".format(file_name))
        in_rel = os.path.join(data_dir, disease, "rel", "{}.rel".format(file_name))

        tok_2d, ner_2d, ast_2d, head_2d, rel_2d = [], [], [], [], []
        with open(in_txt, 'r') as fi:
            for line in fi:
                toks = line.rstrip().split()
                tok_2d.append(toks)
                ner_2d.append(['O'] * len(toks))
                ast_2d.append(['_'] * len(toks))
                head_2d.append([[i] for i in list(range(len(toks)))])
                rel_2d.append([['N']] * len(toks))

        with open(in_con, 'r') as fi:
            for line in fi:
                try:
                    tl, cl = line.rstrip().split('||')
                    tg, sent_id, tid_begin, tid_end = self.read_con(tl)
                    tp = ' '.join(tok_2d[sent_id][tid_begin: tid_end + 1])

                    assert tg == tp.lower()
                    con = quoted.findall(cl)[0].strip('"')
                    ner_2d[sent_id][tid_begin] = "B-{}".format(con)
                    if tid_end > tid_begin:
                        for i in range(tid_begin + 1, tid_end + 1):
                            ner_2d[sent_id][i] = "I-{}".format(con)
                except AssertionError as ex:
                    print('[ner]')
                    print(disease, file_name)
                    print(line)
                    print(tp, '||', tg)
                    print()

        with open(in_ast, 'r') as fi:
            for line in fi:
                try:
                    tl, cl, al = line.rstrip().split('||')
                    tg, sent_id, tid_begin, tid_end = self.read_con(tl)
                    tp = ' '.join(tok_2d[sent_id][tid_begin: tid_end + 1])
                    assert tg == tp.lower()
                    ast = quoted.findall(al)[0].strip('"')
                    ast_2d[sent_id][tid_end] = ast
                except AssertionError as ex:
                    print('[ast]')
                    print(disease, file_name)
                    print(line)
                    print(tp, '||', tg)
                    print()

        with open(in_rel, 'r') as fi:
            for line in fi:
                try:
                    tl, rl, hl = line.rstrip().split('||')
                    tg, t_sent_id, t_tid_begin, t_tid_end = self.read_con(tl)
                    tp = ' '.join(tok_2d[t_sent_id][t_tid_begin: t_tid_end + 1])
                    assert tg == tp.lower()
                    hg, h_sent_id, h_tid_begin, h_tid_end = self.read_con(hl)
                    hp = ' '.join(tok_2d[h_sent_id][h_tid_begin: h_tid_end + 1])
                    assert hg == hp.lower()
                    rel = quoted.findall(rl)[0].strip('"')
                    if (head_2d[t_sent_id][t_tid_end] == [t_tid_end]) or (rel_2d[t_sent_id][t_tid_end] == ['N']):
                        head_2d[t_sent_id][t_tid_end] = [h_tid_end]
                        rel_2d[t_sent_id][t_tid_end] = [rel]
                    else:
                        head_2d[t_sent_id][t_tid_end].append(h_tid_end)
                        rel_2d[t_sent_id][t_tid_end].append(rel)
                except AssertionError as ex:
                    print('[rel]')
                    print(disease, file_name)
                    print(line)
                    print(tp, '||', tg)
                    print(hp, '||', hg)
                    print()

        self.tok_3d.append(tok_2d)
        self.ner_3d.append(ner_2d)
        self.ast_3d.append(ast_2d)
        self.head_3d.append(head_2d)
        self.rel_3d.append(rel_2d)
        self.doc_info.append("{}-{}".format(disease, file_name))


mhsc_training = MultiheadConllConvertor()
# mhsc.load_single_from_i2b2("/home/feicheng/Resources/concept_assertion_relation_training_data/", "beth", "record-13")
mhsc_training.load_batch_from_i2b2("/home/feicheng/Resources/i2b2va_2010/training_data", "partners")
mhsc_training.load_batch_from_i2b2("/home/feicheng/Resources/i2b2va_2010/training_data", "beth")
print(len(mhsc_training.doc_info))

mhsc_test = MultiheadConllConvertor()
mhsc_test.load_batch_from_i2b2("/home/feicheng/Resources/i2b2va_2010/test_data", "test")

len_thres = 100
mhsc_training.filter_by_length(len_thres)
# mhsc_training.filter_by_empty()
mhsc_test.filter_by_length(len_thres)

mhsc_training.split_train_dev("data/i2b2/i2b2_training.conll", "data/i2b2/i2b2_dev.conll", dev_ratio=0.05)
mhsc_test.output_conll("data/i2b2/i2b2_test.conll")

for d in mhsc_training.tok_3d:
    for s in d:
        if len(s) > len_thres:
            print(' '.join(s))
