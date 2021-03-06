#!/usr/bin/env python
# coding: utf-8
import copy
import numpy as np

MOD_DICT = {
        'positive': 'certainty', 'suspicious': 'certainty', 'negative': 'certainty', 'general': 'certainty',
        'executed': 'state', 'negated': 'state', 'scheduled': 'state', 'other': 'state',
        'DATE': 'type',  'TIME': 'type', 'DURATION': 'type', 'CC': 'type', 'SET': 'type', 'AGE': 'type', 'MISC': 'type'
}
tag2name = {
    'd': 'Disease',
    'a': 'Anatomical',
    'f': 'Feature',
    'c': 'Change',
    'p': 'Pending',
    'TIMEX3': 'TIMEX3',
    't-test': 'TestTest',
    't-key': 'TestKey',
    't-val': 'TestVal',
    'cc': 'ClinicalContext',
    'r': 'Remedy',
    'm-key': 'MedicineKey',
    'm-val': 'MedicineVal',
}

NER_DICT = {
    'D': 'Disease',
    'A': 'Anatomical',
    'F': 'Feature',
    'C': 'Change',
    'Timex3': 'TIMEX3',
    'T-test': 'TestTest',
    'T-key': 'TestKey',
    'T-val': 'TestVal',
    'M-key': 'MedicineKey',
    'M-val': 'MedicineVal',
    'R': 'Remedy',
    'Cc': 'ClinicalContext',
    'P': 'Pending'
}


def bio_to_spans(ner_tags):
    entities = []
    start = None
    for i, tag in enumerate(ner_tags):
        if i == 0 and tag != 'O':
            start = i
        else:
            if tag.startswith('O'):
                if start is not None:
                    entities.append((ner_tags[i - 1][2:], start, i))
                    start = None
            elif tag.startswith('I'):
                if start is not None:
                    if tag[2:] != ner_tags[i - 1][2:]:
                        entities.append((ner_tags[i - 1][2:], start, i))
                        start = i
                else:
                    start = i
            elif tag.startswith('B'):
                if start is not None:
                    entities.append((ner_tags[i - 1][2:], start, i))
                start = i
            else:
                raise ValueError("[ERROR] tag: ", tag)
    if start is not None:
        entities.append((ner_tags[-1][2:], start, len(ner_tags)))
    return entities


# MultiheadConll Object
class MultiheadConllObject(object):

    def __init__(self, conll_file, delimiter=('#doc', '## line')):
        self._doc_lines = []
        self._comments = []
        self._tok_ids = []
        self._toks = []
        self._ner_tags = []
        self._mod_tags = []
        self._head_rels = []
        self._head_ids = []
        self._multihead_mat = []
        self.load_lines(conll_file, delimiter)
        self.load_columns()

    ''' split conll lines with the delimiters of the commented lines 
        self._doc_lines as a 2D list 
    '''
    def load_lines(self, conll_file, delimiter):
        with open(conll_file, 'r', encoding='utf8') as conll_fi:
            sent_lines = []
            for line in conll_fi:
                if line.startswith(delimiter):
                    self._comments.append(line.strip())
                    if sent_lines:
                        self._doc_lines.append(sent_lines)
                        sent_lines = []
                    continue
                sent_lines.append(line)
            if sent_lines:
                self._doc_lines.append(sent_lines)

    def load_columns(self):
        for sent_lines in self._doc_lines:
            s_tok_ids, s_toks, s_ner_tags, s_mod_tags, s_head_rels, s_head_ids = [], [], [], [], [], []
            for line in sent_lines:
                tok_items = line.strip().split('\t')
                s_tok_ids.append(int(tok_items[0]))
                s_toks.append(tok_items[1])
                s_ner_tags.append(tok_items[2])
                s_mod_tags.append(tok_items[3])
                s_head_rels.append(eval(tok_items[4]))
                s_head_ids.append(eval(tok_items[5]))
            self._tok_ids.append(s_tok_ids)
            self._toks.append(s_toks)
            self._ner_tags.append(s_ner_tags)
            self._mod_tags.append(s_mod_tags)
            self._head_rels.append(s_head_rels)
            self._head_ids.append(s_head_ids)

    def generate_multihead_mat(self):
        for ner_tags, mod_tags, head_rels, head_ids in zip(self._ner_tags, self._mod_tags, self._head_rels, self._head_ids):
            seq_len = len(ner_tags)
            mh_mat = np.empty((seq_len, seq_len), 'U25')
            mh_mat.fill("O")
            mh_mat = self.ner_to_mat(ner_tags, mh_mat)
            mh_mat = self.rel_to_mat(head_rels, head_ids, mh_mat)
            print(ner_tags)
            print(mh_mat)
            print()
            self._multihead_mat.append(mh_mat)

    @staticmethod
    def ner_to_mat(ner_tags, mh_mat):
        prev_start = None
        for tag_id, tag in enumerate(ner_tags):
            ne_tag = tag.split('-', 1)[-1]
            if tag_id == 0:
                prev_start = (tag_id, ne_tag)
            else:
                if tag == 'O' or tag.startswith('B-'):
                    mh_mat[prev_start[0], tag_id-1] = prev_start[1]
                    prev_start = (tag_id, ne_tag)
                elif tag.startswith('I-'):
                    if tag.split('-')[-1] != prev_start[1]:
                        mh_mat[prev_start[0], tag_id - 1] = prev_start[1]
                        prev_start = (tag_id, ne_tag)
                else:
                    raise Exception(f"Incorrect tag: {tag}...")
        mh_mat[prev_start[0]][-1] = prev_start[1]
        return mh_mat

    @staticmethod
    def rel_to_mat(head_rels, head_ids, mh_mat):
        for modifier_id, (tok_head_rels, tok_head_ids) in enumerate(zip(head_rels, head_ids)):
            for head_rel, head_id in zip(tok_head_rels, tok_head_ids):
                if modifier_id != head_id and head_rel != 'N':
                    mh_mat[modifier_id, head_id] = head_rel
        return mh_mat


# MultiheadConll document class
class MultiheadConll(object):

    def __init__(self, conll_file, prune_type=False, delimiter=('#doc', '## line')):
        self._doc_lines = []
        self._comments = []
        self._tok_ids = []
        self._toks = []
        self._ner_tags = []
        self._mod_tags = []
        self._rel_tag_lists = []
        self._head_id_lists = []
        self._entities = []
        self._mod_entities = []
        self._rel_triplets = []
        self._rel_detailed_triplets = []
        self._rel_mention_triplets = []
        self.load_doc(conll_file, delimiter)
        self.update_columns()
        self.update_entities()
        self.update_mod_entities()
        self.update_rel_triplets()
        self.update_rel_detailed_triplets()
        self.update_rel_mention_triplets()

    def load_doc(self, conll_file, empty_comment):
        with open(conll_file, 'r', encoding='utf8') as conll_fi:
            sent_lines = []
            for line in conll_fi:
                if line.startswith(empty_comment):
                    self._comments.append(line.strip())
                    if sent_lines:
                        self._doc_lines.append(sent_lines)
                        sent_lines = []
                    continue
                sent_lines.append(line)
            if sent_lines:
                self._doc_lines.append(sent_lines)

    def update_columns(self):
        for sent_lines in self._doc_lines:
            s_tok_ids, s_toks, s_ner_tags, s_mod_tags, s_rels, s_heads = [], [], [], [], [], []
            for line in sent_lines:
                tok_items = line.strip().split('\t')
                s_tok_ids.append(int(tok_items[0]))
                s_toks.append(tok_items[1].replace('[JASP]', '\u3000').replace('[SEP]', '\n'))
                s_ner_tags.append(tok_items[2])
                s_mod_tags.append(tok_items[3])
                s_rels.append(eval(tok_items[4]))
                s_heads.append(eval(tok_items[5]))
            self._tok_ids.append(s_tok_ids)
            self._toks.append(s_toks)
            self._ner_tags.append(s_ner_tags)
            self._mod_tags.append(s_mod_tags)
            self._rel_tag_lists.append(s_rels)
            self._head_id_lists.append(s_heads)

    def update_entities(self):
        for sent_id in range(len(self._doc_lines)):
            self._entities.append(bio_to_spans(self._ner_tags[sent_id]))

    def update_mod_entities(self):
        for sent_id in range(len(self._doc_lines)):
            sent_entities = [e + (self._mod_tags[sent_id][e[-1] - 1],) for e in bio_to_spans(self._ner_tags[sent_id])]
            self._mod_entities.append(sent_entities)

    def update_rel_triplets(self):
        for sent_id in range(len(self._doc_lines)):
            sent_triplets = []
            for tail_id, head_ids, rels in zip(self._tok_ids[sent_id], self._head_id_lists[sent_id], self._rel_tag_lists[sent_id]):
                for head_id, rel in zip(head_ids, rels):
                    if rel not in ['N']:
                        sent_triplets.append((tail_id, head_id, rel))
            self._rel_triplets.append(sent_triplets)

    def update_rel_detailed_triplets(self):
        for sent_id in range(len(self._doc_lines)):
            sent_dic = {(entity[-1]-1): entity[1:3] for entity in self._entities[sent_id]}
            sent_triplets = []
            for tail_id, head_id, rel in self._rel_triplets[sent_id]:
                if rel not in ['N']:
                    tail_span = sent_dic[tail_id] if tail_id in sent_dic else (tail_id, tail_id + 1)
                    head_span = sent_dic[head_id] if head_id in sent_dic else (head_id, head_id + 1)
                    sent_triplets.append((tail_span, head_span, rel))
            self._rel_detailed_triplets.append(sent_triplets)

    def update_rel_mention_triplets(self):
        for sent_id in range(len(self._doc_lines)):
            sent_dic = {(entity[-1]-1): entity[1:3] for entity in self._entities[sent_id]}
            sent_triplets = []
            for tail_id, head_id, rel in self._rel_triplets[sent_id]:
                if rel not in ['N']:
                    # tail_mention = ''.join(self._toks[sent_id][sent_dic[tail_id][0]: sent_dic[tail_id][1]]) if tail_id in sent_dic else ''.join(self._toks[sent_id][tail_id: tail_id + 1])
                    # head_mention = ''.join(self._toks[sent_id][sent_dic[head_id][0]: sent_dic[head_id][1]]) if head_id in sent_dic else ''.join(self._toks[sent_id][head_id: head_id + 1])

                    # ignore the cases if tail_id/head_id not equal to the last tok_id of entity
                    if tail_id in sent_dic and head_id in sent_dic:
                        tail_mention = ''.join(self._toks[sent_id][sent_dic[tail_id][0]: sent_dic[tail_id][1]])
                        head_mention = ''.join(self._toks[sent_id][sent_dic[head_id][0]: sent_dic[head_id][1]])
                        sent_triplets.append((tail_mention, head_mention, rel))
            self._rel_mention_triplets.append(sent_triplets)

    def doc_to_xml(self, xml_file):
        current_tid = 1
        current_rid = 1
        span2tid = {}
        span2rel = {}
        for sent_id in range(len(self._doc_lines)):
            # update cid2tid
            for ner_tag, begin_cid, end_cid, mod_tag in self._mod_entities[sent_id]:
                span = (begin_cid, end_cid)
                span2tid[span] = (f"T{current_tid}", ner_tag)
                current_tid += 1
            # update ttid2rel
            for tail_span, head_span, rel in self._rel_detailed_triplets[sent_id]:
                span2rel[(tail_span, head_span)] = rel

        with open(xml_file, 'w', encoding='utf8') as fo:
            for sent_id in range(len(self._doc_lines)):
                sent_str = ""
                if not self._comments[sent_id].startswith(('#doc', '## line')):
                    sent_str += self._comments[sent_id].strip() + '\n'
                output_toks = copy.deepcopy(self._toks[sent_id])
                for ner_tag, begin_cid, end_cid, mod_tag in reversed(self._mod_entities[sent_id]):
                    span = (begin_cid, end_cid)
                    output_toks.insert(
                        end_cid,
                        f"</{ner_tag}>"
                    )
                    output_toks.insert(
                        begin_cid,
                        f"<{ner_tag} tid=\"{span2tid[span][0]}\"" +
                        (f" {MOD_DICT[mod_tag]}=\"{mod_tag}\"" if mod_tag != '_' else "") +
                        (f" DCT-Rel=\"{span2rel[(span, span)]}\"" if (span, span) in span2rel else "") +
                        ">"
                    )
                sent_str += ''.join(output_toks) + '\n'
                fo.write(sent_str)
                fo.write("\n")
            for (tail_span, head_span), rel in span2rel.items():
                tail_tid, tail_tag = span2tid[tail_span]
                head_tid, head_tag = span2tid[head_span]
                rel_tag = "brel" if (tail_tag != "Timex3" and head_tag != "Timex3") else "trel"
                if tail_tid != head_tid:
                    fo.write(f"<{rel_tag} rid=\"R{current_rid}\" arg1=\"{tail_tid}\" arg2=\"{head_tid}\" reltype=\"{rel}\" />\n")
                    current_rid += 1

    def doc_to_brat(self, brat_file, with_rel=True, is_prism=True):
        with open(brat_file + '.txt', 'w', encoding='utf8') as brat_txt, open(brat_file + '.ann', 'w', encoding='utf8') as brat_ann:
            line_start = 0
            eid_start = 1
            mid_start = 1
            rid_start = 1
            charid2eid = {}  # begin_tid: T{eid}
            prev_comment = ""
            for sent_id in range(len(self._doc_lines)):
                if self._comments[sent_id].startswith(('#doc', '## line')) and self._comments[sent_id] != prev_comment:
                    print(self._comments[sent_id])
                    comment_str = f'{self._comments[sent_id]}\n'
                    if comment_str.startswith('## line'):
                        brat_txt.write(comment_str)
                        line_start += len(comment_str)
                    prev_comment = self._comments[sent_id]
                sent_str = ''.join(self._toks[sent_id]) + '\n'
                brat_txt.write(sent_str)
                # entity: 'T{}\t{} {} {}\t{}', modality: 'A{}\t{} T{} {}'
                for ner_tag, begin_tid, end_tid, mod_tag in self._mod_entities[sent_id]:
                    begin_char_id = line_start + len(''.join(self._toks[sent_id][:begin_tid]))
                    end_char_id = line_start + len(''.join(self._toks[sent_id][:end_tid]))
                    char_surface = ''.join(self._toks[sent_id][begin_tid:end_tid])
                    if not is_prism:
                        brat_ann.write(f'T{eid_start}\t{ner_tag} {begin_char_id} {end_char_id}\t{char_surface}\n')
                    else:
                        # convert ner tag
                        brat_ann.write(f'T{eid_start}\t{NER_DICT[ner_tag.capitalize()]} {begin_char_id} {end_char_id}\t{char_surface}\n')
                    charid2eid[end_char_id - 1] = f'T{eid_start}'
                    if mod_tag != '_':
                        if not is_prism:
                            brat_ann.write(f'A{mid_start}\t{mod_tag} T{eid_start} {mod_tag}\n')
                        else:
                            brat_ann.write(f'A{mid_start}\t{MOD_DICT[mod_tag]} T{eid_start} {mod_tag}\n')
                        mid_start += 1
                    eid_start += 1
                if with_rel:
                    for tail_tid, head_tid, rel in self._rel_triplets[sent_id]:
                        tail_char_id = line_start + len(''.join(self._toks[sent_id][:tail_tid + 1])) - 1
                        head_char_id = line_start + len(''.join(self._toks[sent_id][:head_tid + 1])) - 1
                        if tail_char_id in charid2eid and head_char_id in charid2eid:
                            if tail_char_id != head_char_id:
                                brat_ann.write(f'R{rid_start}	{rel} Arg1:{charid2eid[tail_char_id]} Arg2:{charid2eid[head_char_id]}\n')
                                rid_start += 1
                            else:
                                brat_ann.write(f"A{mid_start}	DCT-Rel {charid2eid[tail_char_id]} {rel}\n")
                                mid_start += 1
                        else:
                            print(f'unknow eid of tail{tail_char_id}, head{head_char_id}')
                line_start += len(sent_str)








