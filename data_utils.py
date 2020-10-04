#!/usr/bin/env python
# coding: utf-8
import copy

MOD_DICT = {
        'positive': 'certainty', 'suspicious': 'certainty', 'negative': 'certainty', 'general': 'certainty',
        'executed': 'state', 'negated': 'state', 'scheduled': 'state', 'other': 'state',
        'DATE': 'type',  'TIME': 'type', 'DURATION': 'type', 'CC': 'type', 'SET': 'type', 'AGE': 'type', 'MISC': 'type'

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


# def span_to_mask(entity_span):

# MultiheadConll document class
class MultiheadConll(object):

    def __init__(self, conll_file, prune_type=False, empty_comment=('#doc', '## line')):
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
        self.load_doc(conll_file, empty_comment)
        self.update_columns()
        self.update_entities()
        self.update_mod_entities()
        self.update_rel_triplets()
        self.update_rel_detailed_triplets(prune_type=prune_type)

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
                s_toks.append(tok_items[1].replace('[JASP]', '\u3000'))
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

    def update_rel_detailed_triplets(self, prune_type):
        for sent_id in range(len(self._doc_lines)):
            sent_dic = {(entity[-1]-1): (entity[1:3], entity[0]) for entity in self._entities[sent_id]}
            # print(sent_dic)
            sent_triplets = []
            for tail_id, head_id, rel in self._rel_triplets[sent_id]:
                if tail_id in sent_dic and head_id in sent_dic:
                    tail_tag = sent_dic[tail_id][1]
                    head_tag = sent_dic[head_id][1]
                    if prune_type:
                        if head_tag == 'problem':
                            tail_span = sent_dic[tail_id][0]
                            head_span = sent_dic[head_id][0]
                            sent_triplets.append((tail_span, head_span, rel))
                    else:
                        tail_span = sent_dic[tail_id][0]
                        head_span = sent_dic[head_id][0]
                        sent_triplets.append((tail_span, head_span, rel))
            self._rel_detailed_triplets.append(sent_triplets)

    def doc_to_xml(self, xml_file):
        with open(xml_file, 'w', encoding='utf8') as fo:
            for sent_id in range(len(self._doc_lines)):
                sent_str = ""
                if not self._comments[sent_id].startswith("#doc"):
                    sent_str += self._comments[sent_id].strip() + '\n'
                output_toks = copy.deepcopy(self._toks[sent_id])
                for ner_tag, begin_tid, end_tid, mod_tag in reversed(self._mod_entities[sent_id]):
                    output_toks.insert(
                        end_tid,
                        f"</{ner_tag}>"
                    )
                    output_toks.insert(
                        begin_tid,
                        f"<{ner_tag} {MOD_DICT[mod_tag]}=\"{mod_tag}\">" if mod_tag != '_' else f"<{ner_tag}>"
                    )
                sent_str += ''.join(output_toks) + '\n'
                fo.write(sent_str)

    def doc_to_brat(self, brat_file, with_rel=False):
        with open(brat_file + '.txt', 'w', encoding='utf8') as brat_txt, open(brat_file + '.ann', 'w', encoding='utf8') as brat_ann:
            line_start = 0
            eid_start = 1
            mid_start = 1
            rid_start = 1
            charid2eid = {}  # begin_tid: T{eid}
            for sent_id in range(len(self._doc_lines)):
                if not self._comments[sent_id].startswith('#doc'):
                    comment_str = f'{self._comments[sent_id]}\n'
                    brat_txt.write(comment_str)
                    line_start += len(comment_str)
                sent_str = ''.join(self._toks[sent_id]) + '\n'
                brat_txt.write(sent_str)
                # entity: 'T{}\t{} {} {}\t{}', modality: 'A{}\t{} T{} {}'
                for ner_tag, begin_tid, end_tid, mod_tag in self._mod_entities[sent_id]:
                    begin_char_id = line_start + len(''.join(self._toks[sent_id][:begin_tid]))
                    end_char_id = line_start + len(''.join(self._toks[sent_id][:end_tid]))
                    char_surface = ''.join(self._toks[sent_id][begin_tid:end_tid])
                    brat_ann.write(f'T{eid_start}\t{NER_DICT[ner_tag]} {begin_char_id} {end_char_id}\t{char_surface}\n')
                    charid2eid[end_char_id - 1] = f'T{eid_start}'
                    if mod_tag != '_':
                        brat_ann.write(f'A{mid_start}\t{MOD_DICT[mod_tag]} T{eid_start} {mod_tag}\n')
                        mid_start += 1
                    eid_start += 1
                if with_rel:
                    for tail_tid, head_tid, rel in self._rel_triplets[sent_id]:
                        tail_char_id = line_start + len(''.join(self._toks[sent_id][:tail_tid + 1])) - 1
                        head_char_id = line_start + len(''.join(self._toks[sent_id][:head_tid + 1])) - 1
                        if tail_char_id in charid2eid and head_char_id in charid2eid:
                            brat_ann.write(f'R{rid_start}	{rel} Arg1:{charid2eid[tail_char_id]} Arg2:{charid2eid[head_char_id]}\n')
                            rid_start += 1
                        else:
                            print(f'unknow eid of tail{tail_char_id}, head{head_char_id}')
                line_start += len(sent_str)








