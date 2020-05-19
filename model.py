import copy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import *
from torchcrf import CRF
import utils
import math

from typing import Dict, List, Tuple, Set, Optional
from functools import partial


class CertaintyClassifier(BertPreTrainedModel):
    
    def __init__(self, config, num_labels):
        super(CertaintyClassifier, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, dm_mask, token_type_ids=None, attention_mask=None, labels=None):
        last_layer_out, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        tag_rep = torch.bmm(dm_mask.unsqueeze(1).float(), last_layer_out)
        pooled_output = self.dropout(tag_rep)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

        
class SeqCertClassifier(BertPreTrainedModel):
    
    def __init__(self, config):
        super(SeqCertClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, ner_masks, ner_clab_masks, token_type_ids=None, attention_mask=None, labels=None):
        last_layer_out = self.bert(input_ids, token_type_ids, attention_mask)
        tag_rep = torch.bmm(ner_masks.float(), last_layer_out[0])
        pooled_output = self.dropout(tag_rep)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = ner_clab_masks.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            if not active_labels.shape[0]:
                return None
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits


class BertCRF(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCRF, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf_emission = nn.Linear(config.hidden_size, config.num_labels)
        self.crf_layer = CRF(self.num_labels, batch_first=True)
        self.crf_layer.reset_parameters()
        self.init_weights()

    def forward(self, input_embeds, attention_mask, labels=None):
        encoder_logits = self.bert(input_embeds, attention_mask=attention_mask)[0]
        emissions = self.crf_emission(encoder_logits)
        crf_loss = -self.crf_layer(self.dropout(emissions), mask=attention_mask, tags=labels, reduction='mean')
        return crf_loss

    def decode(self, input_embeds, attention_mask):
        encoder_logits = self.bert(input_embeds, attention_mask=attention_mask)[0]
        emissions = self.crf_emission(encoder_logits)
        return self.crf_layer.decode(emissions, mask=attention_mask)


class LSTMCRF(nn.Module):

    def __init__(self, embed_dim, hidden_dim, vocab_size, tag_size, pretrain_embed=None):
        super(LSTMCRF, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        if pretrain_embed is not None:
            self.word_embed = nn.Embedding.from_pretrained(
                torch.from_numpy(pretrain_embed),
                freeze=False
            )
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.hidden2tag = nn.Linear(2 * hidden_dim, tag_size)
        self.crf_layer = CRF(tag_size, batch_first=True)

    def forward(self, input_ix, attention_mask, labels=None):
        embedded_input = self.word_embed(input_ix)
        encoder_logits, _ = self.encoder(embedded_input)
        encoder_out = self.dropout(self.hidden2tag(encoder_logits))
        if labels is not None:
            crf_loss = -self.crf_layer(encoder_out, mask=attention_mask, tags=labels)
            return crf_loss
        else:
            return self.crf_layer.decode(encoder_out, mask=attention_mask)


class BertRel(BertPreTrainedModel):

    def __init__(self, config, ne_size, num_ne, num_rel):
        super(BertRel, self).__init__(config)
        self.num_rel = num_rel
        self.num_ne = num_ne
        self.ne_size = ne_size
        self.bert = BertModel(config)
        if ne_size:
            self.ne_embed = nn.Embedding(num_ne, ne_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.h2o = nn.Linear(2 * config.hidden_size + 2 * ne_size, num_rel)
        self.init_weights()

    def forward(self, tok_ix, attn_mask, tail_mask, tail_labs, head_mask, head_labs, rel_labs=None):
        # import pdb; pdb.set_trace()
        encoder_out = self.bert(tok_ix, attention_mask=attn_mask)[0]
        tail_rep = torch.bmm(tail_mask.unsqueeze(1).float(), encoder_out).squeeze(1)
        head_rep = torch.bmm(head_mask.unsqueeze(1).float(), encoder_out).squeeze(1)
        if self.ne_size:
            tail_ne = self.ne_embed(tail_labs)
            head_ne = self.ne_embed(head_labs)
            concat_out = self.dropout(F.relu(torch.cat((tail_rep, tail_ne, head_rep, head_ne), dim=-1)))
        else:
            concat_out = self.dropout(F.relu(torch.cat((tail_rep, head_rep), dim=-1)))
        logits = self.h2o(concat_out)
        outputs = (logits, )

        if rel_labs is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_rel), rel_labs.view(-1))
            outputs = (loss, ) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class HeadSelectModel(BertPreTrainedModel):

    def __init__(self,
                 config,
                 ner_emb_dim,
                 rel_emb_dim,
                 ner_num_labels,
                 rel_num_labels,
                 rel_prob_threshold):
        super(HeadSelectModel, self).__init__(config)

        self.ner_num_labels = ner_num_labels
        self.rel_num_labels = rel_num_labels
        self.rel_prob_threshold = rel_prob_threshold
        self.encoder = BertModel(config)
        self.ner_emb = nn.Embedding(ner_num_labels, embedding_dim=ner_emb_dim)
        self.rel_emb = nn.Embedding(rel_num_labels, embedding_dim=rel_emb_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.emit_layer = nn.Linear(config.hidden_size, ner_num_labels)
        self.crf_layer = CRF(ner_num_labels, batch_first=True)
        self.sel_u_mat = nn.Parameter(torch.Tensor(rel_emb_dim, config.hidden_size + ner_emb_dim))
        nn.init.kaiming_uniform_(self.sel_u_mat, a=math.sqrt(5))
        self.sel_w_mat = nn.Parameter(torch.Tensor(rel_emb_dim, config.hidden_size + ner_emb_dim))
        nn.init.kaiming_uniform_(self.sel_w_mat, a=math.sqrt(5))
        self.out_layer = nn.Linear(rel_emb_dim, rel_num_labels)
        self.selection_u = nn.Linear(config.hidden_size + ner_emb_dim,
                                     rel_emb_dim)
        self.selection_v = nn.Linear(config.hidden_size + ner_emb_dim,
                                     rel_emb_dim)
        self.selection_uv = nn.Linear(2 * rel_emb_dim,
                                      rel_emb_dim)
        # self.crf_layer.reset_parameters()
        # self.init_weights()

    def infer_rel(self, rel_logis, rel_mask, decoded_ner):
        pred_scores = (F.sigmoid(rel_logis) * rel_mask) > self.rel_prob_threshold
        # for tu in pred_rels:
        #     if tu[-1] != 0:
        #         print(tu)
        pred_triples = None
        return pred_scores

    def forward(self, input_ids, ner_mask, ner_labels=None, rel_labels=None):
        """
        :param input_ids: [b, l]
        :param ner_mask: [b, l]
        :param ner_labels: [b, l]
        :param rel_labels: [b, l, l, r]
        :return:
        """
        # import pdb;pdb.set_trace()

        batch_size, cls_max_len = input_ids.shape
        # print(input_ids[0])
        encoder_logits = self.bert(input_ids, attention_mask=ner_mask)[0]

        if ner_labels is not None and rel_labels is not None:
            ner_label_emb = self.ner_emb(ner_labels)  # [b, l, n]
        else:
            emissions = self.emit_layer(encoder_logits)
            decoded_ner = self.crf_layer.decode(emissions, mask=ner_mask)
            ner_labels = torch.tensor(utils.padding_2d(decoded_ner, cls_max_len)).cuda()
            print()
        ner_label_emb = self.ner_emb(ner_labels)

        ner_enhenced_logits = torch.cat((encoder_logits, ner_label_emb), dim=-1)

        # # word representations: [b, l, r_s]
        # sel_u_out = ner_enhenced_logits.matmul(self.sel_u_mat.t())  # [b, l, h_s] -> [b, l, r_s]
        # # print(sel_u_out.shape)
        # # head word representations: [b, l, r_s]
        # sel_w_out = ner_enhenced_logits.matmul(self.sel_w_mat.t())  # [b, l, h_s] -> [b, l, r_s]
        #
        # # broadcast sum: [b, l, 1, r] + [b, 1, l, r] = [b, l, l, r]
        # sel_out = sel_u_out.unsqueeze(2) + sel_w_out.unsqueeze(1)
        #
        # sel_out = torch.tanh(sel_out)
        #
        # sel_logits = self.out_layer(sel_out) # out: [b, l, l_h, rel_num_labels]

        u = self.selection_u(ner_enhenced_logits).unsqueeze(1).expand(batch_size, cls_max_len, cls_max_len, -1)
        v = self.selection_v(ner_enhenced_logits).unsqueeze(2).expand(batch_size, cls_max_len, cls_max_len, -1)
        uv = F.tanh(self.selection_uv(torch.cat((u, v), dim=-1)))
        sel_logits = torch.einsum('bijh,rh->bijr', uv, self.rel_emb.weight)

        # attention_mask: [b, l] -> sel_mask: [b, l, l, rel_num_labels]
        sel_mask = (ner_mask.unsqueeze(1) * ner_mask.unsqueeze(2)).unsqueeze(3).expand(-1, -1, -1, self.rel_num_labels)
        # sel_mask = (ner_mask.unsqueeze(2) * ner_mask.unsqueeze(1)).unsqueeze(2).expand(-1, -1, self.rel_num_labels, -1).transpose(2,3)
        # print(ner_mask.sum().item(), sel_mask.sum().item())
        loss_func = nn.BCEWithLogitsLoss(reduction='none')
        # print(rel_labels[0])
        # print(rel_labels.sum().item(), rel_labels.numel(), rel_labels.sum().item() / rel_labels.numel())
        # print(sel_logits.sum())
        if ner_labels is not None and rel_labels is not None:
            emissions = self.emit_layer(encoder_logits)
            crf_loss = -self.crf_layer(emissions, mask=ner_mask, tags=ner_labels, reduction='sum')
            rel_loss = loss_func(
                sel_logits,
                rel_labels
            )
            # print(rel_loss.masked_select(sel_mask).sum().item(), ner_mask.sum().item())
            rel_loss = rel_loss.masked_select(sel_mask).sum()   # rel_mean rel_loss
            return crf_loss, rel_loss
        else:
            infered_rel = self.infer_rel(sel_logits, sel_mask, decoded_ner)
            return decoded_ner, infered_rel


class MultiHeadSelection(nn.Module):
    def __init__(self, bert_url, bio_emb_size, bio_vocab, rel_emb_size, relation_vocab,
                 hidden_size=768, reduction='token_mean', gpu_id=0):
        super(MultiHeadSelection, self).__init__()

        bio_num = len(bio_vocab)
        rel_num = len(relation_vocab)

        self.gpu = gpu_id

        self.reduction = reduction

        self.bio_emb = nn.Embedding(num_embeddings=bio_num,
                                    embedding_dim=bio_emb_size)

        self.relation_emb = nn.Embedding(num_embeddings=rel_num,
                                         embedding_dim=rel_emb_size)

        self.encoder = BertModel.from_pretrained(bert_url)

        self.activation = nn.Tanh()

        self.crf_tagger = CRF(bio_num, batch_first=True)

        self.crf_emission = nn.Linear(hidden_size, bio_num)

        self.mhs_u = nn.Linear(hidden_size + bio_emb_size,
                               rel_emb_size, bias=False)
        self.mhs_v = nn.Linear(hidden_size + bio_emb_size,
                               rel_emb_size, bias=False)

        self.sel_u_mat = nn.Parameter(torch.Tensor(rel_emb_size, hidden_size + bio_emb_size))
        nn.init.kaiming_uniform_(self.sel_u_mat, a=math.sqrt(5))

        self.sel_v_mat = nn.Parameter(torch.Tensor(rel_emb_size, hidden_size + bio_emb_size))
        nn.init.kaiming_uniform_(self.sel_v_mat, a=math.sqrt(5))

        self.drop_uv = nn.Dropout(p=0.2)
        self.rel_linear = nn.Linear(rel_emb_size, rel_num)

        self.relation_vocab = relation_vocab
        self.bio_vocab = bio_vocab
        self.id2bio = {v: k for k, v in self.bio_vocab.items()}

    def inference(self, mask, text_list, decoded_tag, selection_logits):
        # mask: B x L x R x L
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(-1, -1, len(self.relation_vocab), -1)
        selection_tags = (torch.sigmoid(selection_logits) *
                          selection_mask.float()) > 0.5

        selection_triplets = self.selection_decode(text_list, decoded_tag,
                                                   selection_tags)
        return selection_triplets

    def masked_BCEloss(self, mask, selection_logits, selection_gold):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
                              -1, -1, len(self.relation_vocab),
                              -1)  # batch x seq x rel x seq
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            selection_gold,
                                                            reduction='none')
        # print(selection_loss[0])
        # print(selection_loss.masked_select(selection_mask).sum().item(), mask.sum().item())
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        if self.reduction in ['token_mean']:
            selection_loss /= mask.sum()
        return selection_loss

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.6f}, L_crf: {:.6f}, L_selection: {:.6f}, epoch: {}/{}:".format(
            output['loss'].item(), output['crf_loss'].item(),
            output['selection_loss'].item(), epoch, epoch_num)

    def forward(self, tokens, mask, bio_gold, selection_gold, text_list, bio_text, spo_gold, is_train: bool):

        B, L = tokens.shape
        o = self.encoder(tokens, attention_mask=mask)[0]  # last hidden of BERT

        emi = self.crf_emission(o)

        output = {}

        crf_loss = 0.

        if is_train:
            crf_loss = -self.crf_tagger(emi, bio_gold,
                                    mask=mask,
                                    reduction=self.reduction)
        else:
            decoded_tag = self.crf_tagger.decode(emissions=emi, mask=mask)
            decoded_bio_text = [list(map(lambda x: self.id2bio[x], tags)) for tags in decoded_tag]
            output['decoded_tag'] = decoded_bio_text
            output['gold_tags'] = bio_text
            temp_tag = copy.deepcopy(decoded_tag)
            for line in temp_tag:
                line.extend([self.bio_vocab['O']] * (L - len(line)))
            bio_gold = torch.tensor(temp_tag).cuda(self.gpu)

        output['crf_loss'] = crf_loss

        tag_emb = self.bio_emb(bio_gold)

        o = torch.cat((o, tag_emb), dim=2)

        # forward multi head selection
        u = self.mhs_u(o).unsqueeze(1).expand(B, L, L, -1)
        v = self.mhs_v(o).unsqueeze(2).expand(B, L, L, -1)
        uv = self.activation(u + v)
        # uv = self.activation(torch.cat((u, v, (u - v).abs()), dim=-1))
        # # correct one

        # word representations: [b, l, r_s]
        # broadcast sum: [b, l, 1, h] + [b, 1, l, h] = [b, l, l, h]
        # u = o.matmul(self.sel_u_mat.t())  # [b, l, h_s] -> [b, l, r_s]
        # v = o.matmul(self.sel_v_mat.t())  # [b, l, h_s] -> [b, l, r_s]
        # uv = self.activation(u.unsqueeze(2) + v.unsqueeze(1))
        uv = self.drop_uv(uv)
        # selection_logits = torch.einsum('bijh,rh->birj', [uv, self.relation_emb.weight])
        selection_logits = self.rel_linear(uv).transpose(2, 3)

        if not is_train:
            output['selection_triplets'] = self.inference(
                mask, text_list, decoded_tag, selection_logits)
            output['spo_gold'] = spo_gold

        selection_loss = torch.tensor([0.]).cuda(self.gpu)
        if is_train:
            selection_loss = self.masked_BCEloss(mask, selection_logits,
                                                 selection_gold)
        output['selection_loss'] = selection_loss

        loss = crf_loss + selection_loss

        output['loss'] = loss

        output['description'] = partial(self.description, output=output)
        return output

    def selection_decode(self, text_list, sequence_tags, selection_tags):
        reversed_relation_vocab = {
            v: k for k, v in self.relation_vocab.items()
        }

        reversed_bio_vocab = {v: k for k, v in self.bio_vocab.items()}

        text_list = list(map(list, text_list))

        def find_entity(pos, text, sequence_tags, return_text=True):
            entity = []

            if sequence_tags[pos][0] in ['B', 'O']:
                entity.append(pos)
            else:
                temp_entity = []
                while sequence_tags[pos][0] == 'I':
                    temp_entity.append(pos)
                    pos -= 1
                    if pos < 0:
                        break
                    if sequence_tags[pos][0] == 'B':
                        temp_entity.append(pos)
                        break
                entity = list(reversed(temp_entity))
            return [text[index] for index in entity] if return_text else entity

        batch_num = len(sequence_tags)
        result = [[] for _ in range(batch_num)]
        idx = torch.nonzero(selection_tags.cpu())

        for i in range(idx.size(0)):
            b, s, p, o = idx[i].tolist()

            predicate = reversed_relation_vocab[p]
            if predicate == 'N':
                continue
            tags = list(map(lambda x: reversed_bio_vocab[x], sequence_tags[b]))
            object = find_entity(o, text_list[b], tags)
            subject = find_entity(s, text_list[b], tags)
            assert object != [] and subject != []

            rel_triplet = {
                'subject': subject,
                'predicate': predicate,
                'object': object
            }
            result[b].append(rel_triplet)
        return result
