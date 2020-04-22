import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import *
from torchcrf import CRF
import utils


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
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        self.crf_layer = CRF(self.num_labels, batch_first=True)
        self.crf_layer.reset_parameters()
        self.init_weights()

    def forward(self, input_embeds, attention_mask, labels=None):
        encoder_logits = self.bert(input_embeds, attention_mask=attention_mask)[0]
        emissions = self.linear(encoder_logits)
        crf_loss = -self.crf_layer(self.dropout(emissions), mask=attention_mask, tags=labels, reduction='mean')
        return crf_loss

    def decode(self, input_embeds, attention_mask):
        encoder_logits = self.bert(input_embeds, attention_mask=attention_mask)[0]
        emissions = self.linear(encoder_logits)
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

    def __init__(self, config, num_ne, num_rel):
        super(BertRel, self).__init__(config)
        self.num_rel = num_rel
        self.num_ne = num_ne
        self.bert = BertModel(config)
        self.ne_embed = nn.Embedding(num_ne, 100)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.h2o = nn.Linear(2 * config.hidden_size + 200, num_rel)
        self.init_weights()

    def forward(self, tok_ix, attn_mask, tail_mask, tail_labs, head_mask, head_labs, rel_labs=None):
        encoder_out = self.bert(tok_ix, attention_mask=attn_mask)[0]
        tail_rep = torch.bmm(tail_mask.unsqueeze(1).float(), encoder_out).squeeze(1)
        head_rep = torch.bmm(head_mask.unsqueeze(1).float(), encoder_out).squeeze(1)
        tail_ne = self.ne_embed(tail_labs)
        head_ne = self.ne_embed(head_labs)
        concat_out = self.dropout(F.relu(torch.cat((tail_rep, tail_ne, head_rep, head_ne), dim=-1)))
        logits = self.h2o(concat_out)
        outputs = (logits, )

        if rel_labs is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_rel), rel_labs.view(-1))
            outputs = (loss, ) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class HeadSelectModel(BertPreTrainedModel):

    def __init__(self, config):
        super(HeadSelectModel, self).__init__(config)
        # model
        self.ner_num_labels = config.ner_num_labels
        self.rel_num_labels = config.rel_num_labels
        self.rel_prob_threshold = config.rel_prob_threshold
        self.bert = BertModel(config)
        self.ner_emb = nn.Embedding(config.ner_num_labels, config.ner_emb_dim)
        self.rel_emb = nn.Embedding(config.rel_num_labels, config.rel_emb_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.emit_layer = nn.Linear(config.hidden_size, config.ner_num_labels)
        self.crf_layer = CRF(self.num_labels, batch_first=True)
        self.crf_layer.reset_parameters()
        self.sel_u_mat = nn.Parameter(torch.Tensor(config.rel_emb_dim, config.hidden_size + config.ner_emb_dim))
        nn.init.kaiming_uniform_(self.sel_u_mat, a=math.sqrt(5))
        self.sel_w_mat = nn.Parameter(torch.Tensor(config.rel_emb_dim, config.hidden_size + config.ner_emb_dim))
        nn.init.kaiming_uniform_(self.sel_w_mat, a=math.sqrt(5))
        self.out_layer = nn.Linear(config.rel_emb_dim, config.rel_num_labels)
        self.init_weights()

    def infer_rel(self, rel_logis, rel_mask, decoded_ner):
        pred_rel = (F.sigmoid(rel_logis) * rel_mask) > self.rel_prob_threshold
        pred_triples = None
        return pred_triples

    def forward(self, input_ids, ner_mask, ner_labels=None, rel_labels=None):
        """
        :param input_ids: [b, l]
        :param ner_mask: [b, l]
        :param ner_labels: [b, l]
        :param rel_labels: [b, l, l, r]
        :return:
        """
        batch_size, max_len = input_ids.shape

        encoder_logits = self.bert(input_ids, attention_mask=ner_mask)[0]

        if ner_labels is not None and rel_labels is not None:
            ner_label_emb = self.ner_emb(ner_labels)  # [b, l, n]
        else:
            emissions = self.emit_layer(encoder_logits)
            decoded_ner_list = self.crf_layer.decode(emissions, mask=ner_mask)
            decoded_ner = torch.tensor(utils.padding_2d(decoded_ner_list, max_len)).cuda()
            ner_label_emb = self.ner_emb(decoded_ner)

        ner_enhenced_logits = torch.cat((encoder_logits, ner_label_emb), dim=-1)

        # word representations: [b, l, r_s]
        sel_u_out = ner_enhenced_logits.matmul(self.sel_u_mat.t())  # [b, l, h_s] -> [b, l, r_s]
        # head word representations: [b, l, r_s]
        sel_w_out = ner_enhenced_logits.matmul(self.sel_w_mat.t())  # [b, l, h_s] -> [b, l, r_s]

        # broadcast sum: [b, l, 1, r] + [b, 1, l, r] = [b, l, l, r]
        sel_out = sel_u_out.unsqueeze(2) + sel_w_out.unsqueeze(1)

        sel_out = F.dropout(F.relu(sel_out))

        sel_logits = self.out_layer(sel_out)  # out: [b, l, l_h, rel_num_labels]

        # attention_mask: [b, l] -> sel_mask: [b, l, l, rel_num_labels]
        sel_mask = (ner_mask.unsqueeze(1) * ner_mask.unsqueeze(2)).unsqueeze(3).expand(-1, -1, -1, self.rel_num_labels)

        if ner_labels is not None and rel_labels is not None:
            emissions = self.linear(encoder_logits)
            crf_loss = -self.crf_layer(self.dropout(emissions), mask=ner_mask, tags=ner_labels, reduction='mean')
            rel_loss = F.binary_cross_entropy_with_logits(
                F.sigmoid(sel_logits),
                rel_labels,
                reduction='none'
            )
            rel_loss = rel_loss.masked_select(sel_mask).sum() / sel_mask.sum()  # mean rel_loss
            return crf_loss + rel_loss
        else:
            infered_rel = self.infer_rel(sel_logits, sel_mask, decoded_ner_list)
            return decoded_ner_list, infered_rel
