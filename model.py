import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import *
from torchcrf import CRF


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
        # self.crf_layer.reset_parameters()
        self.init_weights()

    def crf_forward(self, input_embeds, attention_mask, labels=None):
        encoder_logits = self.bert(input_embeds, attention_mask=attention_mask)[0]
        encoder_out = self.linear(encoder_logits)
        crf_loss = self.crf_layer(self.dropout(encoder_out), mask=attention_mask, tags=labels)
        return crf_loss

    def decode(self, input_embeds, attention_mask):
        encoder_logits = self.bert(input_embeds, attention_mask=attention_mask)[0]
        encoder_out = self.linear(encoder_logits)
        return self.crf_layer.decode(encoder_out, mask=attention_mask)


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


