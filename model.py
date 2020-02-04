import torch
from torch import nn
from torch.nn import CrossEntropyLoss
# from pytorch_pretrained_bert.modeling import BertPreTrainedModel
# from pytorch_pretrained_bert import BertModel, BertTokenizer, WEIGHTS_NAME, CONFIG_NAME
from transformers import *


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
    
    def __init__(self, config, num_labels):
        super(SeqCertClassifier, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, ner_masks, ner_clab_masks, token_type_ids=None, attention_mask=None, labels=None):
        last_layer_out, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        tag_rep = torch.bmm(ner_masks.float(), last_layer_out)
        pooled_output = self.dropout(tag_rep)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = ner_clab_masks.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits


def save_bert_model(model, tokenizer, model_dir):
    import os
    if os.path.exists(model_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(model_dir))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(model_dir, CONFIG_NAME)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(model_dir)
