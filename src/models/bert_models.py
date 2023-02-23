import torch
from torch import nn
from torch.nn import MSELoss

from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

class BertForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels, output_attentions, output_hidden_states):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.bert.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        logits = torch.sigmoid(logits) # This is important since our regression values are bounded [0-1]

        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )