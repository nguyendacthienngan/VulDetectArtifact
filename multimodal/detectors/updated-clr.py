import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

class CLRModel(nn.Module):
    def __init__(self, model_name='roberta-base', num_labels=2):
        super(CLRModel, self).__init__()
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.classification_head = RobertaClassificationHead(self.config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # Last hidden state
        
        # Extract the CLS token as a pooled output
        pooled_output = sequence_output[:, 0, :]
        
        # Pass through the classification head to get logits
        logits = self.classification_head(sequence_output)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        if labels is not None:
            # Compute the loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            return loss, probs, pooled_output  # Return the loss, probabilities, and features
        else:
            return probs, pooled_output  # Return the probabilities and features
