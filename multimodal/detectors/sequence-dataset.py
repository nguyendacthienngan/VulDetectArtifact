import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

# Similarity measure for contrastive learning
class Similarity(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp 
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

# ModelWithCLR definition
class ModelWithCLR(nn.Module):   
    def __init__(self, encoder, config, tokenizer, clr_temp=0.07, clr_mask=True):
        super(ModelWithCLR, self).__init__()
        self.encoder = encoder
        self.cls_head = RobertaClassificationHead(config)
        self.sim = Similarity(temp=clr_temp)
        self.clr_mask = clr_mask
        self.tokenizer = tokenizer
    
    def forward(self, group_inputs=None, group_labels=None): 
        batch_size = group_inputs.size(0)
        group_size = group_inputs.size(1) 
        
        # Duplicate group_inputs for contrastive learning
        dup_group_inputs = group_inputs.repeat(1, 2, 1)
        input_ids = dup_group_inputs.view((-1, group_inputs.size(-1)))
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        # Pass through encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.cls_head(outputs[0]).view((batch_size, 2, group_size, 1)).contiguous()
        cls_output = cls_output[:, 0, :, :].contiguous().view((batch_size * group_size, 1)).contiguous()
        
        # Contrastive learning output
        clr_output = outputs.pooler_output.view((batch_size, 2, group_size, -1)).contiguous()
        z_1, z_2 = clr_output[:, 0, :, :], clr_output[:, 1, :, :]
        cos_sim = self.sim(z_1.unsqueeze(2), z_2.unsqueeze(1))
        
        # Contrastive loss
        clr_labels = torch.arange(group_size).long().to(group_inputs.device).unsqueeze(0).expand(batch_size, group_size)
        if self.clr_mask:
            clr_mask = torch.eye(group_size).to(group_inputs.device).unsqueeze(0).expand(batch_size, group_size, group_size)
            clr_mask[:, 0, :] = 1
            clr_mask[:, :, 0] = 1
            masked_cos_sim = cos_sim.masked_fill(clr_mask == 0, -1e9)
            softmax_output = F.log_softmax(masked_cos_sim, dim=2)
            clr_loss_fct = nn.NLLLoss()
            clr_loss = clr_loss_fct(softmax_output, clr_labels)
        else:
            clr_loss_fct = nn.CrossEntropyLoss()
            clr_loss = clr_loss_fct(cos_sim, clr_labels)
        
        # Classification loss
        labels = group_labels.view(-1)
        prob = torch.sigmoid(cls_output)
        labels = labels.float()
        cls_loss = - (torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)).mean()

        # Combined loss
        loss = clr_loss + cls_loss
        
        return loss, prob, labels

