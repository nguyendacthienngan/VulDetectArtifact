import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from dgl.nn.pytorch import GatedGraphConv
import dgl

class CombinedModel(nn.Module):
    def __init__(self, clr_model, devign_model, combined_output_dim=128):
        super(CombinedModel, self).__init__()
        self.clr_model = clr_model
        self.devign_model = devign_model
        
        # Assuming CLR and DevignModel have the same output dimension
        clr_output_dim = clr_model.config.hidden_size
        devign_output_dim = devign_model.concat_dim
        
        self.combined_output_dim = combined_output_dim
        
        # Linear layer to combine the features
        self.fc1 = nn.Linear(clr_output_dim + devign_output_dim, combined_output_dim)
        self.fc2 = nn.Linear(combined_output_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, group_inputs=None, g=None, dataset=None, labels=None, weight=None):
        # Get CLR model features
        clr_loss, clr_probs, clr_features = self.clr_model(group_inputs, labels)
        
        # Get Devign model features
        devign_result, devign_avg, devign_features = self.devign_model(g, dataset)
        
        # Combine the features
        combined_features = torch.cat((clr_features, devign_features), dim=1)
        
        # Pass through additional layers
        x = F.relu(self.fc1(combined_features))
        logits = self.fc2(x)
        prob = self.sigmoid(logits).squeeze(dim=-1)
        
        if labels is not None:
            loss = F.binary_cross_entropy(prob, labels.float(), weight=weight)
            return loss, prob
        else:
            return prob

# Initialize the model
# clr_model = ...  # Load or initialize the CLR model
# devign_model = DevignModel(input_dim=..., output_dim=...)
# combined_model = CombinedModel(clr_model, devign_model)
