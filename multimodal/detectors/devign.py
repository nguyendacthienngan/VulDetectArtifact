import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GatedGraphConv

class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types=3, num_steps=6):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.conv_l1 = nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = nn.MaxPool1d(3, stride=2)
        self.conv_l2 = nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, dataset, cuda=False):
        features = g.ndata['_WORD2VEC']
        edge_types = g.edata["_ETYPE"]
        outputs = self.ggnn(g, features, edge_types) 
    
        g.ndata['GGNNOUTPUT'] = outputs
        
        x_i, h_i = self.unbatch_features(g)
        x_i = torch.stack(x_i)
        h_i = torch.stack(h_i) 
        c_i = torch.cat((h_i, x_i), dim=-1) 
        Y_1 = self.maxpool1(
            F.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            F.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            F.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            F.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        temp = torch.cat((Y_2.sum(1), Z_2.sum(1)), 1)
        result = self.sigmoid(avg).squeeze(dim=-1) 
        return result, avg, temp

    def unbatch_features(self, g):
        x_i = []
        h_i = []
        max_len = -1
        for g_i in dgl.unbatch(g):
            x_i.append(g_i.ndata['_WORD2VEC']) 
            h_i.append(g_i.ndata['GGNNOUTPUT'])
            max_len = max(g_i.number_of_nodes(), max_len)
        for i, (v, k) in enumerate(zip(x_i, h_i)):
            x_i[i] = torch.cat(
                (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                                device=v.device)), dim=0)
            h_i[i] = torch.cat(
                (k, torch.zeros(size=(max_len - k.size(0), *(k.shape[1:])), requires_grad=k.requires_grad,
                                device=k.device)), dim=0)
        return x_i, h_i


class CombinedModel(nn.Module):
    def __init__(self, clr_model, devign_model, combined_output_dim=128):
        super(CombinedModel, self).__init__()
        self.clr_model = clr_model
        self.devign_model = devign_model
        
        # CLR and DevignModel output dimensions
        clr_output_dim = clr_model.config.hidden_size
        devign_output_dim = devign_model.concat_dim
        
        self.combined_output_dim = combined_output_dim
        
        # Linear layer to combine the features
        self.fc1 = nn.Linear(clr_output_dim + devign_output_dim, combined_output_dim)
        self.fc2 = nn.Linear(combined_output_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, g=None, dataset=None, labels=None, weight=None):
        # Get CLR model features
        clr_loss, clr_probs, clr_features = self.clr_model(input_ids, attention_mask, labels)
        
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

# Initialize the models
clr_model = CLRModel(model_name='roberta-base', num_labels=2)
devign_model = DevignModel(input_dim=100, output_dim=128)  # Adjust dimensions based on your data
combined_model = CombinedModel(clr_model, devign_model)
