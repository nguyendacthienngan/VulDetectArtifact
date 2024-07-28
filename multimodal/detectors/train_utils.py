import torch
import torch.nn as nn
import torch.nn.functional as F
from sequence.detectors.models.sysyer import build_model
from graph.detectors.models.ivdetect import IVDetectModel

class MultimodalVulnerabilityDetector(nn.Module):
    def __init__(self, sysevr_model, ivdetect_model):
        super(MultimodalVulnerabilityDetector, self).__init__()
        self.sysevr_model = sysevr_model
        self.ivdetect_model = ivdetect_model
        
        # Define the final layers
        combined_input_size = 512 + 128  # Adjusted based on the output sizes from sysevr and ivdetect
        self.final_fc1 = nn.Linear(combined_input_size, 128)
        self.final_fc2 = nn.Linear(128, 2)  # Adjust according to num_classes
        
    def forward(self, sysevr_input, ivdetect_input):
        # Forward pass through Sysevr model
        sysevr_output = self.sysevr_model.predict(sysevr_input, verbose=0)
        sysevr_output = torch.tensor(sysevr_output, dtype=torch.float32)
        
        # Forward pass through IVDetect model
        x, edge_index, batch = self.ivdetect_model.arguments_read(data=ivdetect_input)
        post_conv = x
        for i in range(self.ivdetect_model.layer_num - 1):
            post_conv = self.ivdetect_model.relu(self.ivdetect_model.convs[i](post_conv, edge_index))
        post_conv = self.ivdetect_model.convs[self.ivdetect_model.layer_num - 1](post_conv, edge_index)
        ivdetect_output = self.ivdetect_model.readout(post_conv, batch)
        
        # Concatenate the outputs from Sysevr and IVDetect
        combined_output = torch.cat((sysevr_output, ivdetect_output), dim=1)
        
        # Pass through the final layers
        x = F.relu(self.final_fc1(combined_output))
        x = self.final_fc2(x)
        
        return x

# Usage example
sysevr_model = build_model()
ivdetect_model = IVDetectModel()

multimodal_model = MultimodalVulnerabilityDetector(sysevr_model, ivdetect_model)
