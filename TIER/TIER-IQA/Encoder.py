import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


def freeze(module):
    """
    Freezes module's parameters.
    """
    
    for parameter in module.parameters():
        parameter.requires_grad = False
        
def odd_layer_freeze(module):
    for i in range(0,1):
        for n,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False
    for i in range(1,12,2):
        for n,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False
    
            
def even_layer_freeze(module):
    for i in range(0,12,2):
        for n,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False
            
def top_half_layer_freeze(module):
    for i in range(0,6,1):
        for n,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False

def bottom_half_layer_freeze(module):
    for i in range(6,12,1):
        for n,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.squeeze(1).unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.text_encoder = AutoModel.from_pretrained("./bert-base-uncased")
        self.pooler = MeanPooling()
        #freeze(self.text_encoder)
        #top_half_layer_freeze(self.text_encoder)

    def forward(self, ids, mask):
        out = self.text_encoder(input_ids=ids.squeeze(1), attention_mask=mask,
                         output_hidden_states=False)
        text_features = self.pooler(out.last_hidden_state, mask) #B, D

        return text_features



class MLP(nn.Module):
    def __init__(self, image_hidden_size):
        super(MLP, self).__init__()
        self.config = AutoConfig.from_pretrained("./bert-large-uncased")

        self.fc1 = nn.Linear((image_hidden_size + self.config.hidden_size), (image_hidden_size + self.config.hidden_size) // 2)
        self.fc2 = nn.Linear((image_hidden_size + self.config.hidden_size) // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
