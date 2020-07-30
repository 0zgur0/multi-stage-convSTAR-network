import torch
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm, Linear, Sequential, ReLU

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TransformerModel']

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class TransformerModel(nn.Module):
    def __init__(self, input_dim=4, num_classes=52, sequencelength=71, d_model=64, n_head=3, n_layers=3,
                 d_inner=256, activation="relu", dropout=0.4):

        super(TransformerModel, self).__init__()
        self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"

        encoder_layer = TransformerEncoderLayer(d_model = d_model, nhead = n_head, dim_feedforward = d_inner, dropout = dropout)
        encoder_norm = LayerNorm(d_model)

        self.sequential = Sequential(
            Linear(input_dim, d_model),
            ReLU(),
            TransformerEncoder(encoder_layer, n_layers, encoder_norm),
            Flatten(),
            ReLU(),
            Linear(d_model*sequencelength, num_classes)
        )

    def forward(self,x):
        logits = self.sequential(x)
        logprobabilities = F.log_softmax(logits, dim=-1)
        return logprobabilities


class TransformerModel2(nn.Module):
    def __init__(self, input_dim=4, num_classes=52, num_classes_l1=6, num_classes_l2=20, sequencelength=71, d_model=64, n_head=3, n_layers=3,
                 d_inner=256, activation="relu", dropout=0.4):

        super(TransformerModel2, self).__init__()
        self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"

        encoder_layer = TransformerEncoderLayer(d_model = d_model, nhead = n_head, dim_feedforward = d_inner, dropout = dropout)
        encoder_norm = LayerNorm(d_model)

        self.sequential = Sequential(
            Linear(input_dim, d_model),
            ReLU(),
            TransformerEncoder(encoder_layer, n_layers, encoder_norm),
            ReLU()
        )

        self.linear1 = Linear(d_model, num_classes_l1)
        self.linear2 = Linear(d_model, num_classes_l2)
        self.linear = Linear(d_model, num_classes)

    def forward(self,x):
        encoder_out = self.sequential(x)
        
        logits_l1 = self.linear1(encoder_out[:,0,:])
        logits_l2 = self.linear2(encoder_out[:,1,:])
        logits = self.linear(encoder_out[:,2,:])

        logprobabilities = F.log_softmax(logits, dim=-1)
        logprobabilities_l1 = F.log_softmax(logits_l1, dim=-1)
        logprobabilities_l2 = F.log_softmax(logits_l2, dim=-1)

        return logprobabilities, logprobabilities_l1, logprobabilities_l2
    
    
    
class TransformerModel3(nn.Module):
    def __init__(self, input_dim=4, num_classes=52, sequencelength=71, d_model=64, n_head=3, n_layers=3,
                 d_inner=256, activation="relu", dropout=0.4):
        super(TransformerModel3, self).__init__()
        self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"
        print('Transformer-3')

        encoder_layer = TransformerEncoderLayer(d_model = d_model, nhead = n_head, dim_feedforward = d_inner, dropout = dropout)
        encoder_norm = LayerNorm(d_model)

        self.attention = Sequential(
            Linear(input_dim, d_model),
            ReLU(),
            TransformerEncoder(encoder_layer, n_layers, encoder_norm),
        )
        
        self.pooling = nn.MaxPool1d(3)
        
        self.final = Sequential(
            Linear(d_model  + input_dim * 2, d_model * 3),
            nn.Dropout(p=dropout),
            ReLU(),
            Linear(d_model * 3, num_classes) 
            )
        
        #self.bayes_linear = Linear(input_dim, d_model)

    def forward(self,x):
        bayes = torch.sum(x, dim=1)
        #bayes = self.bayes_linear(bayes)
        
        enc = self.attention(x)
        enc = enc.permute(0,2,1)
        enc = self.pooling(enc)
        enc = enc.squeeze()
        
        #fusion
        agg = torch.cat((bayes,enc,x[:,-1,:]),1)
        
        logits = self.final(agg)
        logits = F.log_softmax(logits, dim=-1)
        return logits
    
    
    
    