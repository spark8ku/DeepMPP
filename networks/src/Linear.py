import torch
import torch.nn as nn

class Linears(nn.Module):
    def __init__(self, in_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False, last= False):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_sum = residual_sum
        self.last = last
        hidden_dim = (in_feats+out_feats)//2
        for i in range(n_layers):
            if i==0:
                _in_feats = in_feats
            else:
                _in_feats = hidden_dim
            if i==n_layers-1:
                _out_feats = out_feats
            else:
                _out_feats = hidden_dim
            self.linears.append(nn.Linear(_in_feats, _out_feats))
            if self.batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(_out_feats))
            if residual_sum:
                if in_feats!=out_feats:
                    self.residual_layer = nn.Linear(_in_feats, _out_feats) 

    def forward(self, x):
        count=0
        for linear in self.linears:
            count+=1
            h = linear(x)
            if count==len(self.linears) and self.last:
                return h
            else:
                if self.batch_norm:
                    h = self.batch_norms[count-1](h)
                h = self.activation(h)
            h = self.dropout(h)
            if self.residual_sum:
                if x.shape[1]==h.shape[1]:
                    x = h + x
            else:
                x = h
        return x
    
