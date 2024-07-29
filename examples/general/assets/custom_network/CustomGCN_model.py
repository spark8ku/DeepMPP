import torch.nn as nn
import torch
from dgl.nn import SumPooling

import torch.nn as nn
import torch
import dgl


class GCN_layer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout=0.2, batch_norm=False, residual_sum = False):
        super(GCN_layer, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.linear = nn.Linear(in_feats, out_feats)
        self.residual_sum = residual_sum
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_feats)
        if residual_sum:
            if in_feats!=out_feats:
                self.residual_layer = nn.Linear(in_feats, out_feats)

    def forward(self, graph, node_feats):
        with graph.local_scope():
            graph= dgl.add_self_loop(graph)
            h = self.linear(node_feats)
            graph.ndata['h'] = h
            graph.update_all(dgl.function.copy_u('h', 'm'),
                            dgl.function.sum(msg='m', out='h'))
            h = graph.ndata['h']
            if self.batch_norm:
                h = self.bn(h)
            h = self.activation(h)
            h = self.dropout(h)
            if self.residual_sum:
                if node_feats.shape[1]!=h.shape[1]:
                    node_feats = self.residual_layer(node_feats)
                h = h + node_feats
        return h
    
class GCNs(nn.Module):
    def __init__(self,  in_feats, hidden_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        super(GCNs, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i==0:
                _in_feats = in_feats
            else:
                _in_feats = hidden_feats
            if i==n_layers-1:
                _out_feats = out_feats
            else:
                _out_feats = hidden_feats
            self.layers.append(GCN_layer(_in_feats, _out_feats, activation, dropout, batch_norm, residual_sum))
        
    def forward(self, graph, node_feats):
        for layer in self.layers:
            node_feats = layer(graph, node_feats)
        return node_feats
    
class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()
        
        hidden_dim = config.get('hidden_dim', 64)
        gcn_layers = config.get('conv_layers', 4)
        dropout = config.get('dropout', 0.2)
        linear_layers = config.get('linear_layers', 2)
        target_dim = config['target_dim']


        self.node_embedding = nn.Linear(config['node_dim'], hidden_dim)

        self.GCNs = GCNs(hidden_dim, hidden_dim, hidden_dim, nn.ReLU(), gcn_layers, dropout, False, True) 
        self.Linears = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim) for i in range(linear_layers-1)]+[nn.Linear(hidden_dim, target_dim)])
        
    def forward(self, graph, node_feats,**kwargs):
        h = self.node_embedding(node_feats)
        h = self.GCNs(graph, h)
        h = SumPooling()(graph,h)
        h = self.Linears(h)
        return h
    
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])
