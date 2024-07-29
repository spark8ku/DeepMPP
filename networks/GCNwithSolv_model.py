import torch.nn as nn
import torch
from dgl.nn import SumPooling

from networks.src.GCN import GCNs
from networks.src.Linear import Linears

class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()
        
        hidden_dim = config.get('hidden_dim', 64)
        conv_layers = config.get('conv_layers', 4)
        linear_layers = config.get('linear_layers', 2)
        self.node_embedding = nn.Linear(config['node_dim'], hidden_dim)
        self.node_embedding_solv = nn.Linear(config['node_dim'], 64)

        self.GCNs = GCNs(hidden_dim, hidden_dim, hidden_dim, nn.ReLU(), conv_layers, 0.2, False, True) # in_feats, hidden_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        self.GCNs_solv = GCNs(64, 64, 64, nn.ReLU(), 4, 0.2, False, True)
        self.Linears1 = Linears(hidden_dim,hidden_dim, nn.ReLU(), 2, 0.2, False, False) # in_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        self.Linears2 = Linears(64,64, nn.ReLU(), 2, 0.2, False, False) # in_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        self.Linears3 = Linears(hidden_dim+64,16, nn.ReLU(), linear_layers, 0.2, False, False) # in_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False

        self.output = nn.Linear(16, config['target_dim'])

    def forward(self, graph, node_feats,solv_graph,solv_node_feats,**kwargs):
        h = self.node_embedding(node_feats)
        h = self.GCNs(graph, h)
        h = SumPooling()(graph,h)

        h_solv = self.node_embedding_solv(solv_node_feats)
        h_solv = self.GCNs_solv(solv_graph, h_solv)
        h_solv = SumPooling()(solv_graph,h_solv)

        h = self.Linears1(h)
        h_solv = self.Linears2(h_solv)
        h = torch.cat([h,h_solv],axis=1)
        h = self.Linears3(h)
        h = self.output(h)
        return h
    
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])
