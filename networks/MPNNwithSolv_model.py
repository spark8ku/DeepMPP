import torch.nn as nn
import torch
from dgl.nn import SumPooling

from networks.src.GCN import GCNs
from networks.src.MPNN import MPNNs
from networks.src.Linear import Linears
from networks.src.SolventLayer import SolventLayer

class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()
        
        hidden_dim = config.get('hidden_dim', 64)
        mpnn_layers = config.get('conv_layers', 4)
        dropout = config.get('dropout', 0.2)
        target_dim = config['target_dim']

        self.node_embedding = nn.Linear(config['node_dim'], hidden_dim)
        self.edge_embedding = nn.Linear(config['edge_dim'], hidden_dim)
        self.MPNNs = MPNNs(hidden_dim, hidden_dim, hidden_dim, hidden_dim, nn.ReLU(), mpnn_layers, dropout, False, True) # in_feats, hidden_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        self.Linears = Linears(hidden_dim,hidden_dim, nn.ReLU(), 2, dropout, False, False) # in_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        self.solvlayer = SolventLayer(config)


    def forward(self, graph, node_feats, edge_feats, solv_graph, solv_node_feats, **kwargs):
        h = self.node_embedding(node_feats)
        e = self.edge_embedding(edge_feats)
        h = self.MPNNs(graph, h, e)
        h = SumPooling()(graph,h)
        h = self.Linears(h)
        h = self.solvlayer(h, solv_graph, solv_node_feats)
        return h
    
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])
