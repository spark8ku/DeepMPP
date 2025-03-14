import torch.nn as nn
import torch

from D4CMPP.networks.src.AFP import  AttentiveFP
from D4CMPP.networks.src.Linear import Linears
from D4CMPP.networks.src.MPNN import MPNNs
from dgl.nn import SumPooling

class network(nn.Module):
    def __init__(self, config):
        super().__init__()
        config['dropout']=config.get('dropout', 0.2)
        config['hidden_dim']=config.get('hidden_dim', 64)

        conv_layers = config.get('conv_layers', 4)
        linear_layers = config.get('linear_layers', 2)
        dropout = config.get('dropout', 0.2)

        self.embedding_node_lin = nn.Linear(config['node_dim'], config['hidden_dim'], bias=True)
        self.embedding_edge_lin = nn.Linear(config['edge_dim'], config['hidden_dim'], bias=True)

        self.AttentiveFP = AttentiveFP(config)
        self.Linears = Linears(config['hidden_dim'],config['target_dim'], nn.ReLU(), linear_layers, dropout, False, False, True) 
        

    def forward(self, graph, node_feats, edge_feats, **kwargs):
        node = self.embedding_node_lin(node_feats)
        edge = self.embedding_edge_lin(edge_feats)

        super_node, att_w = self.AttentiveFP(graph, node, edge)

        output = self.Linears(super_node)
        
        if kwargs.get('get_score',False):
            return {'prediction':output, 'positive':att_w}
        
        return output

        
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])
