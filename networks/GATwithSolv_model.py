import torch.nn as nn
import torch
from dgl.nn import SumPooling

from networks.src.GAT import GATs
from networks.src.SolventLayer import SolventLayer

class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()
        
        hidden_dim = config.get('hidden_dim', 64)
        gcn_layers = config.get('conv_layers', 4)
        dropout = config.get('dropout', 0.2)
        linear_layers = config.get('linear_layers', 2)
        last_linear_dim = int(hidden_dim/(2**linear_layers))
        target_dim = config['target_dim']


        self.node_embedding = nn.Linear(config['node_dim'], hidden_dim)

        self.GATs = GATs(hidden_dim, hidden_dim, hidden_dim, nn.ReLU(), gcn_layers, dropout, False, True)
        self.SolventLayer = SolventLayer(config)

    def forward(self, graph, node_feats, solv_graph, solv_node_feats, **kwargs):
        h = self.node_embedding(node_feats)
        h = self.GATs(graph, h)
        h = SumPooling()(graph,h)
        h = self.SolventLayer(h,solv_graph,solv_node_feats)
        return h
    
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])
