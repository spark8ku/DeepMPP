import torch.nn as nn
import torch
from dgl.nn import Set2Set

from D4CMPP.networks.src.MegNet import MegNets, Megnet_pooling
from D4CMPP.networks.src.Linear import Linears

class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()
        
        n1 = config.get('n1', 64)
        n2 = config.get('n2', 32)
        n3 = config.get('n3', 16)
        conv_layers = config.get('conv_layers', 3)
        conv_dropout = config.get('conv_dropout', 0.1)
        pred_dropout = config.get('pred_dropout', 0.2)
        linear_layers = config.get('linear_layers', 2)
        target_dim = config['target_dim']

        self.node_embedding = nn.Linear(config['node_dim'], n2)
        self.edge_embedding = nn.Linear(config['edge_dim'], n2)
        self.graph_embedding = nn.Linear(config['graph_dim'], n2)



        self.MegNets = MegNets(in_dim = n2,
                                 hidden_dims = [n1,n1],
                                activation=nn.LeakyReLU(), 
                                n_layers=conv_layers, 
                                dropout=conv_dropout)
        
        self.node_linear_last = nn.Linear(n2, n3)
        self.edge_linear_last = nn.Linear(n2, n3)
    
        self.node_set2set = Set2Set(n3,3,2)
        self.edge_set2set = Set2Set(n3,3,2)

        self.Linears = Linears(n3*2+n3*2, target_dim, nn.LeakyReLU(), linear_layers, pred_dropout, False, False, last=True) 

    def forward(self, graph, node_feats, edge_feats, graph_feats, **kwargs):
        h = self.node_embedding(node_feats)
        e = self.edge_embedding(edge_feats)
        u = self.graph_embedding(graph_feats)

        h, e, u = self.MegNets(graph, h, e,u)

        h = self.node_linear_last(h)
        e = self.edge_linear_last(e)

        h, e, u = Megnet_pooling(graph, h, e, u, self.node_set2set, self.edge_set2set)
        h = torch.cat([h,e],axis=1)
        h = self.Linears(h)
        return h
    
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])
