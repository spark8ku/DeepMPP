import torch.nn as nn
import torch
from dgl.nn import Set2Set

from networks.src.MegNet import MegNets, Megnet_pooling
from networks.src.Linear import Linears

class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()
        
        hidden_dim = config.get('hidden_dim', 64)
        conv_layers = config.get('conv_layers', 4)
        dropout = config.get('dropout', 0.2)
        linear_layers = config.get('linear_layers', 2)
        target_dim = config['target_dim']
        residual_sum = config.get('residual_sum', True)

        self.node_embedding = nn.Linear(config['node_dim'], hidden_dim)
        self.edge_embedding = nn.Linear(config['edge_dim'], hidden_dim)
        self.graph_embedding = nn.Linear(config['graph_dim'], hidden_dim)

        in_node_feats = hidden_dim
        in_edge_feats = hidden_dim
        in_graph_feats = hidden_dim

        hidden_node_feats = hidden_dim
        hidden_edge_feats = hidden_dim
        hidden_graph_feats = hidden_dim

        out_node_feats = hidden_dim
        out_edge_feats = hidden_dim
        out_graph_feats = hidden_dim



        self.MegNets = MegNets(in_node_feats, in_edge_feats, in_graph_feats, 
                                hidden_node_feats, hidden_edge_feats, hidden_graph_feats,
                                out_node_feats, out_edge_feats, out_graph_feats,                              
                                nn.LeakyReLU(), conv_layers, dropout, residual_sum) #
        self.Linears = Linears(out_node_feats*2+out_edge_feats*2+out_graph_feats, target_dim, nn.LeakyReLU(), linear_layers, dropout, False, False, last=True) 
        self.node_set2set = Set2Set(out_node_feats,2,1)
        self.edge_set2set = Set2Set(out_edge_feats,2,1)

    def forward(self, graph, node_feats, edge_feats, graph_feats, **kwargs):
        h = self.node_embedding(node_feats)
        e = self.edge_embedding(edge_feats)
        u = self.graph_embedding(graph_feats)

        h, e, u = self.MegNets(graph, h, e,u)
        h, e, u = Megnet_pooling(graph, h, e, u, self.node_set2set, self.edge_set2set)
        h = torch.cat([h,e,u],axis=1)
        h = self.Linears(h)
        return h
    
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])
