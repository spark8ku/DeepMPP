import torch.nn as nn
import torch
from dgl.nn import AvgPooling

from networks.src.GCN import GCNs
from networks.src.ALIGNN import ALIGNNs
from networks.src.Linear import Linears

class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()
        
        hidden_dim = config.get('hidden_dim', 64)
        ALLIGN_conv_layers = config.get('conv_layers', 4)
        GCN_conv_layers = config.get('GCN_conv_layers', 2)
        dropout = config.get('dropout', 0.2)
        linear_layers = config.get('linear_layers', 2)
        target_dim = config['target_dim']
        residual_sum = config.get('residual_sum', True)

        self.node_embedding = nn.Linear(config['node_dim'], hidden_dim)
        self.edge_embedding = nn.Linear(config['edge_dim'], hidden_dim)
        self.triplet_embedding = nn.Linear(config['triplet_dim'], hidden_dim)

        in_node_feats = hidden_dim
        in_edge_feats = hidden_dim
        in_triplet_feats = hidden_dim

        hidden_node_feats = hidden_dim
        hidden_edge_feats = hidden_dim
        hidden_triplet_feats = hidden_dim

        out_node_feats = hidden_dim
        out_edge_feats = hidden_dim
        out_triplet_feats = hidden_dim

        self.ALIGNNs = ALIGNNs(in_node_feats, in_edge_feats, in_triplet_feats, 
                                hidden_node_feats, hidden_edge_feats, hidden_triplet_feats,
                                out_node_feats, out_edge_feats, out_triplet_feats,                              
                                nn.LeakyReLU(), ALLIGN_conv_layers, dropout, residual_sum) #
        self.GCNs = GCNs(out_node_feats, out_node_feats, out_node_feats, nn.LeakyReLU(), GCN_conv_layers, dropout, False, False)
        self.Linears = Linears(out_node_feats, target_dim, nn.LeakyReLU(), linear_layers, dropout, False, False, last=True) 

    def forward(self, graph, node_feats, edge_feats, triplet_feats, **kwargs):
        h = self.node_embedding(node_feats)
        e = self.edge_embedding(edge_feats)
        t = self.triplet_embedding(triplet_feats)

        edge_graph=graph.node_type_subgraph(['atom'])
        edge_graph.set_batch_num_nodes(graph.batch_num_nodes('atom'))
        edge_graph.set_batch_num_edges(graph.batch_num_edges('bond'))
        
        line_graph=graph.node_type_subgraph(['bond'])
        line_graph.set_batch_num_nodes(graph.batch_num_nodes('bond'))
        line_graph.set_batch_num_edges(graph.batch_num_edges('triplet'))

        h = self.node_embedding(node_feats)
        e = self.edge_embedding(edge_feats)
        t = self.triplet_embedding(triplet_feats)

        h = self.ALIGNNs(edge_graph,line_graph, h, e, t)
        h = self.GCNs(edge_graph, h)
        h = AvgPooling()(edge_graph, h)
        h = self.Linears(h)
        return h
    
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])
