import torch
import torch.nn as nn

import dgl
from dgl.nn import SumPooling

import matplotlib.pyplot as plt
from networks.src.GC import GCconvolution
from networks.src.Linear import Linears

class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()

        
        hidden_dim = config.get('hidden_dim', 64)
        gcn_layers = config.get('conv_layers', 4)
        dropout = config.get('dropout', 0.2)
        linear_layers = config.get('linear_layers', 2)
        last_linear_dim = int(hidden_dim/(2**linear_layers))
        target_dim = config['target_dim']

        self.embedding_rnode_lin = nn.Sequential(
            nn.Linear(config['node_dim'], hidden_dim, bias=False)
        )
        self.embedding_inode_lin = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False)
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(config['edge_dim'], hidden_dim, bias=False)
        )

        self.GCconv = GCconvolution(hidden_dim, hidden_dim, hidden_dim, nn.LeakyReLU(), gcn_layers, dropout, False, True, 0.1)
        
        self.reduce = SumPooling()
        self.linear = nn.Linear(1,1)
        self.normalize = nn.LayerNorm(1)
        self.func_f = getattr(nn, config.get('func_f', 'Tanh'))()
    
    def forward(self, graph, r_node, i_node, r_edge, d_edge,**kargs):
        r_node = r_node.float()
        r_node = self.embedding_rnode_lin(r_node)
        i_node = i_node.float()
        i_node = self.embedding_inode_lin(i_node)
        r_edge = r_edge.float()
        r_edge = self.embedding_edge_lin(r_edge)
        
        dot_graph=graph.node_type_subgraph(['d_nd'])
        dot_graph.set_batch_num_nodes(graph.batch_num_nodes('d_nd'))
        dot_graph.set_batch_num_edges(graph.batch_num_edges('d2d'))
        
        score = self.GCconv(graph, r_node, r_edge, i_node, d_edge)
        
        h = self.reduce(dot_graph, score)
        h = self.normalize(h)
        h = self.func_f(h)
        h = self.linear(h)
        
        if kargs.get('get_score',False):
            return {'prediction':h, 'positive':score}

        return h
        
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])
