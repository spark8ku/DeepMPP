import torch
import torch.nn as nn


from networks.src.AGC import AGC
from networks.src.Linear import Linears

class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()

        hidden_dim = config.get('hidden_dim', 64)
        gcn_layers = config.get('conv_layers', 4)
        dropout = config.get('dropout', 0.1)
        linear_layers = config.get('linear_layers', 4)
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
        self.AGC = AGC(hidden_dim, hidden_dim, nn.LeakyReLU(), gcn_layers,dropout, False, True, 1)
        
        self.linears = Linears(hidden_dim, target_dim, nn.ReLU(), linear_layers, dropout, False, False, last=True)
    
    def forward(self, graph, r_node, **kargs):
        r_node = r_node.float()
        r_node = self.embedding_rnode_lin(r_node)
        
        real_graph=graph.node_type_subgraph(['r_nd'])
        real_graph.set_batch_num_nodes(graph.batch_num_nodes('r_nd'))
        
        h, score = self.AGC(graph, r_node)
        
        h = self.linears(h)

        if kargs.get('get_score',False):
            return {'prediction':h, 'positive':score}

        return h
        
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])
