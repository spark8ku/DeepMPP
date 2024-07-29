
import dgl
import torch
import torch.nn as nn

from networks.src.Linear import Linears
from networks.src.DMPNN import DMPNNLayer


class network(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.get('hidden_dim', 64)
        conv_layers = config.get('conv_layers', 4)
        dropout = config.get('dropout', 0.2)
        linear_layers = config.get('linear_layers', 2)
        target_dim = config['target_dim']

        self.embedding_node_lin = nn.Linear(config['node_dim'], hidden_dim, bias=True)
        self.embedding_edge_lin = nn.Linear(config['edge_dim'], hidden_dim, bias=True)
        self.init_h_func = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU()
        )
        self.W_a = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU()
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.DMPNNLayer = nn.ModuleList([DMPNNLayer(hidden_dim,hidden_dim,nn.LeakyReLU(),0.2) for _ in range(conv_layers)])
        self.Linears = Linears(hidden_dim, target_dim, nn.LeakyReLU(), linear_layers, dropout, last = True)
        
    def send_income_edge(self, edges):
        return {'mail': edges.data['feat']}

    def sum_income_edge(self, nodes):
        hidden_feats = self.W_a(torch.cat([nodes.data['feat'], torch.sum(nodes.mailbox['mail'], 1)], dim=-1))
        hidden_feats = self.dropout_layer(hidden_feats)
        return {'hidden_feats': hidden_feats}

    def forward(self, graph, node_feats, edge_feats, **kwargs):
        node = self.embedding_node_lin(node_feats)
        edge = self.embedding_edge_lin(edge_feats)
        src_node_id = graph.edges()[0]
        hidden_feats = self.init_h_func(torch.cat([node[src_node_id], edge], dim=-1)) # hidden states include the source node and edge features, num_bonds x num_features
        inputs = hidden_feats
        for layer in self.DMPNNLayer:
            hidden_feats = layer(graph, node, inputs, hidden_feats)

        graph.edata['feat'] = hidden_feats
        graph.ndata['feat'] = node
        graph.update_all(self.send_income_edge, self.sum_income_edge)
        graph_feats = dgl.sum_nodes(graph, 'hidden_feats')
        output = self.Linears(graph_feats)
        return output
    
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])


