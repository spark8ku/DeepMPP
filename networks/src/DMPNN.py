
"""
This codes are modified from the project "GC-GNN" (https://github.com/gsi-lab/GC-GNN)
The original codes are under the MIT License. (https://github.com/gsi-lab/GC-GNN/blob/main/networks/DMPNN.py)
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class DMPNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout=0.2):
        super(DMPNNLayer, self).__init__()
        self.activation = activation

        self.W_m = nn.Linear(in_feats, out_feats, bias=True)
        self.dropout_layer = nn.Dropout(dropout)

    def message_func(self, edges):
        return {'edge_feats': edges.data['edge_feats']}

    def reducer_sum(self, nodes):
        return {'full_feats': torch.sum(nodes.mailbox['edge_feats'], 1)}

    def edge_func(self, edges):
        return {'new_edge_feats': edges.dst['full_feats'] - edges.data['edge_feats']}

    def forward(self, graph, node, inputs,hidden_feats):
        with graph.local_scope():
            graph.ndata['node_feats'] = node
            graph.edata['edge_feats'] = hidden_feats
            graph.update_all(self.message_func, self.reducer_sum)
            graph.apply_edges(self.edge_func)
            hidden_feats = F.relu(inputs + self.W_m(graph.edata['new_edge_feats']))
            hidden_feats = self.dropout_layer(hidden_feats)
            return hidden_feats

