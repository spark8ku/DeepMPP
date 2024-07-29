import torch.nn as nn
import torch
import dgl


class MPNN_layer(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_feats, activation, dropout=0.2, batch_norm=False, residual_sum = False):
        super(MPNN_layer, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.linear = nn.Linear(in_node_feats*2+in_edge_feats, out_feats)
        self.residual_sum = residual_sum
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_feats)
        if residual_sum:
            if in_node_feats!=out_feats:
                self.residual_layer = nn.Linear(in_node_feats, out_feats)

    def message_func(self, edges):
        return {'mail': self.linear( torch.cat([edges.src['h'], edges.dst['h'], edges.data['f']], dim=1) )}
    
    def update_func(self, nodes):
        return {'h': torch.mean(nodes.mailbox['mail'], 1)}


    def forward(self, graph, node_feats, edge_feats):
        with graph.local_scope():
            
            graph.ndata['h'] = node_feats
            graph.edata['f'] = edge_feats
            graph.update_all(self.message_func, self.update_func)

            h = graph.ndata['h']
            if self.batch_norm:
                h = self.bn(h)
            h = self.activation(h)
            h = self.dropout(h)
            if self.residual_sum:
                if node_feats.shape[1]!=h.shape[1]:
                    node_feats = self.residual_layer(node_feats)
                h = h + node_feats
        return h
    
class MPNNs(nn.Module):
    def __init__(self,  in_node_feats, in_edge_feats, hidden_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        super(MPNNs, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i==0:
                _in_feats = in_node_feats
            else:
                _in_feats = hidden_feats
            if i==n_layers-1:
                _out_feats = out_feats
            else:
                _out_feats = hidden_feats
            self.layers.append(MPNN_layer(_in_feats, in_edge_feats, _out_feats, activation, dropout, batch_norm, residual_sum))
        
    def forward(self, graph, node_feats, edge_feats):
        for layer in self.layers:
            node_feats = layer(graph, node_feats, edge_feats)
        return node_feats