import torch.nn as nn
import torch
import dgl
from dgl.nn.functional import edge_softmax

class GAT_layer(nn.Module):
    def __init__(self, in_node_feats, hidden_feats, out_feats, activation, dropout=0.2, batch_norm=False, residual_sum = False):
        super(GAT_layer, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.attention_W = nn.Linear(in_node_feats*2, hidden_feats)
        self.attention_a = nn.Linear(hidden_feats,1, bias=False)

        self.linear = nn.Linear(in_node_feats, out_feats)

        self.batch_norm = batch_norm
        self.residual_sum = residual_sum
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_feats)
        if residual_sum:
            if in_node_feats!=out_feats:
                self.residual_layer = nn.Linear(in_node_feats, out_feats)

    def calc_val(self, edges):
        return {'val': self.attention_a(
                            nn.LeakyReLU()(
                            self.attention_W( torch.cat([edges.src['h'], edges.dst['h']], dim=1) 
                )))}
    
    def message_func(self, edges):
        return {'score': edges.data['val'], 'h': edges.src['h']}
    
    def update_func(self, nodes):
        return {'h': nn.LeakyReLU()(torch.sum(nodes.mailbox['score']*self.linear(nodes.mailbox['h']), 1))}


    def forward(self, graph, node_feats):
        with graph.local_scope():
            
            graph.ndata['h'] = node_feats
            graph.apply_edges(self.calc_val)
            graph.edata['val'] = edge_softmax(graph, graph.edata['val'])
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
    
class GATs(nn.Module):
    def __init__(self,  in_node_feats, hidden_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        super(GATs, self).__init__()

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
            self.layers.append(GAT_layer(_in_feats, hidden_feats, _out_feats, activation, dropout, batch_norm, residual_sum))
        
    def forward(self, graph, node_feats):
        for layer in self.layers:
            node_feats = layer(graph, node_feats)
        return node_feats