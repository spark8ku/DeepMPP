import torch.nn as nn
import torch

import dgl
from dgl.nn.functional import edge_softmax
from dgl.nn import AvgPooling

class EdgeGatedGCN(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats,
                    out_node_feats, out_edge_feats, 
                    activation=nn.SiLU(), dropout=0.2, batch_norm=True, residual_sum = True):
        super(EdgeGatedGCN, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.residual = residual_sum

        self.upd_linear = nn.Linear(in_node_feats*2+in_edge_feats, out_edge_feats)
        if residual_sum:
            if in_edge_feats!=out_edge_feats:
                self.residual_edge_layer = nn.Linear(in_edge_feats, out_edge_feats)
        self.upd_norm = nn.BatchNorm1d(in_edge_feats) if batch_norm else nn.Identity()
        self.activation_linear = nn.Linear(in_edge_feats, in_node_feats)

        self.gating_dst_linear = nn.Linear(in_node_feats, in_node_feats)
        self.gating_src_linear = nn.Linear(in_node_feats, in_node_feats)
        self.gating_norm = nn.BatchNorm1d(in_node_feats) if batch_norm else nn.Identity()

        self.linear = nn.Linear(in_node_feats, out_node_feats)

    def update_func(self, edges):
        e = self.upd_linear( torch.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=1))
        e = self.upd_norm(e)
        e2 = self.activation(e)
        if self.residual:
            if edges.data['e'].shape[1]!=e2.shape[1]:
                e2 = e2 + self.residual_edge_layer(edges.data['e'])
            else:
                e2 = edges.data['e'] + e2
        return {'e2': e2}
    
    def gating_message_func(self, edges):
        return {'mail': edges.data['a']*self.gating_dst_linear(edges.dst['h'])}

    def gating_update_func(self, nodes):
        h = self.gating_src_linear(nodes.data['h'])+torch.sum(nodes.mailbox['mail'],1)
        return {'h2': h}

    def forward(self, graph, h, e):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.edata['e'] = e
            graph.apply_edges(self.update_func)
            a = self.activation_linear(graph.edata['e2'])
            graph.edata['a'] = edge_softmax(graph, a)
            e2 = graph.edata['e2']

            graph.update_all(self.gating_message_func, self.gating_update_func)
            
            h2 = graph.ndata['h2']
            h2 = self.gating_norm(h2)
            h2 = self.activation(h2)
            h2 = h2 + h
            h2 = self.linear(h2)
        return h2, e2
    
class ALIGNNs(nn.Module):
    def __init__(self,  in_node_feats, in_edge_feats, in_triplet_feats, 
                 hidden_node_feats, hidden_edge_feats, hidden_triplet_feats,  
                 out_node_feats, out_edge_feats, out_triplet_feats, 
                 activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        super(ALIGNNs, self).__init__()

        self.lineEG = nn.ModuleList()
        self.edgeEG = nn.ModuleList()
        for i in range(n_layers):
            if i==0:
                _in_node_feats = in_node_feats
                _in_edge_feats = in_edge_feats
                _in_triplet_feats = in_triplet_feats
            else:
                _in_node_feats = hidden_node_feats
                _in_edge_feats = hidden_edge_feats
                _in_triplet_feats = hidden_triplet_feats
            if i==n_layers-1:
                _out_node_feats = out_node_feats
                _out_edge_feats = out_edge_feats
                _out_triplet_feats = out_triplet_feats
            else:
                _out_node_feats = hidden_node_feats
                _out_edge_feats = hidden_edge_feats
                _out_triplet_feats = hidden_triplet_feats
            self.lineEG.append(EdgeGatedGCN(_in_edge_feats, _in_triplet_feats, 
                                            _out_edge_feats, _out_triplet_feats, 
                                            activation, dropout, batch_norm, residual_sum))
            self.edgeEG.append(EdgeGatedGCN(_in_node_feats, _in_edge_feats,
                                            _out_node_feats, _out_edge_feats, 
                                            activation, dropout, batch_norm, residual_sum))
        
    def forward(self, edge_graph, line_graph, node_feats, edge_feats, triplet_feats):

        for edge_layer, line_layer in zip(self.edgeEG, self.lineEG):
            # edge_feats = self.tight_edge_features(edge_feats, graph.batch_num_edges('bond'))
            edge_feats, triplet_feats = line_layer(line_graph, edge_feats, triplet_feats)
            # edge_feats = self.restore_edge_features(edge_feats, graph.batch_num_edges('bond'))
            node_feats, edge_feats = edge_layer(edge_graph, node_feats, edge_feats)
        # h = AvgPooling()(edge_graph, node_feats)
        return node_feats
    
    # def restore_edge_features(self, edge_feats, edge_batch):
    #     edges=[]
    #     accum_i=0
    #     for i in range(len(edge_batch)):
    #         btch_num = edge_batch[i]//2
    #         edges.append(edge_feats[accum_i:accum_i+btch_num])
    #         edges.append(edge_feats[accum_i:accum_i+btch_num])
    #         accum_i+= btch_num
    #     return torch.cat(edges, 0)
    
    # def tight_edge_features(self, edge_feats, edge_batch):
    #     edges=[]
    #     accum_i=0
    #     for i in range(len(edge_batch)):
    #         edges.append(edge_feats[accum_i:accum_i+edge_batch[i]//2])
    #         accum_i+=edge_batch[i]
    #     return torch.cat(edges, 0)