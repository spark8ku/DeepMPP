import torch
import torch.nn as nn
import dgl

class distGCN_layer(nn.Module):
    def __init__(self, in_feats, edge_feats, out_feats, activation, alpha = 0.1):
        super().__init__()
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats)
        self.edge_linear = nn.Linear(edge_feats, out_feats)
        self.alpha = alpha

    def forward(self, graph, node_feats, edge_feats):
        if len(edge_feats)==0:
            return node_feats
        with graph.local_scope():
            graph.ndata['h'] = self.linear(node_feats)
            graph.edata['f'] = self.edge_linear(edge_feats)
            graph.update_all(dgl.function.u_mul_e('h', 'f', 'm'),
                            dgl.function.sum('m', 'h'))
            h = graph.ndata['h']
            h = self.activation(h)
            h = node_feats + self.alpha * h
        return h
    

# class distMPNN_layer(nn.Module):
#     def __init__(self, in_feats, edge_feats, out_feats, activation, alpha = 0.1):
#         super().__init__()
#         self.activation = activation
#         self.in_feats = in_feats
#         self.node_linear = nn.Linear(in_feats, out_feats)
#         self.edge_linear = nn.Linear(edge_feats, out_feats)
#         self.linear_mpnn = nn.Linear(3*out_feats, edge_feats*out_feats)
#         self.linear_mpnn2 = nn.Linear(edge_feats*out_feats, out_feats)

#         self.alpha = alpha

#     def message_func(self, edges):
#         message = self.linear_mpnn( torch.cat([edges.src['h'], edges.dst['h'], edges.data['f']], dim=1))
#         message = self.linear_mpnn2( edges.data['d'] * message)
#         return {'mail':  message}
#     def update_func(self, nodes):
#         return {'h': torch.mean(nodes.mailbox['mail'], 1)}

#     def forward(self, graph, node_feats, edge_feats):
#         if len(edge_feats)==0:
#             return node_feats
#         with graph.local_scope():
#             graph.ndata['h'] = self.node_linear(node_feats)
#             graph.edata['f'] = self.edge_linear(edge_feats)
#             graph.edata['d'] = torch.tile(edge_feats, (1,self.in_feats))
#             graph.update_all(self.message_func, self.update_func)
#             h = graph.ndata['h']
#             h = self.activation(h)
#             h = node_feats + self.alpha * h
#         return h
    
class distMPNN_layer(nn.Module):
    def __init__(self, in_feats, edge_feats, out_feats, activation, alpha = 0.1):
        super().__init__()
        self.activation = activation
        self.in_feats = in_feats
        self.node_linear = nn.Linear(in_feats, out_feats)
        self.edge_linear = nn.Linear(edge_feats, out_feats)
        self.linear_mpnn = nn.Linear(3*out_feats, edge_feats*edge_feats)
        self.linear_mpnn2 = nn.Linear(edge_feats*out_feats, out_feats)
        self.linear_mpnn3 = nn.Linear(out_feats*2, out_feats)
        self.alpha = alpha

    def message_func(self, edges):
        message = self.linear_mpnn( torch.cat([edges.src['h'], edges.dst['h'], edges.data['f']], dim=1))
        message = self.linear_mpnn2( edges.data['d'] * message)
        return {'mail':  message}
    def update_func(self, nodes):
        return {'h': torch.mean(nodes.mailbox['mail'], 1)}

    def forward(self, graph, node_feats, edge_feats):
        if len(edge_feats)==0:
            return node_feats
        with graph.local_scope():
            node_feats = self.node_linear(node_feats)
            graph.ndata['h'] = node_feats
            graph.edata['f'] = self.edge_linear(edge_feats)
            graph.edata['d'] = torch.tile(edge_feats, (1,self.edge_feats))
            graph.update_all(self.message_func, self.update_func)
            h = graph.ndata['h']
            h = self.activation(h)
            h = self.linear_mpnn3( torch.cat([node_feats, h], dim=1))
            h = node_feats + h
        return h