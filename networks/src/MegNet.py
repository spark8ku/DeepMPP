import torch.nn as nn
import torch
import dgl
from dgl.nn import SumPooling

class MegNet_layer(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, in_graph_feats, 
                 out_node_feats=None, out_edge_feats=None, out_graph_feats=None, 
                  activation=nn.LeakyReLU(), dropout=0.2, residual_sum = True):
        super(MegNet_layer, self).__init__()
        if out_node_feats is None:
            out_node_feats = in_node_feats
        if out_edge_feats is None:
            out_edge_feats = in_edge_feats
        if out_graph_feats is None:
            out_graph_feats = in_graph_feats

        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.residual_sum = residual_sum

        self.node_linear = nn.Sequential(nn.Linear(in_node_feats, out_node_feats), self.activation, self.dropout)
        self.edge_linear = nn.Sequential(nn.Linear(in_edge_feats, out_edge_feats), self.activation, self.dropout)
        self.graph_linear = nn.Sequential(nn.Linear(in_graph_feats, out_graph_feats), self.activation, self.dropout)

        self.edge_update_linear = nn.Sequential(nn.Linear(out_node_feats*2+out_edge_feats+out_node_feats, out_edge_feats), self.activation, self.dropout)
        self.node_update_linear = nn.Sequential(nn.Linear(out_node_feats+out_edge_feats+out_graph_feats, out_node_feats), self.activation, self.dropout)
        self.graph_update_linear = nn.Sequential(nn.Linear(out_node_feats+out_edge_feats+out_graph_feats, out_graph_feats), self.activation, self.dropout)

        if residual_sum:
            if in_node_feats!=out_node_feats:
                self.residual_layer_node = nn.Linear(in_node_feats, out_node_feats)
            if in_edge_feats!=out_edge_feats:
                self.residual_layer_edge = nn.Linear(in_edge_feats, out_edge_feats)
            if in_graph_feats!=out_graph_feats:
                self.residual_layer_graph = nn.Linear(in_graph_feats, out_graph_feats)

    def edge_update_func(self, edges):
        return {'f': self.edge_update_linear( torch.cat([edges.src['h'], edges.dst['h'], edges.data['f'],edges.dst['u']], dim=1) )}

    def message_func(self, edges):
        return {'mail': edges.data['f']}
    
    def update_func(self, nodes):
        return {'hf': torch.mean(nodes.mailbox['mail'], 1)}

    def forward(self, graph, node_feats, edge_feats, graph_feats):
        with graph.local_scope():
            
            graph.ndata['h'] = self.node_linear(node_feats)
            graph.edata['f'] = self.edge_linear(edge_feats)
            graph.ndata['u'] = self.graph_linear(graph_feats)

            graph.apply_edges(self.edge_update_func)
            graph.update_all(self.message_func, self.update_func)
            edge_feats_new = graph.edata['f']
            
            node_feats_new = self.node_update_linear(torch.cat([graph.ndata['h'],graph.ndata['hf'],graph.ndata['u']],dim=1))

            node_pooling, edge_pooling, graph_pooling = Megnet_pooling(graph, node_feats_new, edge_feats_new, graph_feats)
            
            graph_feats_new = self.node_update_linear(torch.cat([node_pooling,edge_pooling,graph_pooling],dim=1))
            expanded_indices = torch.repeat_interleave( torch.arange(graph_feats_new.shape[0]).to(device=graph_feats_new.device), graph.batch_num_nodes().int() )                                                   
            graph_feats_new = graph_feats_new[expanded_indices]

            if self.residual_sum:
                if node_feats.shape[1]!=node_feats_new.shape[1]:
                    node_feats = self.residual_layer_node(node_feats)
                node_feats_new = node_feats_new + node_feats

                if edge_feats.shape[1]!=edge_feats_new.shape[1]:
                    edge_feats = self.residual_laye_edge(edge_feats)
                edge_feats_new = edge_feats_new + edge_feats

                if graph_feats.shape[1]!=graph_feats_new.shape[1]:
                    graph_feats = self.residual_layer_graph(graph_feats)
                graph_feats_new = graph_feats_new + graph_feats
        return node_feats_new, edge_feats_new, graph_feats_new
    
class MegNets(nn.Module):
    def __init__(self,  in_node_feats, in_edge_feats, in_graph_feats, 
                 hidden_node_feats,  hidden_edge_feats,  hidden_graph_feats, 
                 out_node_feats, out_edge_feats, out_graph_feats,
                 activation = nn.LeakyReLU, n_layers=4, dropout=0.1, residual_sum = True):
        super(MegNets, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i==0:
                in_node_feats = in_node_feats
                in_edge_feats = in_edge_feats
                in_graph_feats = in_graph_feats
            else:
                in_node_feats = hidden_node_feats
                in_edge_feats = hidden_edge_feats
                in_graph_feats = hidden_graph_feats
            if i==n_layers-1:
                out_node_feats = out_node_feats
                out_edge_feats = out_edge_feats
                out_graph_feats = out_graph_feats
            else:
                out_node_feats = hidden_node_feats
                out_edge_feats = hidden_edge_feats
                out_graph_feats = hidden_graph_feats
            self.layers.append(MegNet_layer(in_node_feats, in_edge_feats, in_graph_feats, 
                 out_node_feats=out_node_feats, out_edge_feats=out_edge_feats, out_graph_feats=out_graph_feats, 
                  activation=activation, dropout=dropout, residual_sum = residual_sum))
        
    def forward(self, graph, node_feats, edge_feats, graph_feats, **kwargs):
        for layer in self.layers:
            node_feats, edge_feats, graph_feats = layer(graph, node_feats, edge_feats, graph_feats)
        return node_feats, edge_feats, graph_feats
    
def Megnet_pooling(graph, node_feats, edge_feats, graph_feats, node_pooling = SumPooling(), edge_pooling = SumPooling()):
    
    def message_func(edges):
        return {'mail': edges.data['f']}
    
    def sum_edges(edges):
        return {'edges': torch.sum(edges.mailbox['mail'], 1)}

    with graph.local_scope():
        graph.edata['f'] = edge_feats
        graph.update_all(message_func, sum_edges)
        edge_feats = graph.ndata['edges']
    node_pooling = node_pooling(graph,node_feats)/graph.batch_num_nodes().view(-1,1)
    edge_pooling = edge_pooling(graph,edge_feats)/graph.batch_num_edges().view(-1,1)
    graph_pooling = SumPooling()(graph,graph_feats)/graph.batch_num_nodes().view(-1,1)
    return node_pooling, edge_pooling, graph_pooling