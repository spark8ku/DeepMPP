import torch.nn as nn
import torch
import dgl
from dgl.nn import SumPooling

class MegNet_layer(nn.Module):
    def __init__(self, 
                 in_dim = 32, 
                 hidden_dims = [64,64], 
                 activation=nn.LeakyReLU(), 
                 dropout=0.1, ):
        super(MegNet_layer, self).__init__()
        
        if type(hidden_dims) is int:
            hidden_dims = [hidden_dims]
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        
        self.ff_node = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, hidden_dims[0]),
            nn.LeakyReLU())        
        self.ff_edge = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, hidden_dims[0]),
            nn.LeakyReLU())
        self.ff_graph = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, hidden_dims[0]),
            nn.LeakyReLU())    

        edge_update_linear = []
        node_update_linear = []
        graph_update_linear = []
        
        hidden_dims = hidden_dims + [in_dim]
        #V + V + E + U
        edge_update_linear.append(nn.Linear(hidden_dims[0]*2 + hidden_dims[0] + hidden_dims[0], hidden_dims[1]))
        edge_update_linear.append(self.activation)
        #V + E + U
        node_update_linear.append(nn.Linear(hidden_dims[0] + hidden_dims[-1] + hidden_dims[0], hidden_dims[1]))
        node_update_linear.append(self.activation)
        #V + E + U
        graph_update_linear.append(nn.Linear(hidden_dims[-1] + hidden_dims[-1] + hidden_dims[0], hidden_dims[1]))
        graph_update_linear.append(self.activation)
        for i in range(1,len(hidden_dims)-1):
            edge_update_linear.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            edge_update_linear.append(self.activation)
            node_update_linear.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            node_update_linear.append(self.activation)
            graph_update_linear.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            graph_update_linear.append(self.activation)
        self.edge_update_linear = nn.Sequential(*edge_update_linear)
        self.node_update_linear = nn.Sequential(*node_update_linear)
        self.graph_update_linear = nn.Sequential(*graph_update_linear)


    def edge_update_func(self, edges):
        return {'e': self.edge_update_linear( torch.cat([edges.src['v'], edges.dst['v'], edges.data['e'],edges.dst['u']], dim=1) )}

    def message_func(self, edges):
        return {'mail': edges.data['e']}
    
    def update_func(self, nodes):
        return {'ef': torch.mean(nodes.mailbox['mail'], 1)}

    def forward(self, graph, node_feats, edge_feats, graph_feats):
        with graph.local_scope():
            
            _node_feats = self.ff_node(node_feats)
            _edge_feats = self.ff_edge(edge_feats)
            _graph_feats = self.ff_graph(graph_feats)
            graph.ndata['v'] = _node_feats
            graph.edata['e'] = _edge_feats
            graph.ndata['u'] = _graph_feats

            # edge update
            graph.apply_edges(self.edge_update_func)
            graph.update_all(self.message_func, self.update_func)
            edge_feats_new = graph.edata['e']
            
            # node update
            node_feats_new = self.node_update_linear(torch.cat([graph.ndata['v'],graph.ndata['ef'],graph.ndata['u']],dim=1))

            # state update
            node_pooling, edge_pooling, graph_pooling = Megnet_pooling(graph, node_feats_new, edge_feats_new, _graph_feats)
            graph_feats_new = self.graph_update_linear(torch.cat([node_pooling,edge_pooling,graph_pooling],dim=1))
            expanded_indices = torch.repeat_interleave( torch.arange(graph_feats_new.shape[0]).to(device=graph_feats_new.device), graph.batch_num_nodes().int() )                                                   
            graph_feats_new = graph_feats_new[expanded_indices]

            node_feats_new = self.dropout(node_feats_new)
            edge_feats_new = self.dropout(edge_feats_new)
            graph_feats_new = self.dropout(graph_feats_new)

            node_feats_new = node_feats_new + node_feats
            edge_feats_new = edge_feats_new + edge_feats
            graph_feats_new = graph_feats_new + graph_feats

        return node_feats_new, edge_feats_new, graph_feats_new
    
class MegNets(nn.Module):
    def __init__(self,  in_dim, hidden_dims,
                 activation = nn.LeakyReLU, n_layers=4, dropout=0.1):
        super(MegNets, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(MegNet_layer(in_dim, hidden_dims,
                  activation=activation, dropout=dropout,))
        
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