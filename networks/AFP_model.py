import torch.nn as nn
import torch

from networks.src.AFP import Atom_AttentiveFP, Mol_AttentiveFP
from networks.src.Linear import Linears
from networks.src.MPNN import MPNNs
from dgl.nn import SumPooling

class network(nn.Module):
    def __init__(self, config):
        super().__init__()
        config['mol_layers']=config.get('mol_layers', 1)
        config['dropout']=config.get('dropout', 0.2)

        hidden_dim = config.get('hidden_dim', 64)
        conv_layers = config.get('conv_layers', 4)
        linear_layers = config.get('linear_layers', 2)
        dropout = config.get('dropout', 0.2)

        self.embedding_node_lin = nn.Linear(config['node_dim'], hidden_dim, bias=True)
        self.embedding_edge_lin = nn.Linear(config['edge_dim'], hidden_dim, bias=True)

        # self.Atom_Attentive = Atom_AttentiveFP(config)
        self.Mol_Attentive = Mol_AttentiveFP(config)
        self.Atom_Attentive = MPNNs(hidden_dim, hidden_dim, hidden_dim,hidden_dim, nn.LeakyReLU(), conv_layers, dropout, False, True)
        self.Linears = Linears(hidden_dim,config['target_dim'], nn.ReLU(), linear_layers, dropout, False, False, True) 
        

    def forward(self, graph, node_feats, edge_feats, **kwargs):
        node = self.embedding_node_lin(node_feats)
        edge = self.embedding_edge_lin(edge_feats)

        new_node = self.Atom_Attentive(graph, node, edge)

        super_node, attention_list = self.Mol_Attentive(graph, new_node)
        # super_node = SumPooling()( graph, new_node)

        output = self.Linears(super_node)
        
        if kwargs.get('get_score',False):
            return {'prediction':output, 'positive':attention_list}
        
        return output

        
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])
