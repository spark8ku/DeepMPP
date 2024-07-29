
from networks.src.GCN import GCNs
from networks.src.Linear import Linears
import torch.nn as nn
import torch
from dgl.nn import SumPooling

class SolventLayer(nn.Module):
    def __init__(self, config):
        super(SolventLayer, self).__init__()
        
        hidden_dim = config.get('hidden_dim', 64)
        linear_layers = min(config.get('linear_layers', 2),6)
        self.node_embedding = nn.Linear(config['node_dim'], hidden_dim)
        self.node_embedding_solv = nn.Linear(config['node_dim'], 64)

        self.GCNs_solv = GCNs(64, 64, 64, nn.ReLU(), 4, 0.2, False, True)
        self.Linears2 = Linears(64,64, nn.ReLU(), 2, 0.2, False, False) # in_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        self.Linears3 = Linears(hidden_dim+64,config['target_dim'], nn.ReLU(), linear_layers, 0.2, False, False,True) # in_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False

    def forward(self, hidden_feats,solv_graph,solv_node_feats,**kwargs):
        h = hidden_feats

        h_solv = self.node_embedding_solv(solv_node_feats)
        h_solv = self.GCNs_solv(solv_graph, h_solv)
        h_solv = SumPooling()(solv_graph,h_solv)

        h_solv = self.Linears2(h_solv)
        h = torch.cat([h,h_solv],axis=1)
        h = self.Linears3(h)
        return h