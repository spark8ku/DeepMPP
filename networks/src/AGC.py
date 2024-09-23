import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl
from dgl.nn import SumPooling

from .AFP import Mol_AttentiveFP
from .GAT import GATs
from .ISAT import i2s_layer

class AGC(nn.Module):
    def __init__(self, in_node_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False, mol_layers=1):
        super().__init__()        
        # Message Passing

        self.i2i = GATs(in_node_feats, out_feats, out_feats, activation, n_layers, dropout, batch_norm, residual_sum)

        self.i2d = i2s_layer()

        self.d2d = GATs(in_node_feats, out_feats, out_feats, activation, n_layers, dropout, batch_norm, residual_sum)

        self.d_context = Mol_AttentiveFP({'hidden_dim':out_feats, 'dropout':dropout, 'mol_layers':mol_layers})
        
        self.reduce = SumPooling()
    
    def forward(self, graph, i_node):
        image_graph=graph.node_type_subgraph(['i_nd'])
        image_graph.set_batch_num_nodes(graph.batch_num_nodes('i_nd'))
        image_graph.set_batch_num_edges(graph.batch_num_edges('i2i'))
                
        dot_graph=graph.node_type_subgraph(['d_nd'])
        dot_graph.set_batch_num_nodes(graph.batch_num_nodes('d_nd'))
        dot_graph.set_batch_num_edges(graph.batch_num_edges('d2d'))

        i_node = self.i2i(image_graph, i_node)
        d_node = self.i2d(graph, i_node)

        d_node = self.d2d(dot_graph, d_node)
        super_node, node_attention = self.d_context(dot_graph, d_node)
        
        return super_node, node_attention
        