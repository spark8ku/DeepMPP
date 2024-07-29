import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl
from dgl.nn import SumPooling

from .GCN import GCN_layer 
from .MPNN import MPNN_layer
from .distGCN import distGCN_layer

class GCconvolution(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False, alpha=0.1, max_dist = 4):
        super().__init__()        
        # Message Passing
        self.r2r = nn.ModuleList([MPNN_layer(in_node_feats, in_edge_feats, out_feats, activation, dropout, batch_norm, residual_sum) for _ in range(n_layers)])
        self.i2i = nn.ModuleList([GCN_layer(out_feats, out_feats, activation, dropout, batch_norm, residual_sum) for _ in range(n_layers)])

        self.r2i = r2i_layer()
        self.i2d = i2s_layer()
        self.d2d = distGCN_layer(out_feats, max_dist, out_feats, activation, alpha)

        self.d2score = nn.Sequential(nn.Linear(out_feats, out_feats//2), 
                                     nn.LeakyReLU(),
                                     nn.Linear(out_feats//2, out_feats//4),
                                     nn.LeakyReLU(),
                                     nn.Linear(out_feats//4, 1)
                                     )
    
    def forward(self, graph, r_node, r2r_edge, i_node, d2d_edge):
                
        real_graph=graph.node_type_subgraph(['r_nd'])
        real_graph.set_batch_num_nodes(graph.batch_num_nodes('r_nd'))
        real_graph.set_batch_num_edges(graph.batch_num_edges('r2r'))
        
        image_graph=graph.node_type_subgraph(['i_nd'])
        image_graph.set_batch_num_nodes(graph.batch_num_nodes('i_nd'))
        image_graph.set_batch_num_edges(graph.batch_num_edges('i2i'))
                
        dot_graph=graph.node_type_subgraph(['d_nd'])
        dot_graph.set_batch_num_nodes(graph.batch_num_nodes('d_nd'))
        dot_graph.set_batch_num_edges(graph.batch_num_edges('d2d'))

        for i in range(len(self.r2r)):
            r_node = self.r2r[i](real_graph, r_node, r2r_edge)
            i_node = self.r2i(graph, r_node, i_node)
            i_node = self.i2i[i](image_graph, i_node)
        d_node = self.i2d(graph, i_node)
        score= self.d2score(d_node)
        
        return score
        
class GCconvolution2(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False, alpha=0.1, max_dist = 4):
        super().__init__()        
        # Message Passing
        self.r2r = nn.ModuleList([GCN_layer(out_feats, out_feats, activation, dropout, batch_norm, residual_sum) for _ in range(n_layers)])
        self.i2i = nn.ModuleList([GCN_layer(out_feats, out_feats, activation, dropout, batch_norm, residual_sum) for _ in range(n_layers)])

        self.r2i = r2i_layer()
        self.i2d = i2s_layer()
        self.d2d = distGCN_layer(out_feats, max_dist, out_feats, activation, alpha)

        self.d2score = nn.Sequential(nn.Linear(out_feats, out_feats//2), 
                                     nn.LeakyReLU(),
                                     nn.Linear(out_feats//2, out_feats//4),
                                     nn.LeakyReLU(),
                                     nn.Linear(out_feats//4, 1)
                                     )
    
    def forward(self, graph, r_node, r2r_edge, i_node, d2d_edge):
                
        image_graph=graph.node_type_subgraph(['i_nd'])
        image_graph.set_batch_num_nodes(graph.batch_num_nodes('i_nd'))
        image_graph.set_batch_num_edges(graph.batch_num_edges('i2i'))
                
        dot_graph=graph.node_type_subgraph(['d_nd'])
        dot_graph.set_batch_num_nodes(graph.batch_num_nodes('d_nd'))
        dot_graph.set_batch_num_edges(graph.batch_num_edges('d2d'))

        for i in range(len(self.r2r)):
            r_node = self.r2r[i](image_graph, r_node)
            i_node = self.r2i(graph, r_node, i_node)
            i_node = self.i2i[i](image_graph, i_node)
        d_node = self.i2d(graph, i_node)
        score= self.d2score(d_node)
        
        return score
        
class r2i_layer(nn.Module):
    def forward(self,graph, r_node, i_node):
        return i_node+r_node
    
class i2s_layer(nn.Module):
    def forward(self,graph, i_node):
        with graph.local_scope():
            graph=graph.edge_type_subgraph([('i_nd','i2d','d_nd')])
            graph.nodes['i_nd'].data['h']= i_node
            graph.update_all(dgl.function.copy_u('h', 'mail'), dgl.function.sum('mail', 'h'))
            d_node = graph.nodes['d_nd'].data['h']
        return d_node    


class s2r_Layer(nn.Module):    
    def forward(self, graph, node):
        with graph.local_scope():
            graph=graph.edge_type_subgraph([('d_nd','d2r','r_nd')])
            graph.nodes['d_nd'].data['h']= node
            graph.update_all(dgl.function.copy_u('h', 'mail'), dgl.function.sum('mail', 'h'))
            score = graph.nodes['r_nd'].data['h']
        return score
