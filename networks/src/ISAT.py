import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl
from dgl.nn import SumPooling

from .GCN import GCN_layer 
from .MPNN import MPNN_layer
from .distGCN import distGCN_layer

class ISATconvolution(nn.Module):
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
                                     nn.Linear(out_feats//2, 1)
                                     )
        self.d2r = s2r_Layer()
        self.batch_norm = nn.BatchNorm1d(1)
        
        self.reduce = SumPooling()
    
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
        d_node = self.d2d(dot_graph, d_node, d2d_edge)
        score= self.d2r(graph, d_node)
        score= self.d2score(score)
        score= nn.Sigmoid()(self.batch_norm(score))
        
        r_node=score*r_node
        return r_node, score
        
class ISATconvolution_PM(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False, alpha=0.1, max_dist = 4):
        super().__init__()        
        # Message Passing
        self.r2r = nn.ModuleList([MPNN_layer(in_node_feats, in_edge_feats, out_feats, activation, dropout, batch_norm, residual_sum) for _ in range(n_layers)])


        self.i2i_P = nn.ModuleList([GCN_layer(out_feats, out_feats, activation, dropout, batch_norm, residual_sum) for _ in range(n_layers)])
        self.d2d_P = distGCN_layer(out_feats, max_dist, out_feats, activation, alpha)
        self.d2score_P = nn.Sequential(nn.Linear(out_feats, out_feats//2), 
                                     nn.LeakyReLU(),
                                     nn.Linear(out_feats//2, 3),
                                     nn.LeakyReLU()
                                     )
        self.d2score_pl = nn.Linear(3, 1)

        self.i2i_M = nn.ModuleList([GCN_layer(out_feats, out_feats, activation, dropout, batch_norm, residual_sum) for _ in range(n_layers)])
        self.d2d_M = distGCN_layer(out_feats, max_dist, out_feats, activation, alpha)
        self.d2score_M = nn.Sequential(nn.Linear(out_feats, out_feats//2), 
                                     nn.LeakyReLU(),
                                     nn.Linear(out_feats//2,3),
                                     nn.LeakyReLU()
                                     )
        self.d2score_ml = nn.Linear(3, 1)
        
        self.r2i = r2i_layer()
        self.i2d = i2s_layer()
        self.d2r = s2r_Layer()

        self.batch_norm_P = nn.BatchNorm1d(1)
        self.batch_norm_M = nn.BatchNorm1d(1)
        
        self.reduce = SumPooling()
    
    def forward(self, graph, r_node, r2r_edge, i_node, d2d_edge, **kargs):
                
        real_graph=graph.node_type_subgraph(['r_nd'])
        real_graph.set_batch_num_nodes(graph.batch_num_nodes('r_nd'))
        real_graph.set_batch_num_edges(graph.batch_num_edges('r2r'))
        
        image_graph=graph.node_type_subgraph(['i_nd'])
        image_graph.set_batch_num_nodes(graph.batch_num_nodes('i_nd'))
        image_graph.set_batch_num_edges(graph.batch_num_edges('i2i'))
                
        dot_graph=graph.node_type_subgraph(['d_nd'])
        dot_graph.set_batch_num_nodes(graph.batch_num_nodes('d_nd'))
        dot_graph.set_batch_num_edges(graph.batch_num_edges('d2d'))

        i_node_P = i_node
        i_node_M = i_node
        for i in range(len(self.r2r)):
            r_node = self.r2r[i](real_graph, r_node, r2r_edge)

            i_node_P = self.r2i(graph, r_node, i_node_P)
            i_node_P = self.i2i_P[i](image_graph, i_node_P)

            i_node_M = self.r2i(graph, r_node, i_node_M)
            i_node_M = self.i2i_M[i](image_graph, i_node_M)


        d_node_P = self.i2d(graph, i_node_P)
        score_P = self.d2d_P(dot_graph, d_node_P, d2d_edge)
        score_P_temp= score_P.clone()
        score_P= self.d2score_P(score_P)
        score_P= self.d2score_pl(score_P)
        score_P= nn.Sigmoid()(self.batch_norm_P(score_P))
        score_P_node= self.d2r(graph, score_P)

        r_node_P=score_P_node*r_node


        d_node_M = self.i2d(graph, i_node_M)
        score_M = self.d2d_M(dot_graph, d_node_M, d2d_edge)     
        score_M_temp= score_M.clone()
        score_M= self.d2score_M(score_M)
        score_M= self.d2score_ml(score_M)   
        score_M= nn.Sigmoid()(self.batch_norm_M(score_M))
        score_M_node= self.d2r(graph, score_M)

        r_node_M=score_M_node*r_node

        if kargs.get('get_feature',False):
            return score_P_temp, score_M_temp, score_P, score_M # score by group, score by atom
        
        self.p_score_var = torch.var(score_P.view(-1))
        self.n_score_var = torch.var(score_M.view(-1))
        self.p_score_mean = torch.mean(score_P.view(-1))
        self.n_score_mean = torch.mean(score_M.view(-1))
        self.p_score_ms = torch.mean(torch.square(0.5-score_P.view(-1)))
        self.n_score_ms = torch.mean(torch.square(0.5-score_M.view(-1)))
        return r_node_P, r_node_M, score_P, score_M

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

