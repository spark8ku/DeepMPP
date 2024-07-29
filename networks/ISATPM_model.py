import torch
import torch.nn as nn

import dgl
from dgl.nn import SumPooling

from networks.src.ISAT import ISATconvolution_PM
from networks.src.Linear import Linears
from networks.src.BiDropout import Bi_Dropout

class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()
        if config['target_dim']!=1:
            raise Exception('target_dim should be 1')
        
        
        hidden_dim = config.get('hidden_dim', 64)
        gcn_layers = config.get('conv_layers', 4)
        dropout = config.get('dropout', 0.1)
        linear_layers = config.get('linear_layers', 4)
        last_linear_dim = int(hidden_dim/(2**linear_layers))
        target_dim = config['target_dim']

        self.embedding_rnode_lin = nn.Sequential(
            nn.Linear(config['node_dim'], hidden_dim, bias=False)
        )
        self.embedding_inode_lin = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False)
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(config['edge_dim'], hidden_dim, bias=False)
        )
        self.ISATconv_PM = ISATconvolution_PM(hidden_dim, hidden_dim, hidden_dim, nn.LeakyReLU(), gcn_layers, dropout, False, True, 0.1)

        self.linears = Linears(hidden_dim,last_linear_dim, nn.ReLU(), linear_layers, dropout, False, False)
        self.out_linear = nn.Linear(last_linear_dim, config['target_dim'])
        self.reduce = SumPooling()

        self.Bi_Dropout=Bi_Dropout(0.4)

        self.score_var = config.get('score_var',0.17)
        self.score_mean = config.get('score_mean',0.5)
        self.alpha = config.get('alpha',0.09)
        self.beta = config.get('beta',0.03)
        self.gamma = config.get('gamma',0.03)
        
    
    def forward(self, graph, r_node, i_node, r_edge, d_edge,**kargs):
        self.Bi_Dropout.set_drop_p(self.training)
        
        r_node = r_node.float()
        r_node = self.embedding_rnode_lin(r_node)
        i_node = i_node.float()
        i_node = self.embedding_inode_lin(i_node)
        r_edge = r_edge.float()
        r_edge = self.embedding_edge_lin(r_edge)
        
        real_graph=graph.node_type_subgraph(['r_nd'])
        real_graph.set_batch_num_nodes(graph.batch_num_nodes('r_nd'))
        real_graph.set_batch_num_edges(graph.batch_num_edges('r2r'))

        r_node_P, r_node_M, score_P, score_M = self.ISATconv_PM(graph, r_node, r_edge, i_node, d_edge, **kargs)
        if kargs.get('get_feature',False):
            return {'feature_P':r_node_P,'feature_N':r_node_M,'positive':score_P,'negative':score_M}

        h_P = self.reduce(real_graph, r_node_P)
        h_P = self.linears(h_P)
        h_P = self.out_linear(h_P)

        h_M = self.reduce(real_graph, r_node_M)
        h_M = self.linears(h_M)
        h_M = self.out_linear(h_M)

        h=torch.concat([h_P,h_M],axis=1)
        h=self.Bi_Dropout(h)
        output=torch.unsqueeze(h[:,0]-h[:,1],axis=1)
        if kargs.get('get_score',False):
            return {'prediction':output, 'positive':score_P,'negative':score_M}
        
        self.p_score_var = self.ISATconv_PM.p_score_var
        self.n_score_var = self.ISATconv_PM.n_score_var
        self.p_score_mean = self.ISATconv_PM.p_score_mean
        self.n_score_mean = self.ISATconv_PM.n_score_mean
        self.p_score_ms = self.ISATconv_PM.p_score_ms
        self.n_score_ms = self.ISATconv_PM.n_score_ms

        return output
        
    def loss_fn(self, scores, targets):
        scores,targets=self.Bi_Dropout.drop_label(scores[~torch.isnan(targets)],targets[~torch.isnan(targets)])
        return torch.mean(torch.square(scores-targets))+\
                (self.p_score_ms+self.n_score_ms)*self.alpha +\
                (torch.abs(self.score_var-self.p_score_var)+torch.abs(self.score_var-self.n_score_var))*self.beta +\
                (torch.abs(self.score_mean-self.p_score_mean)+torch.abs(self.score_mean-self.n_score_mean))*self.gamma
    


