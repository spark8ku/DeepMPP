
"""
This codes are modified from the project "GC-GNN" (https://github.com/gsi-lab/GC-GNN)
The original codes are under the MIT License. (https://github.com/gsi-lab/GC-GNN/blob/main/networks/AttentiveFP.py)
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax


class Atom_AFPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.embedding_node_lin = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.LeakyReLU()
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim, bias=True),
            nn.LeakyReLU()
        )
        self.cal_alignment = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(self.hidden_dim + self.hidden_dim, 1, bias=True),
            nn.LeakyReLU()
        )
        self.attend = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        )
        self.GRUCell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(config['dropout'])
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_node_lin[0].reset_parameters()
        self.embedding_edge_lin[0].reset_parameters()
        self.cal_alignment[1].reset_parameters()
        self.attend[0].reset_parameters()
        self.GRUCell.reset_parameters()

    def update_edge_by_neighbor(self, edges):
        neighbor_message = self.embedding_edge_lin(torch.cat([edges.src['node_embedded_feats'], edges.data['edge_feats']], dim=-1))
        return {'neighbor_message': neighbor_message}

    def cal_alignment_score(self, edges):
        alignment_score = self.cal_alignment(torch.cat([edges.data['neighbor_message'], edges.dst['node_embedded_feats']], dim=-1))
        return {'score': alignment_score}

    def att_context_passing(self, edges):
        return {'mail': edges.data['att_context']}

    def cal_context(self, nodes):
        return {'context': torch.sum(self.dropout(nodes.mailbox['mail']), dim=-2)}

    def forward(self, graph, node, edge):
        graph = graph.local_var()
        graph.ndata['node_feats'] = node
        graph.ndata['node_embedded_feats'] = self.embedding_node_lin(graph.ndata['node_feats'])
        graph.edata['edge_feats'] = edge

        # update the edge feats by concat edge feats together with the neighborhood node feats
        graph.apply_edges(self.update_edge_by_neighbor)
        graph.apply_edges(self.cal_alignment_score)
        graph.edata['att_context'] = edge_softmax(graph, graph.edata['score']) * self.attend(graph.edata['neighbor_message'])
        graph.update_all(self.att_context_passing, self.cal_context)
        context = nn.ELU()(graph.ndata['context'])
        new_node = nn.ReLU()(self.GRUCell(context, graph.ndata['node_embedded_feats']))
        # new_node = context + graph.ndata['node_embedded_feats']
        return new_node

class Mol_AFPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.GRUCell = nn.GRUCell(config['hidden_dim'], config['hidden_dim'])

        self.cal_alignment = nn.Sequential(
            nn.Linear(config['hidden_dim'] + config['hidden_dim'], 1, bias=True),
            nn.LeakyReLU()
        )
        self.attend = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['hidden_dim'], bias=True)
        )
        self.dropout = nn.Dropout(config['dropout'])
        self.reset_parameters()

    def reset_parameters(self):
        self.GRUCell.reset_parameters()
        self.cal_alignment[0].reset_parameters()
        self.attend[1].reset_parameters()

    def forward(self, graph, super_node, node):
        graph = graph.local_var()
        super_node = F.leaky_relu(super_node)

        graph.ndata['score'] = self.cal_alignment(torch.cat([node, dgl.broadcast_nodes(graph, super_node)], dim=1))
        graph.ndata['attention_weight'] = dgl.softmax_nodes(graph, 'score')
        graph.ndata['hidden_node'] = self.attend(self.dropout(node))
        super_context = F.elu(dgl.sum_nodes(graph, 'hidden_node', 'attention_weight'))
        # super_node = super_context + super_node
        super_node = F.relu(self.GRUCell(super_context, super_node))
        return super_node, graph.ndata['attention_weight']


class AttentiveFP(nn.Module):
    # Generate Context of each nodes
    def __init__(self, config):
        super().__init__()
        conv_layers = config.get('conv_layers', 4)
        T = config.get("T", conv_layers)
        self.PassingDepth = nn.ModuleList([Atom_AFPLayer(config) for _ in range(conv_layers)])
        self.MultiTimeSteps = nn.ModuleList([Mol_AFPLayer(config) for d in range(T)])
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.PassingDepth:
            l.reset_parameters()

        for l in self.MultiTimeSteps:
            l.reset_parameters()

    def forward(self, graph, node, edge):
        with graph.local_scope():
            for i in range(len(self.PassingDepth)):
                node = self.PassingDepth[i](graph, node, edge)
            
            graph.ndata['node_feats'] = node
            super_node = dgl.sum_nodes(graph, 'node_feats')

            for i in range(len(self.MultiTimeSteps)):
                super_node, att_w = self.MultiTimeSteps[i](graph, super_node, node)
        return super_node, att_w


class Mol_AttentiveFP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.MultiTimeSteps = nn.ModuleList([Mol_AFPLayer(config) for d in range(config['mol_layers'])])
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.MultiTimeSteps:
            l.reset_parameters()

    def forward(self, graph, node):
        with graph.local_scope():
            attention_list = []
            graph.ndata['hidden_node'] = node
            super_node = dgl.sum_nodes(graph, 'hidden_node')
            for step in self.MultiTimeSteps:
                super_node, attention_t = step(graph, super_node, node)
                attention_list.append(attention_t)
            attention_list = torch.cat(attention_list, dim=1)
        return super_node, attention_list



