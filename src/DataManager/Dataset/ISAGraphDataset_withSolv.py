import torch
import dgl
import numpy as np
from .ISAGraphDataset import ISAGraphDataset

class ISAGraphDataset_withSolv(ISAGraphDataset):
    def __init__(self, graphs=None, solv_graphs=None, target=None, smiles=None, solv_smiles=None):
        super().__init__(graphs, target, smiles)
        if graphs is None: return
        self.solv_node_feature = [g.nodes['r_nd'].data['f'] for g in solv_graphs]
        self.solv_edge_feature = [g.edges['r2r'].data['f'] for g in solv_graphs]
        self.solv_graphs = solv_graphs
        self.solv_smiles = solv_smiles

    def __getitem__(self, idx):
        args = super().__getitem__(idx)
        return args + (self.solv_graphs[idx], self.solv_node_feature[idx], self.solv_edge_feature[idx], self.solv_smiles[idx])
    
    def reload(self, data):
        *args, self.solv_graphs, self.solv_node_feature, self.solv_edge_feature, self.solv_smiles = data
        super().reload(args)


    def subDataset(self, idx):
        super().subDataset(idx)
        self.solv_graphs = [self.solv_graphs[i] for i in idx]
        self.solv_node_feature = [self.solv_node_feature[i] for i in idx]
        self.solv_edge_feature = [self.solv_edge_feature[i] for i in idx]
        self.solv_smiles = [self.solv_smiles[i] for i in idx]

    def get_subDataset(self, idx):
        
        graphs = [self.graphs[i] for i in idx]
        r_node = [self.r_node[i] for i in idx]
        r2r_edge = [self.r2r_edge[i] for i in idx]
        i_node = [self.i_node[i] for i in idx]
        d_node = [self.d_node[i] for i in idx]
        d2d_edge = [self.d2d_edge[i] for i in idx]
        target = [self.target[i] for i in idx]
        smiles = [self.smiles[i] for i in idx]
        
        solv_graphs = [self.solv_graphs[i] for i in idx]
        solv_node_feature = [self.solv_node_feature[i] for i in idx]
        solv_edge_feature = [self.solv_edge_feature[i] for i in idx]
        solv_smiles = [self.solv_smiles[i] for i in idx]

        dataset = ISAGraphDataset_withSolv()
        dataset.reload((graphs, r_node, r2r_edge, i_node, d_node, d2d_edge, target, smiles, solv_graphs, solv_node_feature, solv_edge_feature, solv_smiles))
        return dataset
        


    @staticmethod
    def collate(samples):
        graphs,  r_node, r2r_edge, i_node, d_node, d2d_edge, target, smiles, solv_graphs, solv_node_feature, solv_edge_feature, solv_smiles = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_solv_graph = dgl.batch(solv_graphs)
        return batched_graph, torch.concat(r_node,dim=0), torch.concat(r2r_edge,dim=0), torch.concat(i_node,dim=0), torch.concat(d_node,dim=0), torch.concat(d2d_edge,dim=0), batched_solv_graph, torch.concat(solv_node_feature,dim=0), torch.concat(solv_edge_feature,dim=0), torch.stack(target,dim=0), smiles, solv_smiles
    
    @staticmethod
    def unwrapper(graph, r_node, r2r_edge, i_node, d_node, d2d_edge,  batch_solv_graph, solv_node_feature, solv_edge_feature, target, smiles, solv_smiles, device='cpu'):
        graph = graph.to(device=device)
        r_node = r_node.float().to(device=device)
        r2r_edge = r2r_edge.float().to(device=device)
        i_node = i_node.float().to(device=device)
        d_node = d_node.float().to(device=device)
        d2d_edge = d2d_edge.float().to(device=device)

        batch_solv_graph = batch_solv_graph.to(device=device)
        solv_node_feature = solv_node_feature.float().to(device=device)
        solv_edge_feature = solv_edge_feature.float().to(device=device)
        target = target.float().to(device=device)

        return {'graph':graph, 'r_node':r_node, 'r_edge':r2r_edge, 'i_node':i_node, 'd_node':d_node, 'd_edge':d2d_edge, "solv_graph":batch_solv_graph, "solv_node_feats":solv_node_feature, "solv_edge_feats":solv_edge_feature, "target":target, "smiles":smiles, "solv_smiles":solv_smiles}