import torch
from torch.utils.data import Dataset
import dgl
import numpy as np

class ISAGraphDataset(Dataset):
    def __init__(self, graphs=None, target=None, smiles=None):
        if graphs is None: return
        self.graphs = graphs
        self.r_node = [g.nodes['r_nd'].data['f'] for g in graphs]
        self.r2r_edge = [g.edges['r2r'].data['f'] for g in graphs]
        self.i_node = [g.nodes['i_nd'].data['f'] for g in graphs]
        self.d_node = [g.nodes['d_nd'].data['f'] for g in graphs]
        self.d2d_edge = [g.edges['d2d'].data['dist'] for g in graphs]
        self.target = torch.tensor(target).float()
        self.smiles = smiles


    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.r_node[idx], self.r2r_edge[idx], self.i_node[idx], self.d_node[idx], self.d2d_edge[idx], self.target[idx], self.smiles[idx]
    
    def reload(self, data):
        self.graphs, self.r_node, self.r2r_edge, self.i_node, self.d_node, self.d2d_edge, self.target, self.smiles = data
        self.target = torch.stack(self.target)
        if self.target.dim() == 1:
            self.target = self.target.unsqueeze(-1)

    def subDataset(self, idx):
        self.graphs = [self.graphs[i] for i in idx]
        self.r_node = [self.r_node[i] for i in idx]
        self.r2r_edge = [self.r2r_edge[i] for i in idx]
        self.i_node = [self.i_node[i] for i in idx]
        self.d_node = [self.d_node[i] for i in idx]
        self.d2d_edge = [self.d2d_edge[i] for i in idx]
        self.target = self.target[np.array(idx, dtype=int)]
        self.smiles = [self.smiles[i] for i in idx]

    @staticmethod
    def collate(samples):
        graphs, r_node, r2r_edge, i_node, d_node, d2d_edge, target, smiles = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.concat(r_node,dim=0), torch.concat(r2r_edge,dim=0), torch.concat(i_node,dim=0), torch.concat(d_node,dim=0), torch.concat(d2d_edge,dim=0), torch.stack(target,dim=0), smiles
    
    @staticmethod
    def unwrapper(graph, r_node, r2r_edge, i_node, d_node, d2d_edge, target, smiles, device='cpu'):
        graph = graph.to(device=device)
        r_node = r_node.float().to(device=device)
        r2r_edge = r2r_edge.float().to(device=device)
        i_node = i_node.float().to(device=device)
        d_node = d_node.float().to(device=device)
        d2d_edge = d2d_edge.float().to(device=device)
        target = target.float().to(device=device)
        return {'graph':graph, 'r_node':r_node, 'r_edge':r2r_edge, 'i_node':i_node, 'd_node':d_node, 'd_edge':d2d_edge, 'target':target, 'smiles':smiles}
    