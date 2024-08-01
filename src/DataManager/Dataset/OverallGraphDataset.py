from .GraphDataset import GraphDataset, GraphDataset_withSolv


import torch
import dgl
import numpy as np

class OverallGraphdataset(GraphDataset):
    def __init__(self, graphs=None, graph_feats=None, target=None, smiles=None):
        if graphs is None: return
        self.graphs = graphs
        self.node_feature = [g.ndata['f'] for g in graphs]
        self.edge_feature = [g.edata['f'] for g in graphs]
        self.graph_feature = graph_feats
        self.target = torch.tensor(target).float()
        self.smiles = smiles

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.node_feature[idx], self.edge_feature[idx], self.graph_feature[idx], self.target[idx], self.smiles[idx]
    
    def reload(self, data):
        self.graphs, self.node_feature, self.edge_feature, self.graph_feature, self.target, self.smiles = data
        self.target = torch.stack(self.target)
        if self.target.dim() == 1:
            self.target = self.target.unsqueeze(-1)

    def subDataset(self, idx):
        self.graphs = [self.graphs[i] for i in idx]
        self.node_feature = [self.node_feature[i] for i in idx]
        self.edge_feature = [self.edge_feature[i] for i in idx]
        self.graph_feature = [self.graph_feature[i] for i in idx]
        self.target = self.target[np.array(idx, dtype=int)]
        self.smiles = [self.smiles[i] for i in idx]

    @staticmethod
    def collate(samples):
        graphs, node_feature, edge_feature, graph_feature, target, smiles = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        new_graph_feature = []
        for i in range(len(graphs)):
            new_graph_feature.append(torch.tile(graph_feature[i],(graphs[i].number_of_nodes(),1)))
        return batched_graph, torch.concat(node_feature,dim=0), torch.concat(edge_feature,dim=0), torch.concat(new_graph_feature,dim=0), torch.stack(target,dim=0), smiles
    
    @staticmethod
    def unwrapper(batch_graph, node_feature, edge_feature ,graph_feature, target, smiles, device='cpu'):
        batch_graph = batch_graph.to(device=device)
        node_feature = node_feature.float().to(device=device)
        edge_feature = edge_feature.float().to(device=device)
        graph_feature = graph_feature.float().to(device=device)
        target = target.float().to(device=device)

        return {"graph":batch_graph, "node_feats":node_feature, "edge_feats":edge_feature, "graph_feats":graph_feature, "target":target, "smiles":smiles}

class OverallGraphdataset_withSolv(GraphDataset_withSolv):
    def __init__(self, graphs=None, graph_feats=None, solv_graphs=None, solv_graph_feats=None, target=None, smiles=None, solv_smiles=None):
        super().__init__(graphs, solv_graphs, target, smiles, solv_smiles)
        if graphs is None: return
        self.graph_feature = graph_feats
        self.solv_graph_feature = solv_graph_feats

    def __getitem__(self, idx):
        args = super().__getitem__(idx)
        return args + (self.graph_feature[idx], self.solv_graph_feature[idx])
    
    def reload(self, data):
        *args, self.graph_feature, self.solv_graph_feature = data
        super().reload(args)


    def subDataset(self, idx):
        super().subDataset(idx)
        self.graph_feature = [self.graph_feature[i] for i in idx]
        self.solv_graph_feature = [self.solv_graph_feature[i] for i in idx]


    @staticmethod
    def collate(samples):
        graphs, node_feature, edge_feature, target, smiles, solv_graphs, solv_node_feature, solv_edge_feature, solv_smiles, graph_feature, solv_graph_feature = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_solv_graph = dgl.batch(solv_graphs)
        new_graph_feature = []
        new_solv_graph_feature = []
        for i in range(len(graphs)):
            new_graph_feature.append(torch.tile(graph_feature[i],(graphs[i].number_of_nodes(),1)))
            new_solv_graph_feature.append(torch.tile(solv_graph_feature[i],(solv_graphs[i].number_of_nodes(),1)))
        return batched_graph, torch.concat(node_feature,dim=0), torch.concat(edge_feature,dim=0), batched_solv_graph, torch.concat(solv_node_feature,dim=0), torch.concat(solv_edge_feature,dim=0), torch.stack(target,dim=0), smiles, solv_smiles, torch.concat(new_graph_feature,dim=0), torch.concat(new_solv_graph_feature,dim=0)
    
    @staticmethod
    def unwrapper(batch_graph, node_feature, edge_feature, batch_solv_graph, solv_node_feature, solv_edge_feature, target, smiles, solv_smiles, graph_feature, solv_graph_feature, device='cpu'):
        batch_graph = batch_graph.to(device=device)
        node_feature = node_feature.float().to(device=device)
        edge_feature = edge_feature.float().to(device=device)
        batch_solv_graph = batch_solv_graph.to(device=device)
        solv_node_feature = solv_node_feature.float().to(device=device)
        solv_edge_feature = solv_edge_feature.float().to(device=device)
        target = target.float().to(device=device)
        graph_feature = graph_feature.float().to(device=device)
        solv_graph_feature = solv_graph_feature.float().to(device=device)

        return {"graph":batch_graph, "node_feats":node_feature, "edge_feats":edge_feature, "solv_graph":batch_solv_graph, "solv_node_feats":solv_node_feature, "solv_edge_feats":solv_edge_feature, "target":target, "smiles":smiles, "solv_smiles":solv_smiles, "graph_feats":graph_feature, "solv_graph_feats":solv_graph_feature}
