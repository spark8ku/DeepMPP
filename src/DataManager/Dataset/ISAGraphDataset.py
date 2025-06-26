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
        self.d2d_edge = [g.edges['d2d'].data['f'] for g in graphs]
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

    # def subDataset(self, idx):
    #     self.graphs = [self.graphs[i] for i in idx]
    #     self.r_node = [self.r_node[i] for i in idx]
    #     self.r2r_edge = [self.r2r_edge[i] for i in idx]
    #     self.i_node = [self.i_node[i] for i in idx]
    #     self.d_node = [self.d_node[i] for i in idx]
    #     self.d2d_edge = [self.d2d_edge[i] for i in idx]
    #     self.target = self.target[np.array(idx, dtype=int)]
    #     self.smiles = [self.smiles[i] for i in idx]

    def get_subDataset(self, idx):
        graphs = [self.graphs[i] for i in idx]
        r_node = [self.r_node[i] for i in idx]
        r2r_edge = [self.r2r_edge[i] for i in idx]
        i_node = [self.i_node[i] for i in idx]
        d_node = [self.d_node[i] for i in idx]
        d2d_edge = [self.d2d_edge[i] for i in idx]
        target = [self.target[i] for i in idx]
        smiles = [self.smiles[i] for i in idx]
        
        dataset = ISAGraphDataset()
        dataset.reload((graphs, r_node, r2r_edge, i_node, d_node, d2d_edge, target, smiles))
        return dataset

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
    
    
class ISAGraphDataset_v1p3(ISAGraphDataset):
    def __init__(self, graphs=None, target=None, smiles=None, numeric_inputs=None):
        if graphs is None: 
            self.graphs = {}
            self.target = None
            self.smiles = {}
            self.data_keys = []
            return


        for key in graphs:
            setattr(self, key + '_graphs', graphs[key])
            try:
                setattr(self, key + '_r_node', [g.nodes['r_nd'].data['f'] for g in graphs[key]])
            except KeyError:
                print(f"Warning: 'r_nd' node type not found in graphs for key '{key}'. Initializing with zeros.")
                setattr(self, key + '_r_node', [torch.zeros((g.num_nodes(), 1)) for g in graphs[key]])

            try:
                setattr(self, key + '_r2r_edge', [g.edges['r2r'].data['f'] for g in graphs[key]])
            except KeyError:
                print(f"Warning: 'r2r' edge type not found in graphs for key '{key}'. Initializing with zeros.")
                setattr(self, key + '_r2r_edge', [torch.zeros((g.num_edges(), 1)) for g in graphs[key]])
            # if 'i_nd' in graphs[key][0].nodes:
            #     setattr(self, key + '_i_node', [g.nodes['i_nd'].data['f'] for g in graphs[key]])
            # else:
            #     setattr(self, key + '_i_node', [torch.zeros((g.num_nodes(), 0)) for g in graphs[key]])
            try:
                setattr(self, key + '_i2i_edge', [g.edges['i2i'].data['f'] for g in graphs[key]])
            except KeyError:
                print(f"Warning: 'i2i' edge type not found in graphs for key '{key}'. Initializing with zeros.")
                setattr(self, key + '_i2i_edge', [torch.zeros((g.num_edges(), 1)) for g in graphs[key]])



            # if 'i2i' in graphs[key][0].nodes:
            #     setattr(self, key + '_i_node', [g.nodes['i_nd'].data['f'] for g in graphs[key]])
            # else:
            #     setattr(self, key + '_i_node', [torch.zeros((g.num_nodes(), 0)) for g in graphs[key]])
            try:
                setattr(self, key + '_i_node', [g.nodes['i_nd'].data['f'] for g in graphs[key]])
            except KeyError:
                print(f"Warning: 'i_nd' node type not found in graphs for key '{key}'. Initializing with zeros.")
                setattr(self, key + '_i_node', [torch.zeros((g.num_nodes(), 1)) for g in graphs[key]])

            try:
                setattr(self, key + '_d_node', [g.nodes['d_nd'].data['f'] for g in graphs[key]])
            except KeyError:
                print(f"Warning: 'd_nd' node type not found in graphs for key '{key}'. Initializing with zeros.")
                setattr(self, key + '_d_node', [torch.zeros((g.num_nodes(), 1)) for g in graphs[key]])

            try:
                setattr(self, key + '_d2d_edge', [g.edges['d2d'].data['f'] for g in graphs[key]])
            except KeyError:
                print(f"Warning: 'd2d' edge type not found in graphs for key '{key}'. Initializing with zeros.")
                setattr(self, key + '_d2d_edge', [torch.zeros((g.num_edges(), 1)) for g in graphs[key]])

            # if 'd_nd' in graphs[key][0].nodes:
            #     setattr(self, key + '_d_node', [g.nodes['d_nd'].data['f'] for g in graphs[key]])
            # else:
            #     setattr(self, key + '_d_node', [torch.zeros((g.num_nodes(), 0)) for g in graphs[key]])

            # if 'd2d' in graphs[key][0].edges:
            #     setattr(self, key + '_d2d_edge', [g.edges['d2d'].data['dist'] for g in graphs[key]])
            # else:
            #     setattr(self, key + '_d2d_edge', [torch.zeros((g.num_edges(), 0)) for g in graphs[key]])

        if numeric_inputs is not None:
            for key in numeric_inputs:
                setattr(self, key + '_var', numeric_inputs[key])
        else:
            numeric_inputs = {}

        for key in smiles:
            if key not in graphs:
                raise ValueError(f"Key '{key}' in smiles is not found in graphs.")
            setattr(self, key + '_smiles', smiles[key])

        self.target = torch.tensor(target).float() if target is not None else None
        self.data_keys = list(graphs.keys()) + list(numeric_inputs.keys()) + ['target']

    def __len__(self):
        if getattr(self, 'target', None) is not None:
            return len(self.target)
        else:
            if len(self.data_keys) == 0:
                return 0
            key = self.data_keys[0]
            if hasattr(self, key + '_graphs'):
                return len(getattr(self, key + '_graphs'))
            elif hasattr(self, key + '_var'):
                return len(getattr(self, key + '_var'))
            elif hasattr(self, key + '_smiles'):
                return len(getattr(self, key + '_smiles'))
            else:
                return 0

    def __getitem__(self, idx):
        item = {}
        for key in self.data_keys:
            if hasattr(self, key + '_graphs'):
                item[key + '_graphs'] = getattr(self, key + '_graphs')[idx]
                # if hasattr(self, key + '_node_feature'):
                #     nf = getattr(self, key + '_node_feature')[idx]
                #     if type(nf) is not torch.Tensor:
                #         nf = torch.tensor(nf, dtype=torch.float32)
                #     item[key + '_node_feature'] = nf
                # if hasattr(self, key + '_edge_feature'):
                #     ef = getattr(self, key + '_edge_feature')[idx]
                #     if type(ef) is not torch.Tensor:
                #         ef = torch.tensor(ef, dtype=torch.float32)
                #     item[key + '_edge_feature'] = ef
                if hasattr(self, key + '_r_node'):
                    f = getattr(self, key + '_r_node')[idx]
                    if type(f) is not torch.Tensor:
                        f = torch.tensor(f, dtype=torch.float32)
                    item[key + '_r_node'] = f
                if hasattr(self, key + '_r2r_edge'):
                    f = getattr(self, key + '_r2r_edge')[idx]
                    if type(f) is not torch.Tensor:
                        f = torch.tensor(f, dtype=torch.float32)
                    item[key + '_r2r_edge'] = f
                if hasattr(self, key + '_i_node'):
                    f = getattr(self, key + '_i_node')[idx]
                    if type(f) is not torch.Tensor:
                        f = torch.tensor(f, dtype=torch.float32)
                    item[key + '_i_node'] = f
                if hasattr(self, key + '_i2i_edge'):
                    f = getattr(self, key + '_i2i_edge')[idx]
                    if type(f) is not torch.Tensor:
                        f = torch.tensor(f, dtype=torch.float32)
                    item[key + '_i2i_edge'] = f
                if hasattr(self, key + '_d_node'):
                    f = getattr(self, key + '_d_node')[idx]
                    if type(f) is not torch.Tensor:
                        f = torch.tensor(f, dtype=torch.float32)
                    item[key + '_d_node'] = f
                if hasattr(self, key + '_d2d_edge'):
                    f = getattr(self, key + '_d2d_edge')[idx]
                    if type(f) is not torch.Tensor:
                        f = torch.tensor(f, dtype=torch.float32)
                    item[key + '_d2d_edge'] = f

            elif hasattr(self, key + '_var'):
                f = getattr(self, key + '_var')[idx]
                if type(f) is not torch.Tensor:
                    f = torch.tensor(f, dtype=torch.float32)
                item[key + '_var'] = f

            elif hasattr(self, key + '_smiles'):
                item[key + '_smiles'] = getattr(self, key + '_smiles')[idx]
        if hasattr(self, 'target') and self.target is not None:
            if type(self.target) is not torch.Tensor:
                self.target = torch.tensor(self.target, dtype=torch.float32)
            item['target'] = self.target[idx]
        return item

    def reload(self, data):
        new_data_keys = []
        for key in data[0]:
            setattr(self, key, [d[key] for d in data])
            if key.endswith('_graphs'):
                new_data_keys.append(key[:-7])  # Remove '_graphs'
            elif key == 'target':
                new_data_keys.append(key)

        self.data_keys = new_data_keys


    def subDataset(self, idx):
        for key in self.data_keys:
            if hasattr(self, key + '_graphs'):
                setattr(self, key + '_graphs', [getattr(self, key + '_graphs')[i] for i in idx])
                # setattr(self, key + '_node_feature', [getattr(self, key + '_node_feature')[i] for i in idx])
                # setattr(self, key + '_edge_feature', [getattr(self, key + '_edge_feature')[i] for i in idx])
            if hasattr(self, key + '_r_node'):
                setattr(self, key + '_r_node', [getattr(self, key + '_r_node')[i] for i in idx])
            if hasattr(self, key + '_r2r_edge'):
                setattr(self, key + '_r2r_edge', [getattr(self, key + '_r2r_edge')[i] for i in idx])
            if hasattr(self, key + '_i_node'):
                setattr(self, key + '_i_node', [getattr(self, key + '_i_node')[i] for i in idx])
            if hasattr(self, key + '_i2i_edge'):
                setattr(self, key + '_i2i_edge', [getattr(self, key + '_i2i_edge')[i] for i in idx])
            if hasattr(self, key + '_d_node'):
                setattr(self, key + '_d_node', [getattr(self, key + '_d_node')[i] for i in idx])
            if hasattr(self, key + '_d2d_edge'):
                setattr(self, key + '_d2d_edge', [getattr(self, key + '_d2d_edge')[i] for i in idx])

            if hasattr(self, key + '_var'):   
                setattr(self, key + '_var', getattr(self, key + '_var')[np.array(idx, dtype=int)])
            if hasattr(self, key + '_smiles'):
                setattr(self, key + '_smiles', [getattr(self, key + '_smiles')[i] for i in idx])
        if self.target is not None:
            self.target = self.target[np.array(idx, dtype=int)]


    @staticmethod
    def collate(samples):
        batched_data = {}
        for key in samples[0].keys():
            if key.endswith('_graphs'):
                batched_data[key] = dgl.batch([s[key] for s in samples])
            # elif key.endswith('_node_feature'):
            #     batched_data[key] = torch.concat([s[key] for s in samples], dim=0)
            # elif key.endswith('_edge_feature'):
            #     batched_data[key] = torch.concat([s[key] for s in samples], dim=0)
            elif key.endswith('_r_node') or key.endswith('_i_node') or key.endswith('_d_node'):
                batched_data[key] = torch.concat([s[key] for s in samples], dim=0)
            elif key.endswith('_r2r_edge') or key.endswith('_i2i_edge') or key.endswith('_d2d_edge'):
                batched_data[key] = torch.concat([s[key] for s in samples], dim=0)
            elif key.endswith('_var'):
                batched_data[key] = torch.stack([s[key].reshape(-1) for s in samples], dim=0)
            elif key.endswith('_smiles'):
                batched_data[key] = [s[key] for s in samples]
            elif key == 'target':
                batched_data[key] = torch.stack([s[key].reshape(-1) for s in samples], dim=0)
        return batched_data

    @staticmethod
    def unwrapper(device='cpu', **batched_data):
        for key in batched_data.keys():
            if key.endswith('_graphs'):
                batched_data[key] = batched_data[key].to(device=device)
            elif key.endswith('_r_node') or key.endswith('_i_node') or key.endswith('_d_node'):
                batched_data[key] = batched_data[key].float().to(device=device)
            elif key.endswith('_r2r_edge') or key.endswith('_i2i_edge') or key.endswith('_d2d_edge'):
                batched_data[key] = batched_data[key].float().to(device=device)
            elif key.endswith('_var'):
                batched_data[key] = batched_data[key].float().to(device=device)
            elif key == 'target':
                batched_data[key] = batched_data[key].float().to(device=device)
        return batched_data