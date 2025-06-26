from rdkit import Chem
import torch
import dgl
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import traceback

from .MolGraphGenerator import MolGraphGenerator
from D4CMPP.src.utils.featureizer import InvalidAtomError
from D4CMPP.src.utils.sculptor import SubgroupSplitter

class ISAGraphGenerator(MolGraphGenerator):
    def __init__(self, frag_ref, sculptor_index):
        self.sculptor = SubgroupSplitter(frag_ref,
                                        get_index=True,
                                        split_order=sculptor_index[0],
                                        combine_rest_order=sculptor_index[1],
                                        absorb_neighbor_order=sculptor_index[2],
                                        overlapped_ring_combine=True
                                )
        self.r_node_dim = None
        self.i_node_dim = None
        self.d_node_dim = None
        self.r_edge_dim = None
        self.i_edge_dim = None
        self.d_edge_dim = None
        self.node_dim= None
        self.edge_dim = None
        super().__init__()


        self.set_feature_dim()
        self.node_dim = self.r_node_dim 
        self.edge_dim = self.r_edge_dim
        self.verbose = True

    def set_feature_dim(self):
        try:
            get_graph = self.get_graph('FC1CCCCC1CCCOCCC')
        except InvalidAtomError as e:
            if self.verbose:
                print("Invalid atom feature: ", e)

    def get_graph(self, smi, **kwargs):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: 
            raise Exception("Invalid SMILES: failed to generate mol object")
        if kwargs.get("explicit_h",False):
            mol = Chem.AddHs(mol)
        g = self.generate_graph(mol, **kwargs)
        atom_feature = self.af(mol)
        bond_feature = self.bf(mol)

        g.nodes['r_nd'].data['f'] = torch.tensor(atom_feature).float()
        if self.r_node_dim is None:
            self.r_node_dim = atom_feature.shape[1]

        edata = torch.tensor(bond_feature).float()
        edata = torch.cat([edata,edata],dim=0)
        g.edges['r2r'].data['f'] = edata
        if self.r_edge_dim is None:
            self.r_edge_dim = edata.shape[1]

        g.nodes['i_nd'].data['f'] = torch.zeros((g.number_of_nodes('i_nd'),1)).float()
        if self.i_node_dim is None:
            self.i_node_dim = 1
        if self.i_edge_dim is None:
            self.i_edge_dim = 0

        g.nodes['d_nd'].data['f'] = torch.zeros((g.number_of_nodes('d_nd'),1)).float()
        if self.d_node_dim is None:
            self.d_node_dim = 1
        if self.d_edge_dim is None:
            self.d_edge_dim = 0


        return g
    
    def generate_sub_graph(self, mol, frags):
        src, dst = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            for frag in frags:
                if start in frag:
                    if end in frag:
                        src.append(start)
                        dst.append(end)
                        break
                    else:
                        break
                elif end in frag:
                    if start in frag:
                        src.append(start)
                        dst.append(end)
                        break
                    else:
                        break
                    
        return (src+dst, dst+src)
        

    def generate_dot_graph(self, mol, frags, max_dist = 4):
        src, dst = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            for i,frag in enumerate(frags):
                if start in frag:
                    start_frag = i
                if end in frag:
                    end_frag = i
            if start_frag != end_frag:
                src.append(start_frag)
                dst.append(end_frag)
        if len(src) == 0:
            return ([],[]), []
        g = dgl.graph((src+dst, dst+src))
        dist = dgl.shortest_dist(g, root=None)
        src, dst, sd_dist = [], [], []
        for i in range(len(dist)):
            for j in range(i+1,len(dist)):
                if dist[i][j] <= max_dist and dist[i][j] > 0:
                    src.append(i)
                    dst.append(j)
                    sd_dist.append(dist[i][j])
                    
        ohe = OneHotEncoder(sparse_output=False, categories=[list(range(1,max_dist+1))])
        sd_dist = ohe.fit_transform(np.array(sd_dist+sd_dist).reshape(-1,1))
        return (src+dst, dst+src), sd_dist

    def generate_graph(self,mol, **kwargs):
        max_dist = kwargs.get('max_dist',4)
        num_atoms = mol.GetNumAtoms()
        mol_data = self.generate_mol_graph(mol)
        
        frag = self.sculptor.fragmentation_with_condition(mol)
        frag_data = self.generate_sub_graph(mol, frag) 

        i2d_src = []
        i2d_dst = []
        for i, f in enumerate(frag):
            for a in f:
                i2d_src.append(a)
                i2d_dst.append(i)

        if len(frag) ==1:
            dot_data = ([0],[0])
            dist = np.zeros((1,max_dist))
        else:
            dot_data, dist = self.generate_dot_graph(mol, frag, max_dist = max_dist)

        graph_data = {
            ('r_nd', 'r2r', 'r_nd'): mol_data,
            ('r_nd', 'r2i', 'i_nd'): (list(range(num_atoms)), list(range(num_atoms))),
            ('i_nd', 'i2i', 'i_nd'): frag_data,
            ('i_nd', 'i2d', 'd_nd'): (i2d_src, i2d_dst),
            ('d_nd', 'd2d', 'd_nd'): dot_data,
            ('d_nd', 'd2r', 'r_nd'): (i2d_dst, i2d_src),

        }


        g = dgl.heterograph(graph_data)
        g.edges['d2d'].data['f'] = torch.tensor(dist).float()
        if self.d_edge_dim is None:
            self.d_edge_dim = dist.shape[1]

        return g