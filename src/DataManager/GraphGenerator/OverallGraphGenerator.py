from .MolGraphGenerator import MolGraphGenerator
from rdkit import Chem
import torch

from D4CMPP.src.utils.featureizer import get_atom_features_and_graph_feature,get_bond_features

class OverallGraphGenerator(MolGraphGenerator):
    def __init__(self, graph_featurizer=None):
        self.af = get_atom_features_and_graph_feature
        self.bf = get_bond_features
        self.set_feature_dim()

    def set_feature_dim(self):
        a,g = self.af(Chem.MolFromSmiles('CCC'))
        self.node_dim = a.shape[1]
        self.graph_dim = g.shape[1]
        self.edge_dim = self.bf(Chem.MolFromSmiles('CCC')).shape[1]
    
    def add_feature(self, g, mol):
        atom_feature, graph_feature = self.af(mol)
        g.ndata['f'] = torch.tensor(atom_feature).float()
        g.ndata['g'] = torch.tensor(graph_feature).float()

        bond_feature = self.bf(mol)
        edata = torch.tensor(bond_feature).float()
        edata = torch.cat([edata,edata],dim=0)
        g.edata['f'] = edata
        return g