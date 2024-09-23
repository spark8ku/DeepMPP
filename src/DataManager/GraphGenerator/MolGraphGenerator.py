import dgl
import torch
import rdkit.Chem as Chem
import numpy as np
import traceback
from D4CMPP.src.utils.featureizer import get_atom_features, get_bond_features, InvalidAtomError

class MolGraphGenerator:
    def __init__(self):
        self.af = get_atom_features
        self.bf = get_bond_features
        self.set_feature_dim()

    # Set the feature dimensions by generating a dummy graph
    def set_feature_dim(self):
        self.node_dim = self.af(Chem.MolFromSmiles('C')).shape[1]
        self.edge_dim = self.bf(Chem.MolFromSmiles('C-C')).shape[1]

    # Get the graph from the SMILES
    def get_graph(self,smi,**kwargs):
        mol = Chem.MolFromSmiles(smi)
        
        if mol.GetNumAtoms() == 1:
            mol = Chem.AddHs(mol)
        if mol is None: 
            raise Exception("Invalid SMILES: failed to generate mol object")
        g = self.generate_graph(mol)
        g = self.add_feature(g,mol)
        
        return g
    
    def get_empty_graph(self):
        return dgl.graph(([],[]))

    # Add the features to the graph
    def add_feature(self, g, mol):
        atom_feature = self.af(mol)
        g.ndata['f'] = torch.tensor(atom_feature).float()

        bond_feature = self.bf(mol)
        edata = torch.tensor(bond_feature).float()
        edata = torch.cat([edata,edata],dim=0)
        g.edata['f'] = edata
        return g
    
    # Generate the graph from the molecule object
    def generate_mol_graph(self,mol):
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 1:
            return ([0],[0])
        src, dst = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.append(start)
            dst.append(end)

        return (src+dst, dst+src)
    
    def generate_graph(self,mol):
        mol_data = self.generate_mol_graph(mol)            
        g = dgl.graph(mol_data, num_nodes=mol.GetNumAtoms())
        return g