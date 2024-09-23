from .MolGraphGenerator import MolGraphGenerator
import dgl
import torch
import rdkit.Chem as Chem
import traceback
from D4CMPP.src.utils.featureizer import get_atom_features, get_bond_features_conformer, triplet_feature, InvalidAtomError, get_conformer

    
class TripletGraphGenerator(MolGraphGenerator):
    def __init__(self, verbose=False, xyz_path=None):
        self.af = get_atom_features
        self.bf = get_bond_features_conformer
        self.tf = triplet_feature
        self.set_feature_dim()
        self.verbose = verbose
        self.xyz_path = xyz_path

    def set_feature_dim(self):
        mol = Chem.MolFromSmiles('CCCC')
        conf = get_conformer(mol, do_optimize=True, xyz_path=self.xyz_path)
        self.node_dim = self.af(mol).shape[1]
        self.edge_dim = self.bf(mol,conf,self.xyz_path).shape[1]

        t = self.tf(mol, conf, 0, 1, 2)
        self.triplet_dim = t.shape[0]

    def generate_graph(self,mol):

        num_atoms = mol.GetNumAtoms()
        mol_data = self.generate_mol_graph(mol)
        conf = get_conformer(mol)
        if conf is None:
            raise Exception("No conformer")

        src_atms, dst_atms = mol_data
        triplet_src, triplet_dst = [], []
        triplet_features = []
        for b in mol.GetBonds():
            start, end = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            atom1 = mol.GetAtomWithIdx(start)
            for neigh1 in atom1.GetNeighbors():
                if neigh1.GetIdx() == end:
                    continue
                b1 = mol.GetBondBetweenAtoms(neigh1.GetIdx(), start)
                triplet_src.append(b.GetIdx())
                triplet_dst.append(b1.GetIdx())
                triplet_features.append(torch.tensor(self.tf(mol, conf, end, start, neigh1.GetIdx())))

            atom2 = mol.GetAtomWithIdx(end)
            for neigh2 in atom2.GetNeighbors():
                if neigh2.GetIdx() == start:
                    continue
                b2 = mol.GetBondBetweenAtoms(neigh2.GetIdx(), end)
                triplet_src.append(b.GetIdx())
                triplet_dst.append(b2.GetIdx())
                triplet_features.append(torch.tensor(self.tf(mol, conf, start, end, neigh2.GetIdx())))

        # adding dummy nodes to match size of line graph with atom graph
        num_bonds = len(src_atms)//2
        triplet_src += [i+num_bonds for i in triplet_src]
        triplet_dst += [i+num_bonds for i in triplet_dst]
        triplet_features += triplet_features

        graph_data = {
            ('atom', 'bond', 'atom'): mol_data,
            ('bond', 'triplet', 'bond'): (triplet_src, triplet_dst),
        }
        g = dgl.heterograph(graph_data)
        if len(triplet_features) == 0:
            g.edges['triplet'].data['f'] = torch.zeros((0, self.triplet_dim)).float()
        else:
            g.edges['triplet'].data['f'] = torch.stack(triplet_features).float()
        return g
    
    def get_graph(self, smi, **kwargs):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: 
            raise Exception("Invalid SMILES: failed to generate mol object")
        if mol.GetNumAtoms() <= 2: 
            raise Exception("Invalid SMILES: too few atoms")
        # try:
        g = self.generate_graph(mol)
        atom_feature = self.af(mol)
        bond_feature = self.bf(mol)

        g.nodes['atom'].data['f'] = torch.tensor(atom_feature).float()

        edata = torch.tensor(bond_feature).float()
        edata = torch.cat([edata,edata],dim=0)
        g.edges['bond'].data['f'] = edata
        # except InvalidAtomError:
        #     if self.verbose:
        #         print("Invalid atom error")
        #     return dgl.graph(([],[]))
        # except:
        #     if self.verbose:
        #         print(smi)
        #     print(traceback.format_exc())
        #     return dgl.graph(([],[]))
        return g
    