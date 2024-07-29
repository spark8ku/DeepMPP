import numpy as np
import os
import hashlib
import re

import rdkit.Chem as Chem
from rdkit.Chem import AllChem, rdMolTransforms

class InvalidAtomError(Exception):
    pass

def get_graph_features(mol):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=1024)).astype(float)

def get_atom_features(mol):
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_feature(atom))
    return np.concatenate(atom_features, axis=0)

def get_atom_features_and_graph_feature(mol, graph_featurizer=None):
    atom_features = get_atom_features(mol)

    if graph_featurizer is not None:
        graph_feature = graph_featurizer(mol)
    else:
        graph_feature = get_graph_features(mol)
    graph_feature = np.tile(graph_feature, (atom_features.shape[0],1))
    return atom_features, graph_feature

def get_bond_features(mol):
    bond_features = []
    for bond in mol.GetBonds():
        bond_features.append(bond_feature(bond))
    if len(bond_features) == 0:
        return np.zeros((1, 10))
    return np.concatenate(bond_features, axis=0)


def bond_feature(bond):
    features = np.concatenate([bond_type(bond),
                               bond_stereo(bond)
                            ], axis=0).reshape(1,-1)
    
    return features

def generate_conformer(mol, save_conf = True, xyz_path = None):
    smiles = Chem.MolToSmiles(mol)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.EmbedMultipleConfs(mol, 10) 
    li = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=5000)
    eg_li = [e if s==0 else np.inf for s,e in li ]
    idx = eg_li.index(min(eg_li))
    conf = mol.GetConformer(idx)
    if save_conf:
        save_xyz(smiles, conf, xyz_path)
    return conf

def get_conformer(mol, do_optimize=False, xyz_path = None):
    smiles = Chem.MolToSmiles(mol)
    if mol is None:
        return None
    
    if xyz_path is not None:
        pos, atoms = load_xyz(smiles, xyz_path)
    if pos is None and do_optimize:
        return generate_conformer(mol, save_conf=True ,xyz_path = xyz_path)
    elif pos is None:
        return None
    
    conf = Chem.Conformer()
    for i in range(len(atoms)):
        conf.SetAtomPosition(i, pos[i])
    return conf
    
def save_xyz(smiles, conf, xyz_path):
    mol = Chem.MolFromSmiles(smiles)
    m = hashlib.sha256()
    m.update(smiles.encode('utf-8'))
    name = m.hexdigest()
    result=f'{mol.GetNumAtoms()} #{smiles}\n'
    result+=f'# {Chem.GetFormalCharge(mol)} 1\n'
    for i, atom in enumerate(mol.GetAtoms()):
        positions = conf.GetAtomPosition(i)
        result+=f'{atom.GetSymbol()} {positions.x} {positions.y} {positions.z}\n'
    with open(f'{xyz_path}/{name}.xyz','w') as t:
        t.write(result)
    return name

def load_xyz(smiles,xyz_path):
    m = hashlib.sha256()
    m.update(smiles.encode('utf-8'))
    name = m.hexdigest()
    path = os.path.join(xyz_path,name+'.xyz')
    if not os.path.exists(path):
        return None, None
    m = hashlib.sha256()
    m.update(smiles.encode('utf-8'))
    name = m.hexdigest()
    with open(path,'r') as t:
        file= t.read()
    xyzs = file.split("\n")[2:]
    pos = []
    atoms = []
    for xyz in xyzs:
        space = '\t' if '\t' in xyz else ' ' 
        a = xyz.split(space)
        if len(a)!=4 and len(a)!=5: break
        pos.append(np.array([to_float(a[1]),to_float(a[2]),to_float(a[3])]))
        atoms.append(a[0])
    return np.stack(pos), atoms

def to_float(a):
    try:
        return float(a)
    except:
        if '*^' in a:
            return float(a.replace('*^','e'))
        else:
            return 0


def get_bond_features_conformer(mol, conf=None, xyz_path=None):
    if conf is None:
        conf = get_conformer(mol, do_optimize=True, xyz_path=xyz_path)

    bond_features = []
    for bond in mol.GetBonds():
        bond_features.append(bond_feature_conformer(bond, conf))
    return np.concatenate(bond_features, axis=0)

def bond_feature_conformer(bond, conf):
    features = np.concatenate([bond_type(bond),
                               bond_stereo(bond),
                               bond_length(bond, conf)
                            ], axis=0).reshape(1,-1)
    return features

def bond_length(bond, conf):
    src_pos = conf.GetAtomPosition(bond.GetBeginAtomIdx())
    dst_pos = conf.GetAtomPosition(bond.GetEndAtomIdx())
    return np.array([np.linalg.norm(src_pos-dst_pos)]).astype(float)

def atom_feature(atom):
    features = np.concatenate([atom_symbol_HNums(atom),
                                atom_degree(atom),
                                atom_Aroma(atom),
                                atom_Hybrid(atom),
                                atom_ring(atom),
                                atom_FC(atom)
                            ], axis=0).reshape(1,-1)
    
    return features

def bond_type(bond):
    return np.array(one_of_k_encoding(str(bond.GetBondType()),['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'])).astype(int)

def bond_stereo(bond):
    return np.array(one_of_k_encoding(str(bond.GetStereo()),['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE'])).astype(int)

def atom_symbol_HNums(atom):
    
    return np.array(one_of_k_encoding(atom.GetSymbol(),
                                      ['C', 'N', 'O','S', 'H', 'F', 'Cl', 'Br', 'I', 'Se', 'Te','Si','P','B','Sn','Ge'])+
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))


def atom_degree(atom):
    return np.array(one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5 ,6])).astype(int) 


def atom_Aroma(atom):
    return np.array([atom.GetIsAromatic()]).astype(int)


def atom_Hybrid(atom):
    return np.array(one_of_k_encoding(str(atom.GetHybridization()),['S','SP','SP2','SP3','SP3D','SP3D2'],'S')).astype(int)


def atom_ring(atom):
    return np.array([atom.IsInRing()]).astype(int)


def atom_FC(atom):
    return np.array(one_of_k_encoding(atom.GetFormalCharge(), [-4,-3,-2,-1, 0, 1, 2, 3, 4])).astype(int)


def one_of_k_encoding(x, allowable_set,default=None):
    if x not in allowable_set:
        if default is not None:
            return list(map(lambda s: default == s, allowable_set))
        raise InvalidAtomError("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def triplet_feature(mol, conf, atom1, atom2, atom3):
    return np.array([rdMolTransforms.GetAngleRad(conf, atom1, atom2, atom3)]).astype(float)



