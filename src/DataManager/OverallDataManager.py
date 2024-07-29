from .MolDataManager import MolDataManager
import torch
import os
import numpy as np
from D4CMPP.src.utils import PATH
from dgl.data.utils import save_graphs, load_graphs
from .GraphGenerator.OverallGraphGenerator import OverallGraphGenerator
from .Dataset.OverallGraphDataset import OverallGraphdataset,OverallGraphdataset_withSolv

class OverallDataManager(MolDataManager):
    def __init__(self, config):
        super(OverallDataManager, self).__init__(config)
        config.update({"graph_dim":self.gg.graph_dim})
        self.graph_features = {}

    def import_others(self):
        self.graph_type = 'mol'
        self.graph_type = 'overall'
        self.gg =OverallGraphGenerator()
        self.dataset =OverallGraphdataset
        self.unwrapper = self.dataset.unwrapper

    def save_graphs(self,col):
        overall_features = []
        new_graphs = []
        for g in self.molecule_graphs[col]:
            if g.number_of_nodes()==0:
                overall_features.append(torch.zeros(self.gg.graph_dim))
                new_graphs.append(g)
                continue
            a = torch.tensor(g.ndata['g'].detach().cpu()[0]).float()
            g = g.local_var()
            g.ndata.pop('g')
            new_graphs.append(g)
            overall_features.append(a)
        overall_features = torch.stack(overall_features)
        self.molecule_graphs[col] = new_graphs
        self.graph_features[col] = overall_features
        save_graphs(os.path.join(self.config['GRAPH_DIR'],self.data+"_"+col+"_"+self.graph_type+".bin"), self.molecule_graphs[col],{'overall':overall_features,'smiles':torch.tensor([hash(smi) for smi in self.molecule_smiles[col]])})


    def load_graphs(self,col):
        self.molecule_graphs[col],_= load_graphs(os.path.join(self.config['GRAPH_DIR'],self.data+"_"+col+"_"+self.graph_type+".bin"))
        self.graph_features[col] = _.get('overall')
        if len(self.molecule_graphs[col]) != len(self.molecule_smiles[col]):    
            raise ValueError("Graphs and smiles are not matched")
        for i in range(len(self.molecule_graphs[col])):
            if hash(self.molecule_smiles[col][i]) != _['smiles'][i]:
                raise ValueError("Graphs and smiles are not matched")

    def init_temp_data(self,smiles_list):
        self.set = None
        self.target_value = np.zeros((len(smiles_list),self.config["target_dim"]))
        self.gg.verbose= False
            
        for col in self.molecule_columns:
            self.molecule_smiles[col] = np.array(smiles_list)
            self.generate_graph(col)
            self.graph_features[col] = [g.ndata['g'].detach().cpu()[0] if g.number_of_nodes()>0 else torch.zeros(self.gg.graph_dim) for g in self.molecule_graphs[col] ]
        self.prepare_dataset(temp=True)
        return self.valid_smiles
    
    def init_dataset(self):
        return self.dataset(self.molecule_graphs['compound'], self.graph_features['compound'], self.target_value, self.molecule_smiles['compound'])

class OverallDataManager_withSolv(OverallDataManager):
    def __init__(self, config):
        super(OverallDataManager_withSolv, self).__init__(config)
        self.molecule_columns.append('solvent')

    def import_others(self):
        self.graph_type = 'mol'
        self.graph_type = 'overall'
        self.gg =OverallGraphGenerator()
        self.dataset =OverallGraphdataset_withSolv
        self.unwrapper = self.dataset.unwrapper

    def init_temp_data(self,smiles_list,solvents_list):
        self.set = None
        self.target_value = np.zeros((len(smiles_list),self.config["target_dim"]))
        self.gg.verbose= False
            
        for col in self.molecule_columns:
            self.molecule_smiles[col] = np.array(smiles_list) if col == 'compound' else np.array(solvents_list)
            self.generate_graph(col)
            self.graph_features[col] =[g.ndata['g'].detach().cpu()[0] if g.number_of_nodes()>0 else torch.zeros(self.gg.graph_dim) for g in self.molecule_graphs[col] ]

        self.prepare_dataset(temp=True)
        return self.valid_smiles

    def init_dataset(self):
        return self.dataset(self.molecule_graphs['compound'], self.graph_features['compound'], self.molecule_graphs['solvent'], self.graph_features['solvent'],  self.target_value, self.molecule_smiles['compound'], self.molecule_smiles['solvent'])
    
    