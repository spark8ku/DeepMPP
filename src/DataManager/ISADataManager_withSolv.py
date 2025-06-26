import numpy as np
import pandas as pd
import os
from dgl.data.utils import save_graphs, load_graphs

from .ISADataManager import ISADataManager
from .Dataset.ISAGraphDataset_withSolv import ISAGraphDataset_withSolv
from D4CMPP.src.utils import PATH

class ISADataManager_withSolv(ISADataManager):
    def __init__(self, config):
        super(ISADataManager_withSolv, self).__init__(config)
        self.molecule_columns.append('solvent')
        self.molecule_graphs={col:[] for col in self.molecule_columns}

    def import_others(self):
        super(ISADataManager_withSolv, self).import_others()
        self.dataset = ISAGraphDataset_withSolv
        self.unwrapper = self.dataset.unwrapper

    def init_temp_data(self,smiles_list,solvents_list, graph_id = None):
        self.molecule_smiles['compound'] = np.array(smiles_list)
        self.valid_smiles['compound'] = np.array(smiles_list)
        self.molecule_smiles['solvent'] = np.array(solvents_list)
        self.valid_smiles['solvent'] = np.array(solvents_list)
        self.molecule_graphs={col:[] for col in self.molecule_columns}
        self.set = None
        self.target_value = np.zeros((len(smiles_list),self.config["target_dim"]))
        self.gg.verbose= False

        if graph_id is not None:
            for col in self.molecule_columns:
                if col=="solvent": continue
                if os.path.exists("temp_graph_"+col+"_"+graph_id+".pkl"):
                    self.molecule_graphs[col],valid_smiles = load_graphs("temp_graph_"+col+"_"+graph_id+".pkl")
                    self.valid_smiles[col] = np.array(pd.read_csv("temp_graph_"+col+"_"+graph_id+".csv")[col])
        for col in self.molecule_columns:
            if len(self.molecule_graphs[col])>0: continue
            self.molecule_smiles[col] = np.array(smiles_list) if col == 'compound' else np.array(solvents_list)
            self.generate_graph(col)
            if graph_id is not None and col !="solvent":
                save_graphs("temp_graph_"+col+"_"+graph_id+".pkl",self.molecule_graphs[col])
                pd.DataFrame({col:self.valid_smiles[col]}).to_csv("temp_graph_"+col+"_"+graph_id+".csv",index=False)

        self.prepare_dataset(temp=True)
        return self.valid_smiles

    def init_dataset(self):
        return self.dataset(self.molecule_graphs['compound'], self.molecule_graphs['solvent'], self.target_value, self.molecule_smiles['compound'], self.molecule_smiles['solvent'])
    