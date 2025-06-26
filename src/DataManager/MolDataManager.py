import os
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from dgl.data.utils import save_graphs, load_graphs
from sklearn.model_selection import train_test_split

from .GraphGenerator.MolGraphGenerator import MolGraphGenerator
from .Dataset.GraphDataset import GraphDataset, GraphDataset_withSolv, GraphDataset_v1p3

from D4CMPP.src.utils.scaler import Scaler

import hashlib


class MolDataManager:
    """The class for the management of the molecular data, preparing the dataset, and dataloader
    
    This class is intended to provide the dataset and dataloader for the training, validation, and test.
    This class works mainly in two steps.

    1. call the init_data() to load the data from the csv file and prepare the graph.
        1-1. the load_data() will load the data from the csv file, saving the smiles and target values.
        1-2. the prepare_graph() will prepare the graph from the saved smiles, trying to load the graph first, then generate the graph if it does not exist.

    2. call the get_Dataloaders() to provide the dataloader for the training, validation, and test.
        2-1. the prepare_dataset() will drop the invalid data and initialize the dataset with generated graphs and target values.
        2-2. the split_data() will split the dataset into train, val, and test. if the set is given, the dataset will be split according to the set.
        2-3. the init_Dataloader() will initialize the dataloader based on the set and partial_train.    
    """
    def __init__(self, config):
        self.config = config
        self.data = config["data"]
        self.target = config["target"]
        self.scaler = Scaler(config.get("scaler","identity"))
        self.explicit_h_columns = config.get("explicit_h_columns",[])
        self.molecule_columns = ['compound']
        self.molecule_graphs={col:[] for col in self.molecule_columns}
        self.molecule_smiles = {}
        self.valid_smiles = {}
        self.random_seed = config.get("split_random_seed",42)
        self.set = None
        self.import_others()
        config.update({"node_dim":self.gg.node_dim, "edge_dim":self.gg.edge_dim})

    # Import the graph generator and dataset
    def import_others(self):
        self.graph_type = 'mol'
        self.gg =MolGraphGenerator()
        self.dataset =GraphDataset
        self.unwrapper = self.dataset.unwrapper

    # Initialize the temporary data for the prediction
    def init_temp_data(self,smiles_list):
        self.molecule_smiles['compound'] = np.array(smiles_list)
        self.valid_smiles['compound'] = np.array(smiles_list)
        self.molecule_graphs={col:[] for col in self.molecule_columns}
        self.target_value = np.zeros((len(smiles_list),self.config["target_dim"]))
        self.gg.verbose= False
        self.generate_graph('compound')
        return self.valid_smiles

    # (main function) Initialize the data from the csv file based on the configuration
    def init_data(self):
        self.load_data()
        self.prepare_graph()

    def load_csv(self):
        try:
            self.df = pd.read_csv(os.path.join(self.config['DATA_PATH']),encoding='utf-8')
        except:
            try:
                self.df = pd.read_csv(os.path.join(self.config['DATA_PATH']),encoding='cp949')
            except:
                self.df = pd.read_csv(os.path.join(self.config['DATA_PATH']),encoding='euc-kr')

    # Load the data from the csv file, saving the smiles and target values
    def load_data(self):
        self.load_csv()
        for col in self.molecule_columns:
            if col in self.df.columns:
                self.molecule_smiles[col] = np.array(list(self.df[col]))
                self.valid_smiles[col] = np.array(list(self.df[col]))
            else:
                raise ValueError(f"{col} is not in the dataframe")
        if torch.tensor(self.df[self.target].values).dim() == 1:
            self.target_value = self.scaler.fit_transform(torch.tensor(self.df[self.target].values).float().unsqueeze(1))
        else:
            self.target_value = self.scaler.fit_transform(torch.tensor(self.df[self.target].values).float())
        self.config.update({"target_dim":self.target_value.shape[1]})
        self.set = self.df.get("set",None)

    # Prepare the graph from the smiles, trying to load the graph first, then generate the graph if it does not exist
    def prepare_graph(self):
        for col in self.molecule_columns:
            if os.path.exists(self.get_graphs_path(col)):
                self.load_graphs(col)
            else:
                self.generate_graph(col)
                self.save_graphs(col)

    # Generate the graph from the smiles. the data failed to generate the graph will be saved as an empty graph
    def generate_graph(self,col,graphs=None):
        invalids = []
        if graphs is not None:
            for smi,g in zip(self.molecule_smiles[col],graphs):
                self.molecule_graphs[col].append(g)
                self.valid_smiles[col].append(smi)
            return
        if len(self.molecule_smiles[col]) < 10:
            iterator = self.molecule_smiles[col]
        else:
            iterator = tqdm(self.molecule_smiles[col], desc=f"Generating {col} graphs")
        for smi in iterator:
            graphgenerator = self.gg
            if col == 'solvent' and hasattr(self,'gg_solv'):
                graphgenerator = self.__getattribute__('gg_solv')
            else:
                graphgenerator = self.gg
            try:
                g=graphgenerator.get_graph(smi, explicit_h = col in self.explicit_h_columns, **self.config)
            except Exception as e:
                g = graphgenerator.get_empty_graph() # save the empty graph instead of the failed graph
                invalids.append(pd.DataFrame({'smiles':[smi],'type':[col],'reason':[str(e)]}))

            if g.number_of_nodes()==0:
                invalid_index = np.array(self.valid_smiles[col])==smi
                for _col in self.molecule_columns:
                    self.valid_smiles[_col] = self.valid_smiles[_col][np.where(~invalid_index)]
                if hasattr(self, 'numeric_inputs'):
                    for _col in self.numeric_inputs:
                        self.numeric_inputs[_col] = self.numeric_inputs[_col][np.where(~invalid_index)]
            self.molecule_graphs[col].append(g)
        if len(invalids)>0:
            invalids = pd.concat(invalids) 
            invalids.to_csv("graph_error.csv",index=False) # save the invalid smiles
        
    def get_graphs_path(self,col):
        file_name = self.data+"_"+col+"_"+self.graph_type
        if col in self.explicit_h_columns:
            file_name += "_explicitH"
        file_name += ".bin"
        file_path = os.path.join(self.config['GRAPH_DIR'],file_name)
        return file_path

    # Save the generated graphs
    def save_graphs(self,col):
        hash_smiles = torch.tensor([hash(smi) for smi in self.molecule_smiles[col]]) # hash the smiles to avoid the mismatch of the graphs and smiles
        save_graphs(self.get_graphs_path(col), self.molecule_graphs[col],{'smiles':hash_smiles})

    # Load the saved graphs
    def load_graphs(self,col):
        self.molecule_graphs[col],molecule_smiles= load_graphs(self.get_graphs_path(col))
        if len(self.molecule_graphs[col]) != len(self.molecule_smiles[col]):    
            raise ValueError("Graphs and smiles are not matched")
        for i in range(len(self.molecule_graphs[col])):
            if hash(self.molecule_smiles[col][i]) != int(molecule_smiles['smiles'][i]): # check the hash of the smiles
                print(hash(self.molecule_smiles[col][i]),int(molecule_smiles['smiles'][i]), self.molecule_smiles[col][i])
                raise ValueError("Graphs and smiles are not matched")

    # Drop the data with the empty graph
    def drop_none_graph(self):
        _masks = []
        for col in self.molecule_columns:
            _masks.append(np.array([g.number_of_nodes()>0 for g in self.molecule_graphs[col]]))
        _masks = np.stack(_masks)
        mask = np.all(_masks,axis=0)
        self.masking(mask)

    # Drop the data with the nan target
    def drop_nan_value(self):
        mask = np.array(~torch.isnan(torch.tensor(self.target_value)).all(dim=1))
        self.masking(mask)

    # Mask the data
    def masking(self,mask):
        self.target_value = self.target_value[mask]
        if self.set is not None:
            self.set = self.set[mask]
        for col in self.molecule_columns:
            self.molecule_smiles[col] = self.molecule_smiles[col][mask]
            self.molecule_graphs[col] = [g for i,g in enumerate(self.molecule_graphs[col]) if mask[i]]

    # drop the invalid data and initialize the dataset
    def prepare_dataset(self,temp=False):
        self.drop_none_graph()
        if not temp:
            self.drop_nan_value()
        self.whole_dataset = self.init_dataset()

    def init_dataset(self):
        return self.dataset(self.molecule_graphs['compound'], self.target_value, self.molecule_smiles['compound'])
        
    # Split the dataset into train, val, and test. if the set is given, the dataset will be split according to the set
    def split_data(self):
        if self.set is None:
            if len(self.whole_dataset) < 10:
                raise ValueError("The dataset is too small to split. Please provide a larger dataset.")
            train_dataset, test_dataset = train_test_split(self.whole_dataset, test_size=0.1, random_state=self.random_seed)
            train_dataset, val_dataset = train_test_split(train_dataset, test_size=1/9, random_state=self.random_seed)
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset(),self.dataset(),self.dataset()
            self.train_dataset.reload(list(zip(*train_dataset)))
            self.val_dataset.reload(list(zip(*val_dataset)))
            self.test_dataset.reload(list(zip(*test_dataset)))
        else:
            if len([s for s in self.set if s=="train"])==0:
                raise ValueError("The set must have 'train'")
            self.train_dataset = self.whole_dataset.get_subDataset([i for i,s in enumerate(self.set) if s=="train"])
            if len([s for s in self.set if s=="val"])==0:
                raise ValueError("The set must have 'val'")
            self.val_dataset = self.whole_dataset.get_subDataset([i for i,s in enumerate(self.set) if s=="val"])
            if len([s for s in self.set if s=="test"])==0:
                self.test_dataset = self.val_dataset
            else:
                self.test_dataset = self.whole_dataset.get_subDataset([i for i,s in enumerate(self.set) if s=="test"])
        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    # Initialize the dataloader
    def init_Dataloader(self,dataset=None,partial_train=1.0, shuffle=False):
        if dataset is None:
            dataset = self.whole_dataset
        if partial_train and partial_train<1.0 and partial_train>0:
            dataset = dataset.get_subDataset(np.random.choice(len(self.train_dataset),int(len(self.train_dataset)*partial_train),replace=False))
            print(f"Partial Training: {len(self.train_dataset)}")

        return DataLoader(dataset, 
                        batch_size=self.config.get('batch_size',32), 
                        shuffle=shuffle,
                        collate_fn=self.dataset.collate, 
                        pin_memory=self.config.get('pin_memory',True),
                        num_workers=self.config.get('num_workers',0))

    # (main function) Provide the dataloader for the training, validation, and test
    def get_Dataloaders(self, temp=False):
        self.prepare_dataset(temp=temp)
        if temp:
            loader = self.init_Dataloader(shuffle=False)
            return loader
        else:
            self.split_data()
            self.train_loader = self.init_Dataloader(self.train_dataset,partial_train=self.config.get('partial_train',1.0),shuffle=self.config.get('shuffle',True))
            self.val_loader = self.init_Dataloader(self.val_dataset)
            self.test_loader = self.init_Dataloader(self.test_dataset)

            return self.train_loader, self.val_loader, self.test_loader
  

class MolDataManager_withSolv(MolDataManager):
    """The class for the management of the molecular data with the solvent, preparing the dataset, and dataloader"""
    def __init__(self, config):
        super().__init__(config)
        self.molecule_columns.append('solvent')
        self.molecule_graphs={col:[] for col in self.molecule_columns}

    def import_others(self):
        self.graph_type = 'mol'
        self.gg =MolGraphGenerator()
        self.gg_solv =MolGraphGenerator()
        self.dataset =GraphDataset_withSolv
        self.unwrapper = self.dataset.unwrapper
        
    def init_temp_data(self,smiles,solvents):
        self.molecule_graphs={col:[] for col in self.molecule_columns}
        self.molecule_smiles={'compound':np.array(smiles),'solvent':np.array(solvents)}
        self.valid_smiles={'compound':np.array(smiles),'solvent':np.array(solvents)}
        self.target_value = np.zeros((len(smiles),self.config["target_dim"]))
        self.gg.verbose= False
        self.generate_graph('compound')
        self.generate_graph('solvent')
        return self.valid_smiles

    def init_dataset(self):
        return self.dataset(self.molecule_graphs['compound'], self.molecule_graphs['solvent'], self.target_value, self.molecule_smiles['compound'], self.molecule_smiles['solvent'])

class MolDataManager_v1p3(MolDataManager):
    def __init__(self, config):
        super().__init__(config)

        self.molecule_columns = config.get("molecule_columns", ['compound'])
        self.numeric_input_columns = config.get("numeric_input_columns", [])

        self.molecule_graphs = {col: [] for col in self.molecule_columns}

    def import_others(self):
        self.graph_type = 'mol'
        self.gg =MolGraphGenerator()
        self.dataset =GraphDataset_v1p3
        self.unwrapper = self.dataset.unwrapper

    def init_temp_data(self, *args, **kwargs):
        if len(args)+len(kwargs) != len(self.molecule_columns) + len(self.numeric_input_columns):
            raise ValueError(f"Please provide the smiles for {self.molecule_columns} and numeric input for {self.numeric_input_columns}.")
        if len(args) > 0:
            print(f"Positional arguments are provided. Note that arguments should be in the order of {self.molecule_columns} and {self.numeric_input_columns}.")
            if len(args) == len(self.molecule_columns) + len(self.numeric_input_columns):
                kwargs = {self.molecule_columns[i]: args[i] for i in range(len(self.molecule_columns))}
                kwargs.update({self.numeric_input_columns[i]: args[i + len(self.molecule_columns)] for i in range(len(self.numeric_input_columns))})
            elif len(args) == len(self.molecule_columns):
                kwargs.update({self.molecule_columns[i]: args[i] for i in range(len(self.molecule_columns))})
            elif len(args) == len(self.numeric_input_columns):
                kwargs.update({self.numeric_input_columns[i]: args[i] for i in range(len(self.numeric_input_columns))})
            else:
                raise ValueError(f"Please provide the smiles for {self.molecule_columns} and numeric input for {self.numeric_input_columns}.")
        if len(kwargs) >0:
            for k in kwargs.keys():
                if k not in self.molecule_columns and k not in self.numeric_input_columns:
                    raise ValueError(f"Unknown key {k}. Please provide the smiles for {self.molecule_columns} or numeric input for {self.numeric_input_columns}.")
            for k in self.molecule_columns:
                if k not in kwargs:
                    raise ValueError(f"Please provide the smiles for {k}.")
            for k in self.numeric_input_columns:
                if k not in kwargs:
                    raise ValueError(f"Please provide the numeric input for {k}.")
        for k in kwargs.keys():
            if type(kwargs[k]) is not list :
                kwargs[k] = [kwargs[k]]
        inputs = {col: kwargs[col] for col in self.molecule_columns if col in kwargs}

        self.molecule_smiles = {col: np.array(inputs[col]) for col in self.molecule_columns}
        self.valid_smiles = {col: np.array(inputs[col]) for col in self.molecule_columns}
        self.numeric_inputs = {col: np.array(kwargs[col]) for col in self.numeric_input_columns if col in kwargs}

        self.molecule_graphs = {col: [] for col in self.molecule_columns}
        self.target_value = np.zeros((len(self.molecule_smiles[self.molecule_columns[0]]), self.config["target_dim"]))
        self.gg.verbose = False
        for col in self.molecule_columns:
            self.generate_graph(col)
        result = self.valid_smiles.copy()
        result.update(self.numeric_inputs)
        return result

        # self.molecule_smiles['compound'] = np.array(smiles_list)
        # self.valid_smiles['compound'] = np.array(smiles_list)
        # self.molecule_graphs={col:[] for col in self.molecule_columns}
        # self.target_value = np.zeros((len(smiles_list),self.config["target_dim"]))
        # self.gg.verbose= False
        # self.generate_graph('compound')
        # return self.valid_smiles

    def init_dataset(self):
        if len(self.numeric_input_columns) > 0:
            if hasattr(self, 'numeric_inputs') and self.numeric_inputs is not None:
                if len(self.numeric_input_columns) != len(self.numeric_inputs):
                    raise ValueError(f"Please provide the numeric input for {self.numeric_input_columns}.")
                numeric_inputs = self.numeric_inputs
            else:
                numeric_inputs = {}
                for col in self.numeric_input_columns:
                    if col in self.df.columns:
                        numeric_inputs[col] = torch.tensor(self.df[col].values, dtype=torch.float32)
                    else:
                        print(f"Warning: {col} is not in the dataframe, initializing with zeros.")
                        numeric_inputs[col] = torch.zeros((len(self.molecule_smiles[self.molecule_columns[0]]), 1), dtype=torch.float32)
        else:
            numeric_inputs = None
        return self.dataset(graphs = self.molecule_graphs,
                            smiles = self.molecule_smiles,
                            target = self.target_value,
                            numeric_inputs = numeric_inputs,)

    def split_data(self):
        if self.set is None:
            if len(self.whole_dataset) < 10:
                raise ValueError("The dataset is too small to split. Please provide a larger dataset.")
            train_dataset, test_dataset = train_test_split(self.whole_dataset, test_size=0.1, random_state=self.random_seed)
            train_dataset, val_dataset = train_test_split(train_dataset, test_size=1/9, random_state=self.random_seed)
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset(),self.dataset(),self.dataset()
            self.train_dataset.reload(train_dataset)
            self.val_dataset.reload(val_dataset)
            self.test_dataset.reload(test_dataset)
        else:
            import copy
            self.train_dataset,self.val_dataset,self.test_dataset = copy.deepcopy(self.whole_dataset),copy.deepcopy(self.whole_dataset),copy.deepcopy(self.whole_dataset)

            self.train_dataset.subDataset([i for i,s in enumerate(self.set) if s=="train"])
            self.val_dataset.subDataset([i for i,s in enumerate(self.set) if s=="val"])
            self.test_dataset.subDataset([i for i,s in enumerate(self.set) if s=="test"])
        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

def hash(s):
    md5_hash = hashlib.md5()
    md5_hash.update(s.encode('utf-8'))
    hex_digest = md5_hash.hexdigest()
    decimal_hash = int(hex_digest, 16) % (10**12+7)
    
    return decimal_hash
