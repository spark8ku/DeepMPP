import torch
import importlib
import os
import yaml
import pandas as pd
from D4CMPP.src.utils import PATH
from .NetworkManager import NetworkManager
    
class ISANetworkManager(NetworkManager):

    # Initialize the network
    def init_network(self):
        super().init_network()
        os.system(f"""cp "{self.config["FRAG_REF"]} "{self.config["MODEL_PATH"]}/functional_group.csv" """)

        
    def load_network(self, path):
        super().load_network(path)
        if os.path.exists(os.path.join(path,'functional_group.csv')):
            self.config["FRAG_REF"] = os.path.join(path,'functional_group.csv')
