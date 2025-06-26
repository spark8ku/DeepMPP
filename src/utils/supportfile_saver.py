import pandas as pd
import yaml
from . import PATH

def save_funtionalgroup_csv(file_path,save_path):
    """
    save the functional group csv file
    """
    df = pd.read_csv(file_path)
    df.to_csv(save_path+'/functional_group.csv',index=False)

def save_network_refer(file_path,save_path):
    """
    save the network refer yaml file
    """
    with open(file_path, "r") as file:
        network_refer = yaml.safe_load(file)
    with open(save_path+'/network_refer.yaml', "w") as file:
        yaml.dump(network_refer, file)

def save_network_module(file_path,save_path):
    """
    save the network module
    """
    with open(file_path, "r") as file:
        network_module = file.read()
    with open(save_path+'/network.py', "w") as file:
        file.write(network_module)

def save_config(config):
    """
    save the configuration file
    """
    with open(config['MODEL_PATH']+'/config.yaml', "w") as file:
        yaml.dump(config, file)

def save_additional_files(config):
    """
    save the additional files
    """
    if "sculptor_s" in config:
        FRAG_REF = PATH.get_frag_ref_path(config)
        save_funtionalgroup_csv(FRAG_REF,config['MODEL_PATH'])

    NET_REF = PATH.get_network_refer(config)
    save_network_refer(NET_REF,config['MODEL_PATH'])

    NET_MODULE = PATH.get_NET_DIR(config)+'/'+config['network']+'.py'
    save_network_module(NET_MODULE,config['MODEL_PATH'])
    
    save_config(config)
