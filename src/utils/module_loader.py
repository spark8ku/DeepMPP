import sys
from D4CMPP.src import TrainManager
from D4CMPP.src import DataManager
from D4CMPP.src import NetworkManager
import importlib.util
import os
def load_train_manager(config):
    return _load_manager('TrainManager_PATH', 'train_manager_module', 'train_manager_class', load_default_train_manager)(config)
        
def load_data_manager(config):
    return _load_manager('DataManager_PATH', 'data_manager_module', 'data_manager_class', load_default_data_manager)(config)

def load_network_manager(config):
    return _load_manager('NetworkManager_PATH', 'network_manager_module', 'network_manager_class', load_default_network_manager)(config)

def _load_manager(path_key, module_key, class_key, default_manager):
    def load_manager(config):
        if path_key in config:
            if os.path.exists(os.path.join(config[path_key],config[module_key]+'.py')):
                return load_module(config[path_key], config[module_key], config[class_key])
        try:
            return default_manager(config[module_key], config[class_key])
        except:
            raise FileNotFoundError(config[path_key]+'/'+config[module_key]+'.py not found')

    return load_manager


def load_default_train_manager(module_name, class_name):
    return getattr(getattr(TrainManager, module_name), class_name)

def load_default_data_manager(module_name, class_name):
    return getattr(getattr(DataManager, module_name), class_name)

def load_default_network_manager(module_name, class_name):
    return getattr(getattr(NetworkManager, module_name), class_name)


def load_module(path, module, class_name):
    sys.path.append(path)
    spec2 = importlib.util.spec_from_file_location(module, f"{path}/{module}.py")
    module2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(module2)
    return getattr(module2, class_name)
