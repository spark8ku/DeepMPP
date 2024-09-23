import yaml
import os
import traceback
import itertools

from _main import run, set_config

def grid_generator(param_grid):
    keys, values = zip(*param_grid.items())
    
    for v in itertools.product(*values):
        yield dict(zip(keys, v))

config0 = {
    "data": None,
    "target": [],
    "network": None,

    "scaler": "standard",
    "optimizer": "Adam",

    "max_epoch": 2000,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0005,
    "lr_patience": 40,
    "early_stopping_patience": 100,

    "device": "cuda:0",
    "pin_memory": False,

}

hyperparameters = {
    "sculptor_index":[[4,0,0],[6,2,0],[9,5,6]],
    "hidden_dim": [64, 128, 256],
    "conv_layers": [4, 6, 8],
    "linear_layers": [2, 4,],
}

def grid_search(hyperparameters, **kwargs):    
    """
    Grid search for the hyperparameters tuning.

    Args:
        hyperparameters: dict
                        the hyperparameters to search. The keys are the hyperparameter names and the values are the list of values to search.
                        e.g. {'hidden_dim': [32, 64, 128], 'conv_layers': [4, 6, 8]}
                        if the given argments and keys in hyperparameters are duplicated, the given arguments will be ignored.

    Args for the training:
        (same as the train function)
        
    Args for the network:
        (same as the train function)
    
    --------------------------------------------
    Example:

        grid_search({'hidden_dim': [32, 64, 128], 'conv_layers': [4, 6, 8]}, data='data.csv', target=['target1','target2'], network='GCN',)
    
    """
    config0.update(kwargs)
    config = set_config(**config0)

    for i,hp in enumerate(grid_generator(hyperparameters)):
        try:
            _config = config.copy()
            _config.update(hp)
            grid_id = ""
            for key in hp.keys():
                grid_id += f"_{key},{hp[key]}"
            _config['MODEL_PATH'] = _config['MODEL_PATH'] + grid_id
            
            run(_config)
        except:
            print(traceback.format_exc())
            continue


if __name__ == "__main__":
    grid_search(hyperparameters,config0)