import time
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from .TrainManager import Trainer

class ISATrainer(Trainer):
    
    def get_feature(self, network_manager, loader):
        "Provide the feature of the model. The model which is used for the training should contains the implementation of 'get_feature'."
        network_manager.eval()
        pred_result = {}
        with torch.no_grad():
            for loader in tqdm(loader, desc='Calculating features'):     
                torch.autograd.set_detect_anomaly(False)
                y_pred = network_manager.step(loader,get_feature=True)
                if pred_result:
                    for key in y_pred:
                        pred_result[key] = torch.cat([pred_result[key],y_pred[key]], dim=0)
                else:
                    pred_result = y_pred
        return pred_result