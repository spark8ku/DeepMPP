import torch
import importlib
import os, sys
import yaml
import pandas as pd
import numpy as np
import re
from D4CMPP.src.utils import PATH
import time
class NetworkManager:

    def __init__(self, config, tf_path=None, unwrapper=None, temp=False):
        
        self.config = config
        self.device = config.get('device', 'cpu')
        self.last_lr = config.get('learning_rate',0.001)
        self.unwrapper = unwrapper
        self.best_loss = float('inf')
        self.es_patience = config.get('early_stopping_patience',50)
        self.es_counter = 0
        self.state= "train"
        self.tf_path = tf_path
        self.schedulers = []
        self.temp = temp

        if self.tf_path:
            self.transferlearn_network()
        else:
            self.init_network()

        self.init_optimizer(config.get('lr_dict',{}) )
        self.init_scheduler()

    def set_unwrapper(self, unwrapper):
        self.unwrapper = unwrapper

    def get_net_module(self,model_path=None):
        if model_path is None:
            model_path = self.config['MODEL_PATH']
        if os.path.exists(os.path.join(model_path,'network.py')):
            sys.path.append(model_path)
            module = importlib.import_module('network')
            net = getattr(module, 'network')
            return net
        else:
            raise FileNotFoundError(model_path+"/network.py not found")

    # Initialize the network
    def init_network(self):
        module = self.get_net_module()
        self.network = module(self.config)
        self.network.to(device = self.device)
        print(f"#params: {sum(p.numel() for p in self.network.parameters() if p.requires_grad)}" )
        self.loss_fn = self.network.loss_fn

        if os.path.exists(os.path.join(self.config['MODEL_PATH'],'result','learning_curve.csv')):
            self.learning_curve = pd.read_csv(os.path.join(self.config['MODEL_PATH'],'result','learning_curve.csv'))
        else:
            self.learning_curve = None

        if os.path.exists(os.path.join(self.config['MODEL_PATH'],'final.pth')):
            self.load_params(os.path.join(self.config['MODEL_PATH'],'final.pth'))
        

    # Transfer learning
    def transferlearn_network(self):
        self.init_network()        
        self.load_params_transfer_learn(self.tf_path)

    # Initialize the optimizer
    def init_optimizer(self,lr_dict={}):        
        params = []
        for name, param in self.network.named_parameters():
            custom_lr = False
            for key in lr_dict.keys():
                for layer_name in name.split("."):
                    if layer_name == key:
                        custom_lr = True
                        break
                if custom_lr:
                    break
            if custom_lr:
                lr = lr_dict[key]
                print(f"Set {name} lr to {lr}")
            else:
                lr = self.config.get('learning_rate',0.001)
            
            params.append({'params': param, 'lr': lr})

        self.optimizer = getattr(torch.optim, self.config['optimizer'])(params,  
                                                                        lr=self.config.get('learning_rate',0.001), 
                                                                        weight_decay=self.config.get('weight_decay',0.0005)
                                                                        )

    # Initialize the scheduler
    def init_scheduler(self):
        self.schedulers.append( torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                    patience=self.config.get('lr_patience',10), 
                                                                    min_lr=self.config.get('min_lr',1e-7),
                                                                    factor=self.config.get('lr_plateau_decay',0.1),
                                                                    ))
        self.schedulers.append( torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                                    step_size=self.config.get('lr_step',40), 
                                                                    gamma=self.config.get('lr_step_decay',0.98)))
    
    "----------------------------------------------------------------------------------------------------------------------"
    "Below are the functions to manage the network during the training, and called by the Trainer class"

    def train(self):
        self.state = "train"
        self.network.train()

    def eval(self):
        self.state = "eval"
        self.network.eval()

    def get_lr(self):
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group['lr'])
        return np.mean(lrs)
        
    # One step of the training including forward and backward
    def step(self, loader, flag= False, **kargs):
        self.optimizer.zero_grad()
        x= self.unwrapper(*loader,device=self.device)
        y= x['target']
        x.update(kargs)
        y_pred = self.network(**x)
        if kargs.get('get_score',False) or kargs.get('get_feature',False):
            return y_pred
                
        loss = self.loss_fn(y_pred, y)
        if self.state == "train":
            loss.backward()
            self.optimizer.step()
        if flag:
            return y, y_pred.detach(), loss.detach().item(), x
        return y, y_pred.detach(), loss.detach().item()
    
    #  One step of the prediction including forward
    def predict(self,loader):
        self.optimizer.zero_grad()
        x= self.unwrapper(*loader,device=self.device)
        y= x['target']
        y_pred = self.network(**x)
        return y_pred, y

    def scheduler_step(self, val_loss=None):
        for scheduler in self.schedulers:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # save the best model and reset the early stopping counter
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss)
            self.es_counter = 0
        else:
            self.es_counter += 1

        # early stopping
        if self.es_counter > self.es_patience:
            self.load_best_checkpoint()
            return True

        # load the best model if the learning rate is reduced
        current_lr = self.get_lr()
        if current_lr<self.last_lr/2:
            self.last_lr=current_lr
            self.load_best_checkpoint()
            print("reducing learning rate to "+str(self.last_lr))
            
    def save_checkpoint(self, val_loss):
        self.save_params(os.path.join(self.config['MODEL_PATH'],"param_"+str(val_loss)+".pth"))

    def load_best_checkpoint(self):
        path = os.path.join(self.config['MODEL_PATH'],"param_"+str(self.best_loss)+".pth")
        if os.path.exists(path):
            self.load_params(path)
            return path
        else:
            return None

    def load_params(self, path):
        self.network.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

    def load_params_transfer_learn(self, tf_path):
        # first, load the pretrained network
        config = yaml.load(open(os.path.join(tf_path,'config.yaml'), 'r'), Loader=yaml.FullLoader)
        module = self.get_net_module(tf_path)
        pretrained_network = module(config)
        params = torch.load(os.path.join(tf_path,'final.pth'), map_location=self.device)
        pretrained_network.load_state_dict(params)

        # then, load params of pretrained network to the current network
        pretrained_dict = pretrained_network.state_dict()
        model_dict = self.network.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        for k in pretrained_dict.keys():
            if pretrained_dict[k].shape != model_dict[k].shape:
                print(f"Shape mismatch: {k}, {pretrained_dict[k].shape} {model_dict[k].shape}")
                pretrained_dict[k] = model_dict[k]
            else:
                print(f"Param {k} loaded")
        model_dict.update(pretrained_dict)
        self.network.load_state_dict(model_dict)

    def save_params(self, path):
        torch.save(self.network.state_dict(), path)

    def dropout_on(self):
        for m in self.network.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()