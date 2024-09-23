import torch
import importlib
import os
import yaml
import pandas as pd
from D4CMPP.src.utils import PATH
    
class NetworkManager:

    # 현재 작업 디렉토리를 얻어서 PYTHONPATH에 추가

    "The class where every operation of the network is managed"
    def __init__(self, config,tf=False, unwrapper=None):
        
        self.config = config
        self.device = config.get('device', 'cpu')
        self.last_lr = config.get('learning_rate',0.001)
        self.unwrapper = unwrapper
        self.best_loss = float('inf')
        self.es_patience = config.get('early_stopping_patience',50)
        self.es_counter = 0
        self.state= "train"
        self.tf = tf

        if self.config.get('LOAD_PATH',None) is not None:
            self.load_network(self.config['LOAD_PATH'])
        else:
            self.init_network()
        self.init_optimizer(config.get('lr_dict',{}) )
        self.init_scheduler()

    def set_unwrapper(self, unwrapper):
        self.unwrapper = unwrapper

    def get_net_module(self):
        if os.path.exists(self.config['MODEL_PATH']+"/network.py"):
            net_path = self.config['MODEL_PATH']+"/network"
        elif os.path.exists(os.path.join(self.config['NET_DIR'],self.config['network']+'.py')):
            net_path = os.path.join(self.config['NET_DIR'],self.config['network'])

        if os.path.isabs(net_path) and '/D4CMPP/' in net_path:
            net_path = 'D4CMPP'+net_path.split('D4CMPP')[-1]
        elif os.path.isabs(net_path):
            pwd = os.getcwd()
            print('@',pwd)
            print('@',net_path)
            if pwd in net_path:
                net_path = net_path.replace(pwd+'/','')
            else:
                raise Exception("Absolute path is not allowed for the network module")
        
        if net_path.startswith('./'):
            net_path=net_path.replace('./','')
        print(net_path)
        return importlib.import_module(net_path.replace('/','.')).network

    # Initialize the network
    def init_network(self):
        module = self.get_net_module()
        self.network = module(self.config)
        self.network.to(device = self.device)
        print(f"#params: {sum(p.numel() for p in self.network.parameters() if p.requires_grad)}" )
        self.loss_fn = self.network.loss_fn

        # Copy the script of the network to the model directory
        os.makedirs(self.config['MODEL_PATH'], exist_ok=True)
        os.system(f"""cp "{self.config['NET_DIR']}/{self.config['network']}.py" "{self.config['MODEL_PATH']}/network.py" """)
        # Save the configuration
        with open(os.path.join(self.config['MODEL_PATH'],'config.yaml'), 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)

    # Load the network
    def load_network(self, path):
        module = self.get_net_module()
        self.network = module(self.config)
        self.network.to(device = self.device)
        print(f"#params: {sum(p.numel() for p in self.network.parameters() if p.requires_grad)}" )
        self.loss_fn = self.network.loss_fn

        if not self.tf: # If not transfer (when the network is loaded), learning curve will be also loaded and same configuration will be used
            self.config = yaml.load(open(os.path.join(path,'config.yaml'), 'r'), Loader=yaml.FullLoader)
            self.learning_curve = pd.read_csv(os.path.join(path,'learning_curve.csv'))
        
        # loading parameters
        config = yaml.load(open(os.path.join(path,'config.yaml'), 'r'), Loader=yaml.FullLoader)
        if config['network'] != self.config['network']:
            self.load_params_transfer_learn(path)
        else:
            self.load_params(os.path.join(path,'final.pth'))

    # Initialize the optimizer
    def init_optimizer(self,lr_dict={}):        
        params = []
        for name, param in self.network.named_parameters():
            name = name.split(".")[0]
            if name in lr_dict:
                lr = lr_dict[name]
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                    patience=self.config.get('lr_patience',10), 
                                                                    min_lr=self.config.get('min_lr',1e-7),
                                                                    verbose=False)
    
    "----------------------------------------------------------------------------------------------------------------------"
    "Below are the functions to manage the network during the training, and called by the Trainer class"

    def train(self):
        self.state = "train"
        self.network.train()

    def eval(self):
        self.state = "eval"
        self.network.eval()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
        
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

    def scheduler_step(self, val_loss):
        self.scheduler.step(val_loss)

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
        if self.get_lr()<self.last_lr:
            self.last_lr=self.get_lr()
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
        self.network.load_state_dict(torch.load(path))

    def load_params_transfer_learn(self, path):
        config = yaml.load(open(os.path.join(path,'config.yaml'), 'r'), Loader=yaml.FullLoader)
        module = self.get_net_module()
        origin_network = module(config)
        origin_network.load_state_dict(torch.load(os.path.join(path,'final.pth')))

        pretrained_dict = origin_network.state_dict()
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

    