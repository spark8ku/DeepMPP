import time
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

    
class Trainer():
    "The class where the operation of the training is defined"
    def __init__(self,config):
        self.model_path=config['MODEL_PATH']
        self.dev=config['device']
        self.max_epoch=config['max_epoch']
        self.learning_curve=None
        self.train_error = None
    
    def train(self, network_manager, train_loader, val_loader):

        self.learning_curve=getattr(network_manager,'learning_curve',None)
        start_time=time.time()

        try:
            with tqdm(range(self.max_epoch)) as t:
                for epoch in t:
                    t.set_description('Epoch %d' % epoch)
                    epoch_train_loss = self.train_epoch(network_manager, train_loader)
                    epoch_val_loss = self.evaluate(network_manager, val_loader)
                    epoch_train_loss=np.mean(epoch_train_loss)
                    epoch_val_loss=np.mean(epoch_val_loss)
                    
                    t.set_postfix({'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss})

                    row=pd.Series({'train_loss':epoch_train_loss, 'val_loss': epoch_val_loss})
                    if self.learning_curve is None:
                        self.learning_curve=pd.DataFrame(row).transpose()
                    else:
                        self.learning_curve = pd.concat([self.learning_curve,pd.DataFrame(row).transpose()],axis=0)

                    if network_manager.scheduler_step(epoch_val_loss):
                        break
                             
                    
        except KeyboardInterrupt:
            print("Forced termination")
        except Exception as e:
            self.train_error = e
            raise Exception()        
        
        network_manager.load_best_checkpoint()
        self.run_time = time.time()-start_time
    
    def train_epoch(self, network_manager, train_loader):
        network_manager.train()
        score_list = []
        target_list = []
        epoch_loss = 0
        count=0
        for loader in (train_loader):       
            torch.autograd.set_detect_anomaly(False)   
            y, y_pred, loss = network_manager.step(loader)
            
            score_list.append(y_pred)
            target_list.append(y)

            count+=y_pred.shape[0]
            epoch_loss += loss*y_pred.shape[0]
        if count: epoch_loss/=count

        return epoch_loss
    
    
    def evaluate(self, network_manager, val_loader):    
        network_manager.eval()
        score_list = []
        target_list = []
        epoch_loss = 0
        count=0
        with torch.no_grad():
            for loader in (val_loader):     
                torch.autograd.set_detect_anomaly(False)
                y, y_pred, loss = network_manager.step(loader)
                

                score_list.append(y_pred)
                target_list.append(y)

                count+=y_pred.shape[0]
                epoch_loss += loss*y_pred.shape[0]
        if count: epoch_loss/=count

        return epoch_loss
    
    def predict(self, network_manager, pred_loader,dropout=False):
        network_manager.eval()
        if dropout:
            network_manager.dropout_on()
        score_list = []
        target_list = []
        smiles_list = []
        solvent_list = []
        with torch.no_grad():
            for loader in tqdm(pred_loader, desc='Predicting'):     
                torch.autograd.set_detect_anomaly(False)
                y, y_pred, loss,x = network_manager.step(loader,flag=True)
                score_list.append(y_pred)
                target_list.append(y)
                smiles_list+=x['smiles']
                solvent_list+=x.get('solv_smiles',[""]*y_pred.shape[0])
        return torch.cat(score_list, dim=0), torch.cat(target_list, dim=0), smiles_list, solvent_list

    def get_score(self, network_manager, loader):
        "Provide the score of the model. The model which is used for the training should contains the implementation of 'get_score'."
        network_manager.eval()
        pred_result = {}
        with torch.no_grad():
            for loader in tqdm(loader, desc='Calculating scores'):     
                torch.autograd.set_detect_anomaly(False)
                y_pred = network_manager.step(loader,get_score=True)
                if type(y_pred) is not dict:
                    y_pred = {'prediction':y_pred}
                if pred_result:
                    for key in y_pred:
                        pred_result[key] = torch.cat([pred_result[key],y_pred[key]], dim=0)
                else:
                    pred_result = y_pred
        return pred_result