import os
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pickle

from .utils.metrics import get_metrics
from .utils import tools

import traceback
class PostProcessor:
    """
    The class for the postprocessing of the trained model.
    This class is used to save the model, learning curve, prediction, and metrics after the training.
    """

    def __init__(self, config):
        self.config = config
        self.model_path = config['MODEL_PATH']

    def postprocess(self, dm, nm, tm, train_loaders, val_loaders, test_loaders):
        "The main function for the postprocessing"
        try:
            self.clear_checkpoint()
            nm.save_params(os.path.join(self.model_path,"final.pth"))
        except:
            print("Model saving failed")
            return
        
        try:
            tm.learning_curve.to_csv(os.path.join(self.model_path,"learning_curve.csv"),index=False)
            self.draw_learning_curve(tm.learning_curve)
        except:
            print(traceback.format_exc())
            print("Learning curve saving failed")
            
        self.save_model_summary(nm.network,tm.run_time)

        try:
            prediction_df = self.save_prediction(nm, tm, train_loaders, val_loaders, test_loaders, scaler=dm.scaler)
            self.plot_prediction(self.config.get("target"),prediction_df)
            self.save_metrics(self.config.get("target"),prediction_df)
        except:
            print(traceback.format_exc())
            print("Prediction saving failed")
        
    def save_prediction(self, nm, tm, train_loader, val_loader, test_loader,scaler=None):
        train_pred, train_true, train_smiles, train_solv = tm.predict(nm, train_loader)
        val_pred, val_true, val_smiles, val_solv = tm.predict(nm, val_loader)
        test_pred, test_true, test_smiles, test_solv = tm.predict(nm, test_loader)
        targets = self.config.get("target")
        if scaler is not None:
            train_true = scaler.inverse_transform(train_true.cpu().numpy())
            val_true = scaler.inverse_transform(val_true.cpu().numpy())
            test_true = scaler.inverse_transform(test_true.cpu().numpy())
            train_pred = scaler.inverse_transform(train_pred.cpu().numpy())
            val_pred = scaler.inverse_transform(val_pred.cpu().numpy())
            test_pred = scaler.inverse_transform(test_pred.cpu().numpy())

            with open(os.path.join(self.model_path,"scaler.pkl"),"wb") as file:
                pickle.dump(scaler,file)
            
        prediction_df = None
        for true,pred,smiles,solvent,sets in zip([train_true,val_true,test_true],[train_pred,val_pred,test_pred],[train_smiles,val_smiles,test_smiles],[train_solv,val_solv,test_solv],['train','val','test']):
            df = pd.DataFrame(pred,columns=[f"{target}_pred" for target in targets])
            for i,target in enumerate(targets):
                df[f"{target}_true"] = true[:,i]
            df['compound'] = smiles
            df['solvent'] = solvent
            df['set'] = sets
            if prediction_df is None:
                prediction_df = df
            else:
                prediction_df = pd.concat([prediction_df,df],axis=0)

        prediction_df.to_csv(os.path.join(self.model_path,"prediction.csv"),index=False)
        return prediction_df

    def save_metrics(self, targets, prediction_df):
        metrics = get_metrics(targets, prediction_df)
        for set in ['train','val','test']:
            table = PrettyTable()
            col = ['target',f'{set}_r2',f'{set}_mae',f'{set}_rmse']
            table.field_names = col
            for t in targets:
                table.add_row([t,metrics[f'{set}_r2'][t],metrics[f'{set}_mae'][t],metrics[f'{set}_rmse'][t]])
            print(table)
        
        metrics.to_csv(os.path.join(self.model_path,"metrics.csv"))

    def plot_prediction(self,target, prediction_df):
        img = tools.plot_prediction(target, prediction_df)
        img.save(os.path.join(self.model_path,"prediction.png"))
        

    def save_model_summary(self, model,run_time):
        with open(os.path.join(self.model_path,"model_summary.txt"), "w") as file:
            file.write(str(model))
            file.write(f"#params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            file.write("\n")
            file.write(str(model.__class__.__name__))
            file.write("\n")
            file.write(str(model.__class__.__doc__))
            file.write("\n")
            file.write("learning_time: "+str(run_time))

    def draw_learning_curve(self, learning_curve):
        img = tools.plot_learning_curve(learning_curve['train_loss'],learning_curve['val_loss'])
        img.save(os.path.join(self.model_path,"learning_curve.png"))

    def clear_checkpoint(self):
        for file in os.listdir(self.model_path):
            if file.endswith(".pth"):
                os.remove(os.path.join(self.model_path, file))
        
                