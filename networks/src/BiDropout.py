import torch
import torch.nn as nn

class Bi_Dropout(nn.Module):
    def __init__(self, p):
        super(Bi_Dropout, self).__init__()
        self.p = p
        self.drop_p = p
        self.training = True

    def set_drop_p(self,trainig):
        self.training = trainig
        self.drop_p = torch.rand(1).item()

    def drop_label(self, score:torch.Tensor,target:torch.Tensor):
        if self.training:    
            if self.drop_p<self.p: 
                target=nn.ReLU()(target)
            elif self.drop_p>1-self.p: 
                target=-nn.ReLU()(-target)
        return score,target
    
    def forward(self, x):
        if self.training:
            if self.drop_p<self.p: 
                x[:,1]= 0
            elif self.drop_p>1-self.p:
                x[:,0]= 0
        return x
