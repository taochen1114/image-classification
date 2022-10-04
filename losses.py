import torch
from torch.nn import Module, CrossEntropyLoss
from torch.nn import BCELoss, BCEWithLogitsLoss

class LossFunction():
    def __init__(self, loss_type):
        loss_func_dict = {'CrossEntropyLoss': CrossEntropyLoss(), 
                      'BCELoss': BCELoss(), 
                      'BCEWithLogitsLoss': BCEWithLogitsLoss()                      
                      }
        self.loss_type = loss_type
        self.loss_func = loss_func_dict[loss_type]

    def __call__(self, outputs, targets):
        return self.loss_func(outputs, targets)
    
