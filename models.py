import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def load_model(model, model_path):
    
    checkpoint = torch.load(model_path)
    
    model.load_state_dict(checkpoint['state_dict'])

    epoch = checkpoint['epoch']
    print(f'==> Load model from {epoch} epoch')
    return model, epoch


class ClassificationModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrain):
        super(ClassificationModel, self).__init__()

        self.model_name = model_name
        featsize_mapper = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048}

        if self.model_name == 'resnet18':
            self.features = nn.Sequential(
                *list(models.resnet18(pretrained=pretrain).children())[:-1])
            self.classifier = nn.Sequential(nn.Linear(512, num_classes))

        elif self.model_name == 'resnet34':
            self.features = nn.Sequential(
                *list(models.resnet34(pretrained=pretrain).children())[:-1])
            self.classifier = nn.Sequential(nn.Linear(512, num_classes))

        elif self.model_name == 'resnet50':
            self.features = nn.Sequential(
                *list(models.resnet50(pretrained=pretrain).children())[:-1])
            self.classifier = nn.Sequential(nn.Linear(2048, num_classes))
        
        # elif self.model_name == 'mobilenet_v2':
        #     self.features = nn.Sequential(
        #         *list(models.mobilenet_v2(pretrained=pretrain).children())[:-1])
        
        # elif self.model_name == 'inception_v3':
        #     self.features = nn.Sequential(
        #         *list(models.inception_v3(pretrained=pretrain).children())[:-1])
            
        else:
            raise('Not support this architecture yet')

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1) # flatten
        return self.classifier(f)
