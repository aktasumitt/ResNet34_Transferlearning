from torchvision.models import resnet34,ResNet34_Weights
import torch.nn as nn

def Loading_Resnet():
    model=resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad=False
    
    return model
 
def Add_Last_Layer(model,label_size):
    in_features=model.fc.in_features
    model.fc=nn.Linear(in_features,label_size)
    
    return model






    

