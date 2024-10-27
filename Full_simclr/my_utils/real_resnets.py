import torch.nn as nn
import torch
from torchvision import datasets, transforms, models
#lets look at resnet for one more time. I know now how to change the shapes and places of the model

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features, 
                                self.out_features, 
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 dropout_rate,
                 head_type = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type
        self.dropout_rate = dropout_rate

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features,self.hidden_features,True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features,self.out_features,False,True))
        elif self.head_type == 'nonlinear_dropout':
            self.layers = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),  # Fully connected layer with 64 output units
            nn.ReLU(),           # ReLU activation function
            nn.Dropout(self.dropout_rate),     # Dropout layer with 10% dropout rate
            nn.Linear(self.hidden_features, self.out_features)     # Fully connected layer with 2 output units for regression
        )
    def forward(self,x):
        x = self.layers(x)
        return x
    

class Resnet_pretrainingmodel(nn.Module):
    def __init__(self,base_model,pretrained_weights,dropout_rate,head_type):
        super().__init__()
        self.base_model = base_model
        self.pretrained_weights = pretrained_weights
        self.dropout_rate = dropout_rate
        self.head_type = head_type
        #PRETRAINED MODEL
        if self.base_model == "resnet18" or self.base_model == 'resnet18_simclr':
            self.pretrained = models.resnet18(pretrained=self.pretrained_weights)
        if self.base_model == "resnet34" or self.base_model == 'resnet34_simclr':
            self.pretrained = models.resnet34(pretrained=self.pretrained_weights)
        if self.base_model == "resnet50" or self.base_model == 'resnet50_simclr':
            self.pretrained = models.resnet50(pretrained=self.pretrained_weights)
        self.pretrained.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False
        #self.pretrained.maxpool = Identity()
        
        
        #this is how i change a model
        self.pretrained.fc = Identity()
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
            
        if self.base_model == "resnet18":
            self.projector = ProjectionHead(512, 256, 2,dropout_rate = self.dropout_rate ,head_type = self.head_type)
        if self.base_model == 'resnet18_simclr':
            self.projector = ProjectionHead(512, 512, 128,dropout_rate = self.dropout_rate ,head_type = self.head_type)
        if self.base_model == "resnet34":
            self.projector = ProjectionHead(512, 256, 2,dropout_rate = self.dropout_rate ,head_type = self.head_type)
        if self.base_model == 'resnet34_simclr':
            self.projector = ProjectionHead(512, 512, 128,dropout_rate = self.dropout_rate ,head_type = self.head_type)
        if self.base_model == "resnet50":
            self.projector = ProjectionHead(2048, 1024, 2,dropout_rate = self.dropout_rate ,head_type = self.head_type)
        if self.base_model == 'resnet50_simclr':
            self.projector = ProjectionHead(2048, 1024, 128,dropout_rate = self.dropout_rate ,head_type = self.head_type)
       
    def forward(self,x):
        out = self.pretrained(x)
        
        xp = self.projector(torch.squeeze(out))
        
        return xp

class Resnet_regressionmodel(nn.Module):
    def __init__(self,base_model,pretrained_weights,dropout_rate,head_type):
        super().__init__()
        self.base_model = base_model
        self.pretrained_weights = pretrained_weights
        self.dropout_rate = dropout_rate
        self.head_type = head_type
        #PRETRAINED MODEL
        if self.base_model == "resnet18":
            self.pretrained = models.resnet18(pretrained=self.pretrained_weights)
        if self.base_model == "resnet34":
            self.pretrained = models.resnet34(pretrained=self.pretrained_weights)
        if self.base_model == "resnet50":
            self.pretrained = models.resnet50(pretrained=self.pretrained_weights)
        self.pretrained.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        #self.pretrained.maxpool = Identity()
        
        
        #this is how i change a model
        self.pretrained.fc = Identity()
        
        for p in self.pretrained.parameters():
            p.requires_grad = True ### CHANGING TO FALSE FOR A TEST
            
        if self.base_model == "resnet18":
            self.projector = ProjectionHead(512, 256, 2,dropout_rate = self.dropout_rate ,head_type = self.head_type)
        if self.base_model == "resnet34":
            self.projector = ProjectionHead(512, 256, 2,dropout_rate = self.dropout_rate ,head_type = self.head_type)
        if self.base_model == "resnet50":
            self.projector = ProjectionHead(2048, 1024, 2,dropout_rate = self.dropout_rate ,head_type = self.head_type)

    def forward(self,x):
        out = self.pretrained(x)
        
        xp = self.projector(torch.squeeze(out))
        
        return xp


class DSModel(nn.Module):
    def __init__(self,premodel,base_model,dropout_rate, head_type):
        super().__init__()
        
        self.premodel = premodel
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.head_type = head_type
        
        #set rquieres grad to false for the premodel to avoid training the main body of it it
        for p in self.premodel.parameters():
            p.requires_grad = False
            
        for p in self.premodel.projector.parameters():
            p.requires_grad = False
        
        #The only trained part of the model is the new projector
        
        
        if self.base_model == "resnet18":
            self.projector = ProjectionHead(512, 256, 2,dropout_rate = self.dropout_rate ,head_type = self.head_type)
        if self.base_model == 'resnet18_simclr':
            self.projector = ProjectionHead(512, 512, 128,dropout_rate = self.dropout_rate ,head_type = self.head_type)
        if self.base_model == "resnet34":
            self.projector = ProjectionHead(512, 256, 2,dropout_rate = self.dropout_rate ,head_type = self.head_type)
        if self.base_model == 'resnet34_simclr':
            self.projector = ProjectionHead(512, 512, 128,dropout_rate = self.dropout_rate ,head_type = self.head_type)
        if self.base_model == "resnet50":
            self.projector = ProjectionHead(2048, 1024, 2,dropout_rate = self.dropout_rate ,head_type = self.head_type)
            
            
        #self.lastlayer = nn.Linear(2048,self.num_classes)
        
    def forward(self,x):
        """
        By not adding here premodel.projector you are omitting it
        """
        out = self.premodel.pretrained(x)
        #omit the premodel projector and replace with the needed new projector
        out = self.projector(out)
        return out