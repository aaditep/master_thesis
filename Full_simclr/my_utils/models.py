import torch.nn as nn




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        #nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
                        #nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class LinearLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias = True,
                 use_bn = False):
        super(LinearLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.use_bn = use_bn

        self.linear = nn.Linear(self.in_channels,
                                self.out_channels,
                                bias = self.use_bias and not self.use_bn)

        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_channels)

    def forward(self, x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self,
                in_channels,
                hidden_channels,
                out_channels,
                head_type = 'nonlinear'):
        super(ProjectionHead, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.head_type = head_type
            
        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_channels,self.out_channels,use_bias=False, use_bn = False)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_channels,self.hidden_channels,use_bias = True, use_bn = True),
                nn.ReLU(),
                LinearLayer(self.hidden_channels,self.out_channels,use_bias = False, use_bn = True))
        elif self.head_type == 'nonlinear_dropout':
            self.layers = nn.Sequential(
            nn.Linear(128, 64),  # Fully connected layer with 64 output units
            nn.ReLU(),           # ReLU activation function
            nn.Dropout(0.2),     # Dropout layer with 10% dropout rate
            nn.Linear(64, 2)     # Fully connected layer with 2 output units for regression
        )
            

    def forward(self, x):
        x = self.layers(x)
        return x
    

class ResNet(nn.Module):
    """

    ResNet model for regression

    Args:
    block : ResidualBlock : residual block
    layers : int : number of layers
    num_classes : int : number of classes
    """
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        #first two layers for downsampling
        self.conv1 = nn.Sequential(
                        nn.Conv2d(4, 32, kernel_size = (3,3), stride = 2, padding =1),
                        nn.BatchNorm2d(32),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size = (3,3), stride = 2, padding =1),
                        nn.ReLU())
        
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 128, layers[3], stride = 2)
        #self.layer4 = self._make_layer(block, 256, layers[4], stride = 2)
        #self.layer5 = self._make_layer(block, 128, layers[5], stride = 2)
        self.avgpool = nn.AvgPool2d(4,stride=1)
        #self.fc = nn.Linear(2048, 2)
        #self.projector = ProjectionHead(2048, 128, 2)
        


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.layer0(x) #residual block with specified layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        #x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) #squeezing
        #x = self.projector(x)
        #x = self.fc(x)# old output

        return x

class Regression_model(nn.Module):
    def __init__(self, layers : list, head_type, hidden_channels, out_params = 2):
        super(Regression_model, self).__init__()
        self.layers = layers
        self.out_params = 2
        self.head_type = head_type
        self.hidden_channels = hidden_channels
        
        #define the extractable part of the model
        self.resnet = ResNet(ResidualBlock, layers = self.layers)

        #define the projection head
        self.projector = ProjectionHead(128 , self.hidden_channels, self.out_params, head_type = self.head_type)

    def forward(self, x):
        x = self.resnet(x)
        x = self.projector(x)
        return x






class Regression_model_2(nn.Module):
    def __init__(self, layers : list, head_type, hidden_channels, out_params = 2):
        super(Regression_model_2, self).__init__()
        self.layers = layers
        self.out_params = 2
        self.head_type = head_type
        self.hidden_channels = hidden_channels
        
        #define the extractable part of the model
        self.resnet = ResNet_2(ResidualBlock, layers = self.layers)

        #define the projection head
        self.projector = ProjectionHead(128 , self.hidden_channels, self.out_params, head_type = self.head_type)

    def forward(self, x):
        x = self.resnet(x)
        x = self.projector(x)
        return x
    



class ResNet_2(nn.Module):
    """

    ResNet model for regression with different build


    Args:
    block : ResidualBlock : residual block
    layers : int : number of layers
    num_classes : int : number of classes
    """
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet_2, self).__init__()
        self.inplanes = 64
        #first two layers for downsampling
        self.conv1 = nn.Sequential(
                        nn.Conv2d(4, 32, kernel_size = (3,3), stride = 2, padding =1),
                        nn.BatchNorm2d(32),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size = (3,3), stride = 2, padding =1),
                        nn.ReLU())
        
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.conv3 = nn.Sequential(
                        nn.Conv2d(512, 256, kernel_size = (1,1), stride = 2),
                        nn.BatchNorm2d(256),
                        nn.ReLU())
        self.conv4 = nn.Sequential(
                        nn.Conv2d(256, 128, kernel_size = (1,1), stride = 2),
                        nn.BatchNorm2d(128),
                        nn.ReLU())
        #self.avgpool = nn.AvgPool2d(4, stride=1)
        #self.fc = nn.Linear(2048, 2)
        #self.projector = ProjectionHead(2048, 128, 2)
        


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.layer0(x) #residual block with specified layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv3(x) 
        x = self.conv4(x)        
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1) #squeezing
        #x = self.projector(x)
        #x = self.fc(x)# old output

        return x