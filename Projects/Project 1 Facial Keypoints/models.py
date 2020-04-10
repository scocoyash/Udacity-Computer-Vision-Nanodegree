import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torchvision import models

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # input(224, 224, 1)
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.drop1 = nn.Dropout(p=0.1)
        # input becomes (110, 110, 1)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.drop2 = nn.Dropout(p=0.2)
        # input becomes (54, 54, 64)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.drop3 = nn.Dropout(p=0.3)
        # input becomes (26, 26, 128)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.drop4 = nn.Dropout(p=0.4)
        # input becomes (13, 13, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256*13*13 ,1000)
        self.drop5 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000 ,1000)
        self.drop6 = nn.Dropout(p=0.6)
        self.fc3 = nn.Linear(1000 ,136)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        
        x = self.fc3(x)    
        return x


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        # for grayscale
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        n_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_inputs, 136)
                        
    def forward(self, x):
        x = self.resnet18(x)
        return x