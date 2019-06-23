import torch 
import torch.nn as nn
import numpy as np
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( #input shape (1,9,9)
            nn.Conv1d(in_channels=1, #input height 
                    out_channels=6, #n_filter
                    kernel_size=3, #filter size
                    stride=1, #filter step
                    padding=1 #填充一维使大小不变
                    ), #output shape (16,9)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) #output shape (16,3)
               
        )
        self.conv2 = nn.Sequential(nn.Conv1d(6, 12 , 3, 1, 1), #output shape (32,3)
                                nn.ReLU(),
                                nn.MaxPool2d(2)
                                )

        self.out = nn.Linear(1620,1)
         
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) 
        output = self.out(x)
        return output
        
class Myloss(nn.Module):
    def __init__(self):
        super(Myloss,self).__init__()

        
    def forward(self,x1,x2):
        length=len(x1)
        diff=torch.abs(x1-x2)
        loss=torch.log((torch.exp(diff)+torch.exp(-diff))/2+ 1e-10)
        return torch.mean(loss)