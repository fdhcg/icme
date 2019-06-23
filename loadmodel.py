import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt
from loadData import WindowedData
BATCH_SIZE=256
WSIZE=120
sum_error=0

cnn=torch.load('cnn.pth')

test_dataset=WindowedData("/Users/fdhcg/Desktop/clshen/data/2016.txt",WSIZE)

test_loader = data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        x = Variable(x).cuda()
        
        y = Variable(y)
        #输入训练数据
        
        output = cnn(x).cpu()
        #计算误差
        sum_error += torch.mean(torch.abs(output-y))
error=sum_error.cpu().data.numpy()/len(test_dataset)
print("average error : "+str(error))