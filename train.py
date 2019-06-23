import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt
from loadData import WindowedData
from net import Myloss 
from net import CNN


# Hyper Parameter
EPOCH=100
BATCH_SIZE=256
LR=0.0001
WSIZE=120

# data
train_dataset=WindowedData("/Users/fdhcg/Desktop/clshen/data/test.txt",WSIZE)

train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset=WindowedData("/Users/fdhcg/Desktop/clshen/data/1998.txt",WSIZE)


test_loader = data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

cnn = CNN().cuda()
#optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=1)


    

loss_fun=Myloss().cuda()

#training loop
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):
        batch_x = Variable(x).cuda()
        batch_y = Variable(y).cuda()
        #输入训练数据
        output = cnn(batch_x)
        #计算误差
        loss = loss_fun(output,batch_y)
        #清空上一次梯度
        optimizer.zero_grad()
        #误差反向传递
        loss.backward()
        #优化器参数更新
        optimizer.step()

    sum_error=0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = Variable(x).cuda()
            y = Variable(y).cuda()
            #输入训练数据

            output = cnn(x)
            #计算误差
            sum_error += torch.mean(torch.abs(output-y))
    error=sum_error.cpu().data.numpy()/len(test_dataset)
    print("epoch: "+str(epoch)+" average error : "+str(error))
        
torch.save(cnn,'cnn.pth')