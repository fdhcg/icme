import torch.utils.data 
import os
import torch
import numpy as np 
import sys


class MyData(torch.utils.data.Dataset):
    def __init__(self,path):
        self.path=path
        self.rawline=[]
        self.data=[]
        self.label=[]
        #self.windowsize=windowsize
        self.load_file()
        self.trans_data()
        print("\ndata preparing finished!")
    
    def __len__(self):
        return len(data)

    def load_file(self):
        f=open(self.path)
        print(self.path)
        self.rawline=f.readlines()[::10]
        print("load dataset finished!") 
    def trans_data(self):
        
        print("reformatting dataset...")
        stdout = sys.stdout
        length=len(self.rawline)
        for i in range(length):           
            stdout.write('\rcomplete percent:'+'#'*int(100*i/length)+'%.2f'%(100*float(i+1)/length)+'%')
            rawline=self.rawline[i].split(" ")
            line=[]
            for x in rawline:
                if x:
                    if x.endswith("/n"):
                        line.append(x[:-2])
                    else:
                        line.append(x)
            self.data.append([float(x) for x in line[1:-2]])
            label1=int(line[-2])
            label2=int(line[-1])
            self.label.append(label1)
        print(i)
        

    def __getitem__(self,idx):
        pass
        #return torch.tensor(self.data[idx]),torch.tensor(self.label[idx]).long()


class WindowedData(MyData):
    def __init__(self,path,wsize):
        super(WindowedData,self).__init__(path)
        self.wsize=wsize
        self.data_=[]
        self.label_=[]
        self.get_label_list()
        self.trans_data_()
        torch.set_default_dtype(torch.float32)
    def get_label_list(self):
        label=self.label
        if label[0]!=0:
            raise Exception("Invalid dataset!")
        for i in range(len(label)):
            if label[i]!=0 and label[i-1]==0:
                idx_left=i
            if label[i]==0 and label[i-1]!=0:
                idx_right=i
                self.label[idx_left:idx_right]=[idx_right-idx_left]*(idx_right-idx_left)
        
        

                  
    def __len__(self):
        return len(self.data_)
                

    def trans_data_(self):
        for i in range(len(self.data)-self.wsize+1):
            label=self.label[i:i+self.wsize]
            
            gsize=max(label)
            
            if gsize!=0:
                
                u=sum(label)/gsize
                self.label_.append([u/(self.wsize+gsize-u)])

            else:
                self.label_.append([0.])
            self.data_.append([self.data[i:i+self.wsize]])
        

    def __getitem__(self,idx):
        
        return torch.tensor(self.data_[idx]).transpose(0,1).reshape(self.wsize*9).unsqueeze(0),torch.tensor(self.label_[idx])







