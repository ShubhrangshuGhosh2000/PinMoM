import torch
from torch.utils import data


class SimpleTorchDictionaryDataset(data.Dataset):
    def __init__(self,featureData,oneDdataMatrix,pairLst,classData=None,full_gpu=False,deviceType='cpu',createNewTensor=False):
        if createNewTensor:
            self.data = torch.tensor(featureData).float()
            self.oneDdata = torch.tensor(oneDdataMatrix).float()
        else:
            self.data=featureData
            self.oneDdata=oneDdataMatrix
        self.pairLst =pairLst  
        self.noClasses=False
        if classData is None:
            self.noClasses=True
            self.classData = torch.ones(self.pairLst.shape[0])*-1
        else:
            self.classData = torch.tensor(classData)
        self.classData = self.classData.long()
        self.full_gpu = full_gpu 
        self.deviceType = deviceType


    def __getitem__(self,index):
        y = self.classData[index]
        x0 = self.data[self.pairLst[index][0]]
        x1 = self.data[self.pairLst[index][1]]
        x0 = x0.unsqueeze(0)
        x1 = x1.unsqueeze(0)
        x0 = x0.float()
        x1 = x1.float()
        aux0 = self.oneDdata[self.pairLst[index][0]]
        aux1 = self.oneDdata[self.pairLst[index][1]]
        aux0 = aux0.float()
        aux1 = aux1.float()
        return (x0,x1,aux0,aux1,y)


    def __len__(self):
        return self.classData.shape[0]


    def activate(self):        
        if self.full_gpu: 
            self.data = self.data.to(torch.device(self.deviceType))
            self.oneDdata = self.oneDdata.to(torch.device(self.deviceType))
            self.classData = self.classData.to(torch.device(self.deviceType))


    def deactivate(self):
        self.data = self.data.cpu()
        self.oneDdata = self.oneDdata.cpu()
        self.classData = self.classData.cpu()
