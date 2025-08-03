from torch import nn 
import torch 
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class mydataset(Dataset):
    def __init__(self,path,type, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.path = path
        self.type = type
        self.data_pre()
    def data_pre(self):
        data = pd.read_csv(self.path)
        data = data.to_numpy()
        data = torch.tensor(data,dtype=torch.float)
        data = data[:,1:]
        if self.type == "train":
            data = data[[x for x in range(data.shape[0]) if x%10!=0]]
            self.data = data[:,:-1]
            self.data[:,40:] = (self.data[:,40:] - self.data[:,40:].min(0)[0]) / (self.data[:,40:].max(0)[0] - self.data[:,40:].min(0)[0])

            self.target = data[:,-1]
            #self.target = (self.target - self.target.min()) / (self.target.max() - self.target.min())
        elif self.type == 'dev':
            
            data = data[[x for x in range(data.shape[0]) if x%10==0]]
            self.data = data[:,:-1]
            self.data[:,40:] = (self.data[:,40:] - self.data[:,40:].min(0)[0]) / (self.data[:,40:].max(0)[0] - self.data[:,40:].min(0)[0])

            self.target = data[:,-1]
            #self.target = (self.target - self.target.min()) / (self.target.max() - self.target.min())
           
        else:
            self.data = data
            self.data[:,40:] = (self.data[:,40:] - self.data[:,40:].min(0)[0]) / (self.data[:,40:].max(0)[0] - self.data[:,40:].min(0)[0])
            target = pd.read_csv("pred.csv")
            target = target.to_numpy()
            target = target[:,1:]
            #target = (target-target.min())/(target.max()-target.min())
            self.target = torch.tensor(target,dtype=torch.float)
            

            print(self.target.shape)
    def __getitem__(self,index):
        return self.data[index],self.target[index]
    def __len__(self):
        return self.data.shape[0]
def mydataloader(path,type,batch_size):
    dataset = mydataset(path,type)
    return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=(type=="train"),drop_last=True)

class neural_network(nn.Module):
    def __init__(self, dim,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(dim, 128),  # 增加宽度
            nn.ReLU(),             # 改用ReLU
            nn.Dropout(0.3),       # 添加Dropout防止过拟合
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            
        )
    def forward(self,input):
        return self.model(input)
class main():
    def __init__(self,path,type,batch_size,device,model=None):
        self.device = device
        
        self.dataset = mydataloader(path,type,batch_size)
    
        self.neuralnetwork = neural_network(93).to(device) if not model else model.to(device)
        
        self.optim = torch.optim.AdamW(self.neuralnetwork.parameters(),lr=0.001,weight_decay=1e-4)
        
    def train(self,num):
        self.neuralnetwork.train()
        for epoch in range(num):
            for data,target in self.dataset:
                data = data.to(self.device)
                target = target.to(self.device)
                pre = self.neuralnetwork(data)
                loss = self.lossfunction(pre,target).to(self.device)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
               
            torch.save(self.neuralnetwork.state_dict(),'./models/model{}.pth'.format(epoch))
            print("{}次训练结束".format(epoch))

    def test(self):
        with torch.no_grad():
            self.neuralnetwork.eval()
            for data,target in self.dataset:
                data = data.to(self.device)
                target = target.to(self.device)
                pre = self.neuralnetwork(data)
                print(pre.shape)
                print(target.shape)
                plt.scatter( pre.cpu().numpy(),target.cpu().numpy(),label='test')
            plt.show()
    def val(self):
        Loss = 0
        with torch.no_grad():
            self.neuralnetwork.eval()
            for data,target in self.dataset:
                data = data.to(self.device)
                target = target.to(self.device)
                pre = self.neuralnetwork(data)
                print(pre)
                loss = self.lossfunction(pre, target).to(self.device)
                Loss += loss
            print('验证得到的损失和为{}'.format(Loss))
    def lossfunction(Self,pre,target):
        loss = nn.HuberLoss()
        return loss(pre.squeeze(),target)
if __name__ == "__main__":
      
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True    

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    path1 = 'covid.train.csv'
    main_ = main(path1,'train',270,device)
    main_.train(100)

    model = neural_network(93)
    mdoel_dict = torch.load('./models/model99.pth')
    model.load_state_dict(mdoel_dict)
    main_ = main(path1,'dev',64,device,model=model)
    main_.val()
    path2 = "covid.test.csv"
    model = neural_network(93)
    mdoel_dict = torch.load('./models/model99.pth')
    model.load_state_dict(mdoel_dict)
    main_ = main(path2,'test',64,device,model=model)
    main_.test()   
    """
    
    model = neural_network(93)
    mdoel_dict = torch.load('models/model99.pth')
    model.load_state_dict(mdoel_dict)
    path1 = 'covid.train.csv'
    mydata = mydataloader(path=path1,type='train',batch_size=270)     
    for data,target in mydata:
        pre = model(data)
        print(pre)
        print("------")
        print(target)
        break
      

    """                                                                                                                          