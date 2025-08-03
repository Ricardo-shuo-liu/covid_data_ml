from torch import nn 
import torch 
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os

class mydataset(Dataset):
    # 类属性保存归一化参数
    feature_mean = None
    feature_std = None
    
    def __init__(self, path, type, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.path = path
        self.type = type
        self.data_pre()
        
    def data_pre(self):
        data = pd.read_csv(self.path)
        data = data.to_numpy()
        data = torch.tensor(data, dtype=torch.float)
        data = data[:, 1:]
        
        if self.type == "train":
            data = data[[x for x in range(data.shape[0]) if x % 10 != 0]]
            self.data = data[:, :-1]
            self.target = data[:, -1]
            
            # 计算并保存归一化参数
            mydataset.feature_mean = self.data[:, 40:].mean(dim=0, keepdim=True)
            mydataset.feature_std = self.data[:, 40:].std(dim=0, keepdim=True)
            
            # 应用归一化
            self.data[:, 40:] = (self.data[:, 40:] - mydataset.feature_mean) / mydataset.feature_std
            
        elif self.type == 'dev':
            data = data[[x for x in range(data.shape[0]) if x % 10 == 0]]
            self.data = data[:, :-1]
            self.target = data[:, -1]
            
            # 使用训练集的统计量进行归一化
            if mydataset.feature_mean is not None and mydataset.feature_std is not None:
                self.data[:, 40:] = (self.data[:, 40:] - mydataset.feature_mean) / mydataset.feature_std
            else:
                # 如果没有训练集参数，使用当前数据集的
                self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) / self.data[:, 40:].std(dim=0, keepdim=True)
                
        else:  # test
            self.data = data
            self.target = torch.zeros(data.shape[0])  # 测试集目标值初始化为0
            
            # 使用训练集的统计量进行归一化
            if mydataset.feature_mean is not None and mydataset.feature_std is not None:
                self.data[:, 40:] = (self.data[:, 40:] - mydataset.feature_mean) / mydataset.feature_std
            else:
                # 如果没有训练集参数，使用当前数据集的
                self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) / self.data[:, 40:].std(dim=0, keepdim=True)
            
            # 加载真实目标值（如果有）
            try:
                target = pd.read_csv("pred.csv")
                target = target.to_numpy()
                target = target[:, 1:]
                self.target = torch.tensor(target, dtype=torch.float)
            except:
                pass
            
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
    def __len__(self):
        return self.data.shape[0]

def mydataloader(path, type, batch_size):
    dataset = mydataset(path, type)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(type=="train"), drop_last=True)

class neural_network(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 保持与参考代码相同的网络结构
        self.model = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, input):
        return self.model(input)

class main():
    def __init__(self, path, type, batch_size, device, model=None):
        self.device = device
        self.dataset = mydataloader(path, type, batch_size)
        self.neuralnetwork = neural_network(93).to(device) if not model else model.to(device)
        # 保持与参考代码相同的优化器设置
        self.optim = torch.optim.SGD(self.neuralnetwork.parameters(), lr=0.001, momentum=0.9)
        
    def train(self, num_epochs):
        self.neuralnetwork.train()
        best_val_loss = float('inf')
        best_model_path = ''
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = len(self.dataset)
            
            for data, target in self.dataset:
                data = data.to(self.device)
                target = target.to(self.device)
                pre = self.neuralnetwork(data)
                
                # 确保维度匹配
                pre = pre.squeeze()
                
                loss = self.lossfunction(pre, target)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                # 使用.item()获取标量值
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            
            # 验证集评估
            val_loss = self.val()
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = f'./models/best_model.pth'
                torch.save(self.neuralnetwork.state_dict(), best_model_path)
                print(f"保存最佳模型，验证损失={val_loss:.4f}")
            
            # 早停检查（如果连续20个epoch没有改进）
            if epoch > 20 and val_loss > best_val_loss:
                print("验证损失不再改善，提前停止训练")
                return best_model_path
        
        return best_model_path

    def test(self):
        self.neuralnetwork.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in self.dataset:
                data = data.to(self.device)
                target = target.to(self.device)
                pre = self.neuralnetwork(data)
                pre = pre.squeeze()
                
                predictions.append(pre.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('真实值 vs 预测值')
        plt.grid(True)
        plt.show()
        
        return predictions
    
    def val(self):
        self.neuralnetwork.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for data, target in self.dataset:
                data = data.to(self.device)
                target = target.to(self.device)
                pre = self.neuralnetwork(data)
                pre = pre.squeeze()
                
                loss = self.lossfunction(pre, target)
                # 使用.item()获取标量值并乘以批次大小
                total_loss += loss.item() * data.size(0)
                num_samples += data.size(0)
        
        avg_loss = total_loss / num_samples
        return avg_loss
        
    def lossfunction(self, pre, target):
        return nn.MSELoss()(pre, target)

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建模型保存目录
    os.makedirs('./models', exist_ok=True)
    
    # 训练模型
    path1 = 'covid.train.csv'
    print("开始训练...")
    trainer = main(path1, 'train', 270, device)
    best_model_path = trainer.train(100)
    
    # 加载最佳模型
    print(f"加载最佳模型: {best_model_path}")
    model = neural_network(93).to(device)
    model.load_state_dict(torch.load(best_model_path))
    
    # 在验证集上评估
    print("验证集评估...")
    validator = main(path1, 'dev', 64, device, model=model)
    val_loss = validator.val()
    print(f"验证集损失: {val_loss:.4f}")
    
    # 在测试集上评估和可视化
    print("测试集评估...")
    path2 = "covid.test.csv"
    tester = main(path2, 'test', 64, device, model=model)
    predictions = tester.test()
    
    # 保存预测结果
    if os.path.exists("pred.csv"):
        print("保存预测结果到 pred.csv")
        df = pd.read_csv("pred.csv")
        df['tested_positive'] = predictions
        df.to_csv("pred.csv", index=False)
    else:
        print("创建预测结果文件 pred.csv")
        df = pd.DataFrame({
            'id': range(len(predictions)),
            'tested_positive': predictions
        })
        df.to_csv("pred.csv", index=False)
    
    print("完成!")