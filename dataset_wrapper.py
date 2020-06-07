import torch
import torch.utils.data as utils_data

class FI_Dataset(utils_data.Dataset):
    
    def __init__(self, X, y, T, target_num=0):
        self.T = T
        self.X = torch.from_numpy(X).unsqueeze(0).float()
        self.y = torch.from_numpy(y[T - 1:, target_num]).long() - 1
    
    def __len__(self):
        return len(self.X[0]) - self.T
    
    def __getitem__(self, idx):
        return self.X[:, idx:idx+self.T], self.y[idx]

class CryptoDataset(utils_data.Dataset):

    def __init__(self, X, y, T=100):
        self.T = T
        self.X = torch.from_numpy(X).unsqueeze(0).float()
        self.y = torch.from_numpy(y[T-1:]).long()

    def __len__(self):
        return len(self.X[0]) - self.T

    def __getitem__(self, idx):
        return self.X[:, idx:idx + self.T], self.y[idx]