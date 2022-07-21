import torch
from torch import nn
import torch.nn.functional as F

class memes_ED(nn.Module):
    #used to compute similarity
    #the gradint can be transfered without optimizing memes
    def __init__(self,memes):
        super(memes_ED,self).__init__()
        self.weight = nn.Parameter(memes)
    def forward(self, x):
        x2=F.conv2d(input=x**2,weight=torch.ones_like(self.weight),stride=1,padding=0)
        x_m = F.conv2d(input=x, weight=self.weight,stride=1,padding=0)
        m2 = torch.sum(torch.sum(self.weight ** 2,dim=(2,3),keepdim=True),dim=1)
        return x2 - 2*x_m + m2
        
class memes_ED_norm(nn.Module):
    #used to compute similarity
    #the gradint can be transfered without optimizing memes
    def __init__(self,memes):
        super(memes_ED_norm,self).__init__()
        self.weight = nn.Parameter(memes)
    def forward(self, x):
        x2=F.conv2d(input=x**2,weight=torch.ones_like(self.weight),stride=1,padding=0)
        x_m = F.conv2d(input=x, weight=self.weight,stride=1,padding=0)
        m2 = torch.sum(torch.sum(self.weight ** 2,dim=(2,3),keepdim=True),dim=1)
        return (x2 - 2*x_m + m2)/m2

class memes_ED_sqrt(nn.Module):
    def __init__(self,memes):
        super(memes_ED_sqrt,self).__init__()
        self.weight = nn.Parameter(memes)
    def forward(self, x):
        x2=F.conv2d(input=x**2,weight=torch.ones_like(self.weight),stride=1,padding=0)
        x_m = F.conv2d(input=x, weight=self.weight,stride=1,padding=0)
        m2 = torch.sum(torch.sum(self.weight ** 2,dim=(2,3),keepdim=True),dim=1)
        return torch.sqrt(F.relu(x2 - 2*x_m + m2) + 1e-8)

class memes_ED_mask(torch.nn.Module):
    #compute ED in specific channels
    def __init__(self,memes,channel):
        super(memes_ED_mask, self).__init__()
        zeros=torch.zeros_like(memes, dtype=torch.float)
        self.mask = nn.Parameter(torch.stack([zeros[i].index_fill_(0,channel[i],1) for i in range(len(zeros))]))
        self.weight = nn.Parameter(memes*self.mask)
    def forward(self, x):
        x2=F.conv2d(input=x**2,weight=self.mask,stride=1,padding=0)
        x_m = F.conv2d(input=x, weight=self.weight,stride=1,padding=0)
        m2 = torch.sum(torch.sum(self.weight ** 2,dim=(2,3),keepdim=True),dim=1)
        dist = torch.sqrt(F.relu(x2 - 2*x_m + m2) + 1e-8)
        return dist
        #return x2 - 2 * x_m + m2
    def partial_forward(self, x, concerned_channels):
        used_mask = self.mask[:,concerned_channels]
        used_weight = self.weight[:,concerned_channels]
        x2=F.conv2d(input=x**2,weight=used_mask,stride=1,padding=0)
        x_m = F.conv2d(input=x, weight=used_weight,stride=1,padding=0)
        m2 = torch.sum(torch.sum(used_weight ** 2,dim=(2,3),keepdim=True),dim=1)
        dist = torch.sqrt(F.relu(x2 - 2*x_m + m2) + 1e-8)
        return dist

class Whitening(torch.nn.Module):
    def __init__(self, channels, length = 0, eps = 1e-5, iter = True):
        super(Whitening, self).__init__()
        self.mean = nn.Parameter(torch.zeros(len(channels)))
        self.var = nn.Parameter(torch.zeros(len(channels)))
        self.length = length
        self.eps = eps
        self.iter = iter
        self.channels = channels
    def update(self, mean, var, length):
        incre_mean = (self.length * self.mean + length * mean) / (self.length + length)
        self.var = nn.Parameter((self.length * (self.var + (incre_mean - self.mean) ** 2) + length * (
                    var + (incre_mean - mean) ** 2)) / (self.length + length))
        self.mean = nn.Parameter(incre_mean)
        self.length += length
    def switch(self):
        if self.iter:
            self.iter = False
        else:
            self.iter = True
    def load(self, mean, var, length, eps, iter = True):
        self.mean = nn.Parameter(torch.Tensor(mean))
        self.var = nn.Parameter(torch.Tensor(var))
        self.length = length
        self.eps = eps
        self.iter = iter
    def forward(self,x):
        x_cal = x[:, self.channels]
        assert x_cal.shape[1] == len(self.mean)
        assert x_cal.shape[1] == len(self.var)
        if self.iter:
            mean = torch.mean(x_cal, dim=[0, 2, 3])
            var = torch.var(x_cal, dim=[0, 2, 3], unbiased=False)
            shape = x_cal.shape
            length = shape[0] * shape[2] * shape[3]
            self.update(mean, var, length)
        result = x.clone()
        result[:,self.channels] = (x_cal - self.mean.view(1,-1,1,1)) / (torch.pow(self.var.view(1,-1,1,1) + self.eps, 0.5))
        return result

class zscore_ED_mask(torch.nn.Module):
    #compute ED in specific channels
    def __init__(self, memes, channel, zscore):
        super(zscore_ED_mask, self).__init__()
        self.bn = zscore
        self.bn.iter = False
        zeros=torch.zeros_like(memes, dtype=torch.float)
        self.mask = nn.Parameter(torch.stack([zeros[i].index_fill_(0,channel[i],1) for i in range(len(zeros))]))
        self.weight = nn.Parameter(memes*self.mask)
    def forward(self, x):
        x = self.bn(x)
        x2 = F.conv2d(input=x**2,weight=self.mask,stride=1,padding=0)
        x_m = F.conv2d(input=x, weight=self.weight,stride=1,padding=0)
        m2 = torch.sum(torch.sum(self.weight ** 2,dim=(2,3),keepdim=True),dim=1)
        dist = torch.sqrt(F.relu(x2 - 2*x_m + m2) + 1e-8)
        return dist

class zscore_sim_mask(torch.nn.Module):
    #compute simliarity in specific channels
    def __init__(self, memes, channel, zscore,
                 std):
        super(zscore_sim_mask,self).__init__()
        self.bn = zscore
        self.bn.switch()
        zeros=torch.zeros_like(memes, dtype=torch.float)
        self.mask=nn.Parameter(torch.stack([zeros[i].index_fill_(0,channel[i],1) for i in range(len(zeros))]))
        self.weight=nn.Parameter(memes*self.mask)
        self.std = std
    def forward(self, x):
        x = self.bn(x)
        x2 = F.conv2d(input=x**2,weight=self.mask,stride=1,padding=0)
        x_m = F.conv2d(input=x, weight=self.weight,stride=1,padding=0)
        m2 = torch.sum(torch.sum(self.weight ** 2,dim=(2,3),keepdim=True),dim=1)
        dist = torch.sqrt(F.relu(x2 - 2*x_m + m2) + 1e-8)
        return self.std(dist)
