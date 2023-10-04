import torch
from torch import nn


class Time2Vec(nn.Module):
    def __init__(self,in_features, out_features, f=torch.sin, arg=None):
        super(Time2Vec, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.f = f
        self.arg = arg
        
    def forward(self, tau):
        if self.arg:
            v1 = self.f(torch.matmul(tau, self.w) + self.b, self.arg)
        else:
            v1 = self.f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], 1)


