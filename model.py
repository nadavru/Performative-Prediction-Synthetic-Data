import torch
from torch import nn

class Regressor(nn.Module):
    def __init__(self, n_features, hidden=[32], n_output=1, bias=True):
        super().__init__()

        layers = []
        in_channel = n_features
        for out_channel in hidden:
            layers.append(nn.Linear(in_channel, out_channel, bias=bias))
            layers.append(nn.LeakyReLU(0.1))
            in_channel = out_channel
        layers.append(nn.Linear(in_channel, n_output, bias=bias))

        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class Polynom(nn.Module):
    def __init__(self, n_features, degree=3, n_output=1):
        super().__init__()

        self.coeffs = nn.Linear(n_features*degree, n_output)
        self.degree = degree
    
    def forward(self, x):
        # [b,f]
        batch = x.shape[0]
        variables = x[...,None].repeat(1,1,self.degree).pow(torch.arange(1,1+self.degree).reshape(1,1,-1)) # [b,f,degree]
        variables = variables.reshape(batch,-1) # [b,f*degree]
        return self.coeffs(variables) # [b,n_output]

class PolynomV2(nn.Module):
    def __init__(self, n_features, degree, n_output):
        super().__init__()

        assert n_features == 2*n_output-1 # [x,y,theta] -> [x',y'] (x,x',theta are same size, y,y' are scalars)
        #self.coeffs = nn.Linear(2*degree, n_output-1)
        self.coeffs = nn.Parameter(torch.zeros(1,n_output-1, 2*degree).uniform_(-1/((2*degree)**0.5), 1/((2*degree)**0.5)))
        self.bias = nn.Parameter(torch.randn(1,n_output-1))
        self.degree = degree
        self.v = n_output-1
    
    def forward(self, x):
        # [b,f]
        batch = x.shape[0]
        batch_x, batch_y, batch_theta = x[:,:self.v], x[:,self.v:self.v+1], x[:,self.v+1:]
        variables = torch.stack((batch_x,batch_theta), dim=2) # [b,v,2]
        variables = variables[...,None].repeat(1,1,1,self.degree).pow(torch.arange(1,1+self.degree).reshape(1,1,1,-1)) # [b,v,2,degree]
        variables = variables.reshape(batch,self.v,-1) # [b,v,2*degree]
        new_x = torch.mul(variables, self.coeffs).sum(-1)+self.bias # [b,v]
        new_y = batch_y # [b,1]
        return torch.cat((new_x, new_y), dim=1) # [b,v+1]


######## Don't use!!!!!!!!
class PolynomV3(nn.Module):
    def __init__(self, n_features, degree, n_output):
        super().__init__()

        assert n_features == 2*n_output-1 # [x,y,theta] -> [x',y'] (x,x',theta are same size, y,y' are scalars)
        #self.coeffs = nn.Linear(2*degree, n_output-1)
        self.coeffs = nn.Linear(2*degree, 1)
        self.degree = degree
        self.v = n_output-1
    
    def forward(self, x):
        # [b,f]
        batch = x.shape[0]
        batch_x, batch_y, batch_theta = x[:,:self.v], x[:,self.v:self.v+1], x[:,self.v+1:]
        variables = torch.stack((batch_x,batch_theta), dim=2) # [b,v,2]
        variables = variables[...,None].repeat(1,1,1,self.degree).pow(torch.arange(1,1+self.degree).reshape(1,1,1,-1)) # [b,v,2,degree]
        variables = variables.reshape(batch,self.v,-1) # [b,v,2*degree]
        new_x = self.coeffs(variables).squeeze(2) # [b,v]
        new_y = batch_y # [b,1]
        return torch.cat((new_x, new_y), dim=1) # [b,v+1]
