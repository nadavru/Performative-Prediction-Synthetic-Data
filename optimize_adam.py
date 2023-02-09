import torch
import torch.nn as nn
import numpy as np
from model import Regressor
from torch.nn import MSELoss

n_examples = 30000
n_features = 30
eps = 50

lr = 0.1
beta1 = 0.9
beta2 = 0.999

seed = 1234
steps = 500

torch.manual_seed(seed)
np.random.seed(seed)

x = torch.randn((n_examples, n_features))
optimal_model = Regressor(n_features, hidden=[], bias=False)
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight)
optimal_model.block.apply(init_weights)

with torch.no_grad():
    y = optimal_model(x)

theta_opt = torch.zeros((n_features,))
ind = 0
for p in optimal_model.parameters():
    if p.requires_grad:
        p_size = p.numel()
        theta_opt[ind:ind+p_size] = p.detach().view(-1)
        ind += p_size

theta = torch.zeros((n_features,1), requires_grad=True)
with torch.no_grad():
    theta.uniform_(-1/((2*n_features)**0.5), 1/((2*n_features)**0.5))

A = x-eps*theta_opt.view(1,-1)
c = torch.full_like(y,eps)
b = y

lossFunc = MSELoss()

mt, vt = 0, 0
for i in range(1,steps+1):
    
    theta = theta.reshape(-1,1)
    preds = A@theta + c*theta.T@theta
    
    loss = lossFunc(preds, y)
    grad = torch.autograd.grad(loss, theta)[0].reshape(-1)
    theta = theta.reshape(-1)
    
    mt = beta1*mt + (1-beta1)*grad
    vt = beta2*vt + (1-beta2)*(grad**2)


    mt_ = mt/(1-beta1**i)
    vt_ = vt/(1-beta2**i)

    with torch.no_grad():
        theta -= lr*mt_/(vt_**0.5+10**-8)

        diff = ((theta-theta_opt.view(-1))**2).sum().item()**0.5

    print(f"{i}: {diff}")
    #print(f"{i}: {grad}")
