import torch
from torch import nn
import numpy as np
from model import Regressor

class EnvironmentStateless:
    def __init__(self, n_examples, n_features, total_elements, tran="tran1", **kwargs):
        assert tran in ["tran1", "tran2", "tran3"]
        ########################################
        self.x = torch.randn((n_examples, n_features))

        optimal_model = Regressor(n_features, hidden=[], bias=False)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight)
        optimal_model.block.apply(init_weights)
        
        with torch.no_grad():
            self.y = optimal_model(self.x)
        self.x_src = torch.clone(self.x)
        self.y_src = torch.clone(self.y)
        theta = torch.zeros((total_elements,))
        self._update_theta(optimal_model, theta)
        self.optimal_theta = theta
        ########################################
        self.total_elements = total_elements
        self.tran = tran
        self.meta = kwargs

        self.theta = torch.zeros((n_features,))

        init_model = Regressor(n_features, hidden=[], bias=False)
        self.x, self.y = self.peek(init_model)

    @staticmethod
    def _update_theta(model, theta):
        ind = 0
        for p in model.parameters():
            if p.requires_grad:
                p_size = p.numel()
                theta[ind:ind+p_size] = p.detach().view(-1)
                ind += p_size
    
    def _update_x_y(self, theta):
        if self.tran == "tran1":
            new_x = self.x_src+self.meta["eps"]*(theta.view(1,-1)-self.optimal_theta.view(1,-1))
            new_y = self.y_src
        if self.tran == "tran2":
            new_x = self.x_src+self.meta["eps"]*(theta.view(1,-1)-self.optimal_theta.view(1,-1))**3
            new_y = self.y_src
        if self.tran == "tran3":
            new_x = self.x_src+self.meta["eps"]*theta.view(1,-1)
            new_y = self.y_src
        return new_x, new_y
    
    def peek(self, model, k=1):
        theta = torch.zeros_like(self.theta)
        self._update_theta(model, theta)
        
        # It doesn't matter if k>1! Stateless world...
        x, y = self._update_x_y(theta)
        return x, y
    
    def step(self, model):
        last_theta = self.theta.clone()
        self._update_theta(model, self.theta)
        theta_diff = torch.dist(self.theta, last_theta)

        self.x, self.y = self._update_x_y(self.theta)
        return theta_diff
    
    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.theta = self.theta.to(device)
        self.optimal_theta = self.optimal_theta.to(device)
        self.device = device

        return self
