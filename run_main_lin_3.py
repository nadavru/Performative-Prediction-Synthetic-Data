from model import Regressor
from environment_lin import EnvironmentStateless
import torch
from torch.nn import MSELoss
import torch.optim as optim
from data_utils import Supervised, Unsupervised
from torch.utils.data import DataLoader
import copy
from itertools import product
import os
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import sys

#################################
n_examples = 30000
n_features = 2
hidden = []

optimizer = "RRM + ADAM (0.1) + lookahead"

assert len(sys.argv)>=4
[tran, eps, seed] = sys.argv[1:4]
eps = float(eps)
seed = int(seed)

trans = [[tran, {"eps": [eps]}]]

steps = 500
lr = 0.01
batch_size = 32
max_train_steps = 100
early_stop = 10
adam_lr = 0.1
beta1 = 0.9
beta2 = 0.999
eps = 10**-8
lookahead_train_steps = 1

seeds = [seed]

device = "cpu"

result_folder = "results"
#################################

for ((tran, metas), seed) in list(product(trans, seeds)):
    keys = metas.keys()
    values = metas.values()
    all_values = list(product(*values))
    for values in all_values:
        meta = {key:val for key,val in zip(keys,values)}
        meta_name = ""
        for key,val in zip(keys,values):
            meta_name += f"{key} {val} , "
        meta_name = meta_name[:-3]

        exp_name = f"{optimizer}__{tran}__{meta_name}__{seed}"
        exp_folder = f"{result_folder}/{optimizer}/{tran}/{meta_name}/{seed}"
        print("#"*50)
        print(exp_name)
        print("#"*50)

        torch.manual_seed(seed)
        np.random.seed(seed)
        
        theta_diffs = []
        step_losses = []
        perf_risks = []
        optimal_diffs = []
        all_thetas = []

        device = torch.device('cuda' if torch.cuda.is_available() and device=="cuda" else 'cpu')
        print(f"training with {device}")

        model = Regressor(n_features, hidden, bias=False).to(device)
        total_elements = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_elements == n_features, "Not a linear model"
        env = EnvironmentStateless(n_examples, n_features, total_elements, tran, **meta).to(device)

        lossFunc = MSELoss()
        
        mt, vt = 0, 0

        for step in range(steps+1):
            X, Y = env.x, env.y
            if step>0:
                theta_diff = env.step(model)
                X, Y = env.x, env.y
                if step==1:
                    theta_diff = None
                
                #########################################
                #print(f"{step}: diff - {theta_diff}, optimal diff - {optimal_diffs[-1]}, risk - {perf_risks[-1]}")
                if theta_diff is None:
                    theta_diffs.append(-1)
                else:
                    theta_diffs.append(theta_diff)
                step_losses.append(step_loss)
                #########################################
            
            ##################################################################################
            dataset = Supervised(X, Y, n_examples)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            best_model = None
            best_ind = None
            best_loss = None
            opt = optim.Adam(model.parameters(), lr=lr)

            with torch.no_grad():
                base_theta = torch.zeros((total_elements,))
                EnvironmentStateless._update_theta(model, base_theta)
            
            for i in range(max_train_steps):
                epoch_total_loss = 0
                epoch_loss = 0
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    batch_len = x.shape[0]
                    preds = model(x)
                    loss = lossFunc(preds, y)
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    epoch_loss += batch_len*loss.item()
                epoch_loss /= n_examples
                if best_loss is None or best_loss>epoch_loss:
                    best_ind = i
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(model)
                    continue
                if i>best_ind+early_stop:
                    break
            model = best_model
            step_loss = best_loss
            
            if step>0:
                with torch.no_grad():
                    theta = torch.zeros((total_elements,))
                    EnvironmentStateless._update_theta(model, theta)
                    model_grad = base_theta-theta
                    
                    mt = beta1*mt + (1-beta1)*model_grad
                    vt = beta2*vt + (1-beta2)*(model_grad**2)

                    mt_ = mt/(1-beta1**step)
                    vt_ = vt/(1-beta2**step)

                    base_theta -= adam_lr*mt_/(vt_**0.5+eps)
                    
                    def init_weights(m):
                        if isinstance(m, nn.Linear):
                            m.weight = nn.Parameter(base_theta[None,...])
                    model.block.apply(init_weights)
                
                with torch.no_grad():
                    X, Y = env.peek(model)
                dataset = Supervised(X, Y, n_examples)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                opt = optim.Adam(model.parameters(), lr=lr)
                for i in range(lookahead_train_steps):
                    epoch_loss = 0
                    for x, y in dataloader:
                        x, y = x.to(device), y.to(device)
                        batch_len = x.shape[0]
                        preds = model(x)
                        loss = lossFunc(preds, y)
                        
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        epoch_loss += batch_len*loss.item()
                    epoch_loss /= n_examples
            ##################################################################################

            with torch.no_grad():
                X, Y = env.peek(model)
                dataset = Supervised(X, Y, n_examples)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                perf_risk = 0
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    batch_len = x.shape[0]
                    preds = model(x)
                    loss = lossFunc(preds, y)
                    perf_risk += batch_len*loss.item()
                perf_risk /= n_examples

                perf_risks.append(perf_risk)
            
            theta = torch.zeros((total_elements,))
            EnvironmentStateless._update_theta(model, theta)
            optimal_diffs.append(((theta-env.optimal_theta.view(-1).detach())**2).sum().item())
            all_thetas.append(theta.numpy())


        os.makedirs(exp_folder, exist_ok=True)
        with open(f"{exp_folder}/theta_diffs.txt", 'w+') as f:
            for theta_diff in theta_diffs:
                f.write(f"{theta_diff}\n")
        with open(f"{exp_folder}/step_losses.txt", 'w+') as f:
            for step_loss in step_losses:
                f.write(f"{step_loss}\n")
        with open(f"{exp_folder}/perf_risks.txt", 'w+') as f:
            for perf_risk in perf_risks:
                f.write(f"{perf_risk}\n")
        with open(f"{exp_folder}/optimal_diffs.txt", 'w+') as f:
            for optimal_diff in optimal_diffs:
                f.write(f"{optimal_diff}\n")
        with open(f"{exp_folder}/all_thetas.npy", 'wb+') as f:
            all_thetas = np.stack(all_thetas)
            np.save(f, all_thetas)
        with open(f"{exp_folder}/optimal_theta.npy", 'wb+') as f:
            np.save(f, env.optimal_theta.view(-1).detach().numpy())
