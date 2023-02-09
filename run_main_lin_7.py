from model import Regressor, PolynomV2
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

optimizer = "RRM + ADAM (0.9) + learned lookahead"

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
adam_lr = 0.9
beta1 = 0.9
beta2 = 0.999
eps = 10**-8
lookahead_train_steps = 1

T_degree = 1
T_lr = 0.01
T_iterations_until_lookahead = 5

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
        all_T_coeffs = []
        all_T_biases = []
        all_thetas_RRM = []
        all_thetas_opt = []
        all_thetas_lookahead = []

        device = torch.device('cuda' if torch.cuda.is_available() and device=="cuda" else 'cpu')
        print(f"training with {device}")

        model = Regressor(n_features, hidden, bias=False).to(device)
        total_elements = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_elements == n_features, "Not a linear model"
        env = EnvironmentStateless(n_examples, n_features, total_elements, tran, **meta).to(device)
        
        T_operator = PolynomV2(n_features+1+total_elements, T_degree, n_output=n_features+1).to(device)

        lossFunc = MSELoss()
        
        mt, vt = 0, 0

        for step in range(steps+1):
            X, Y = env.x, env.y
            if step>0:
                theta_diff = env.step(model)
                X, Y = env.x, env.y
                if step==1:
                    theta_diff = None
                
                theta = torch.zeros((total_elements,)).to(device)
                EnvironmentStateless._update_theta(model, theta)

                dataset = Supervised(torch.cat((env.x_src, env.y_src), 1), torch.cat((X,Y), 1), n_examples)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                best_model = None
                best_ind = None
                best_loss = None
                opt_T = optim.Adam(T_operator.parameters(), lr=T_lr)
                base_model = copy.deepcopy(T_operator)
                for i in range(max_train_steps):
                    epoch_loss = 0
                    for x, y in dataloader:
                        x, y = x.to(device), y.to(device)
                        batch_len = x.shape[0]
                        x_cat = torch.cat((x,theta.repeat(batch_len,1)), 1)
                        preds = T_operator(x_cat)
                        loss = lossFunc(preds, y)

                        opt_T.zero_grad()
                        loss.backward()
                        opt_T.step()
                        epoch_loss += batch_len*loss.item()
                    epoch_loss /= n_examples
                    if best_loss is None or best_loss>epoch_loss:
                        best_ind = i
                        best_loss = epoch_loss
                        best_model = copy.deepcopy(T_operator)
                        continue
                    if i>best_ind+early_stop:
                        break
                T_operator = best_model
                T_loss = best_loss
                
                #########################################
                #print(f"{step}: diff - {theta_diff}, optimal diff - {optimal_diffs[-1]}, risk - {perf_risks[-1]}, T-loss - {T_loss}")
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
            
            if step>=T_iterations_until_lookahead:
                adam_step = step-(T_iterations_until_lookahead-1)
                with torch.no_grad():
                    theta = torch.zeros((total_elements,))
                    EnvironmentStateless._update_theta(model, theta)
                    all_thetas_RRM.append(theta.numpy())
                    model_grad = base_theta-theta
                    
                    mt = beta1*mt + (1-beta1)*model_grad
                    vt = beta2*vt + (1-beta2)*(model_grad**2)

                    mt_ = mt/(1-beta1**adam_step)
                    vt_ = vt/(1-beta2**adam_step)

                    base_theta -= adam_lr*mt_/(vt_**0.5+eps)
                    
                    all_thetas_opt.append(base_theta.numpy())
                    
                    def init_weights(m):
                        if isinstance(m, nn.Linear):
                            m.weight = nn.Parameter(base_theta[None,...])
                    model.block.apply(init_weights)
                
                #####
                with torch.no_grad():
                    X, Y = env.peek(model)
                dataset = Supervised(X, Y, n_examples)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                opt = optim.Adam(model.parameters(), lr=lr)
                for i in range(lookahead_train_steps):
                    for x, y in dataloader:
                        x, y = x.to(device), y.to(device)
                        batch_len = x.shape[0]
                        preds = model(x)
                        loss = lossFunc(preds, y)
                        
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                
                theta = torch.zeros((total_elements,))
                EnvironmentStateless._update_theta(model, theta)
                all_thetas_lookahead.append(theta.numpy())
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        m.weight = nn.Parameter(base_theta[None,...])
                model.block.apply(init_weights)
                #####
                
                dataset = Supervised(env.x_src, env.y_src, n_examples)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                opt = optim.Adam(model.parameters(), lr=lr)
                T_operator.eval()
                for i in range(lookahead_train_steps):
                    epoch_loss = 0
                    for x, y in dataloader:
                        x, y = x.to(device), y.to(device)
                        batch_len = x.shape[0]

                        eye = torch.eye(total_elements).to(device)
                        params = model(eye).view(1,-1)

                        x_cat = torch.cat((x,y,params.repeat(batch_len,1)), 1)
                        y_cat = T_operator(x_cat)
                        x_pred = y_cat[:,:n_features]
                        y_pred = y_cat[:,n_features:]

                        preds = model(x_pred)
                        loss = lossFunc(preds, y_pred)
                        
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        epoch_loss += batch_len*loss.item()
                    epoch_loss /= n_examples
                T_operator.train()
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
            coeffs = T_operator.coeffs.detach().numpy()
            bias = T_operator.bias.detach().numpy()
            all_T_coeffs.append(coeffs)
            all_T_biases.append(bias)


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
        
        with open(f"{exp_folder}/all_thetas_RRM.npy", 'wb+') as f:
            all_thetas_RRM = np.stack(all_thetas_RRM)
            np.save(f, all_thetas_RRM)
        with open(f"{exp_folder}/all_thetas_opt.npy", 'wb+') as f:
            all_thetas_opt = np.stack(all_thetas_opt)
            np.save(f, all_thetas_opt)
        with open(f"{exp_folder}/all_thetas_lookahead.npy", 'wb+') as f:
            all_thetas_lookahead = np.stack(all_thetas_lookahead)
            np.save(f, all_thetas_lookahead)
        
        with open(f"{exp_folder}/all_T_coeffs.npy", 'wb+') as f:
            all_T_coeffs = np.concatenate(all_T_coeffs)
            np.save(f, all_T_coeffs)
        with open(f"{exp_folder}/all_T_biases.npy", 'wb+') as f:
            all_T_biases = np.concatenate(all_T_biases)
            np.save(f, all_T_biases)
