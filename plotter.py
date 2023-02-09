import matplotlib.pyplot as plt
import numpy as np
import math
import os

#################################
optimizers = ["RRM", "RRM + ADAM (0.1)", "RRM + ADAM (0.1) + lookahead", "RRM + ADAM (0.1) + learned lookahead", "RRM + ADAM (0.9)", "RRM + ADAM (0.9) + lookahead", "RRM + ADAM (0.9) + learned lookahead"]
#optimizers = ["RRM", "RRM + ADAM (0.1)", "RRM + ADAM (0.1) + lookahead", "RRM + ADAM (0.1) + learned lookahead"]
tran = "tran3"
meta = {"eps": 10000}
# eps: [0.1, 1, 10, 100, 1000, 10000]
seeds = list(range(1234,1244))

smooth = 20

#trans = [["tran1", {"eps": [0.1, 1, 25, 50]}], ["tran2", {}]]

result_folder = "results"
#################################

all_colors = ["r","g","b","c","m","y","purple"]

theta_diffs = []
step_losses = []
perf_risks = []
optimal_diffs = []

meta_name = ""
for key, val in meta.items():
    meta_name += f"{key} {float(val)} , "
meta_name = meta_name[:-3]
for optimizer in optimizers:
    opt_theta_diffs = [] # [seeds,100]
    opt_step_losses = [] # [seeds,100]
    opt_perf_risks = [] # [seeds,101]
    opt_optimal_diffs = [] # [seeds,101]
    
    for seed in seeds:
        exp_folder = f"{result_folder}/{optimizer}/{tran}/{meta_name}/{seed}"
        
        with open(f"{exp_folder}/theta_diffs.txt") as f:
            opt_theta_diffs.append([float(val) for val in f.read().splitlines()])
        with open(f"{exp_folder}/step_losses.txt") as f:
            opt_step_losses.append([float(val) for val in f.read().splitlines()])
        with open(f"{exp_folder}/optimal_diffs.txt") as f:
            opt_optimal_diffs.append([float(val) for val in f.read().splitlines()])
        with open(f"{exp_folder}/perf_risks.txt") as f:
            opt_perf_risks.append([float(val) for val in f.read().splitlines()])
    
    opt_theta_diffs = np.array(opt_theta_diffs).mean(0)
    opt_step_losses = np.array(opt_step_losses).mean(0)
    opt_perf_risks = np.array(opt_perf_risks).mean(0)
    opt_optimal_diffs = np.array(opt_optimal_diffs).mean(0)

    theta_diffs.append(opt_theta_diffs)
    step_losses.append(opt_step_losses)
    perf_risks.append(opt_perf_risks)
    optimal_diffs.append(opt_optimal_diffs)

theta_diffs = np.array(theta_diffs) # [optimizers,100]
step_losses = np.array(step_losses) # [optimizers,100]
perf_risks = np.array(perf_risks) # [optimizers,101]
optimal_diffs = np.array(optimal_diffs) # [optimizers,101]


kernel = np.full((smooth,), 1/smooth)
optimal_diffs = np.stack([np.convolve(optimal_diff, kernel, "valid") for optimal_diff in optimal_diffs])
perf_risks = np.stack([np.convolve(perf_risk, kernel, "valid") for perf_risk in perf_risks])

##################################################################################
'''for j, opt in enumerate(optimizers):
    plt.plot(optimal_diffs[j], color=all_colors[j], label=opt)

#plt.ylim(ymin=0)
plt.xlabel('Round')
plt.ylabel('Distance to optimal')
plt.title(f"Optimal Differences")

plt.legend()
plt.yscale("log")
plt.show()'''
##################################################################################
for j, opt in enumerate(optimizers):
    plt.plot(perf_risks[j], color=all_colors[j], label=opt)

#plt.ylim(ymin=0)
#plt.ylim(ymax=3)
plt.xlabel('Round')
plt.ylabel('Risk')
plt.title("Performative Risk")

plt.legend()
plt.yscale("log")
plt.show()
##################################################################################
'''for j, opt in enumerate(optimizers):
    plt.plot(theta_diffs[j], color=all_colors[j], label=opt)

plt.ylim(ymin=0)
plt.xlabel('Round')
plt.ylabel('Distance between thetas')
plt.title(f"Theta Differences")

plt.legend()
plt.show()'''
##################################################################################
'''for j, opt in enumerate(optimizers):
    plt.plot(step_losses[j], color=all_colors[j], label=opt)

plt.ylim(ymin=0)
plt.xlabel('Round')
plt.ylabel('Round loss')
plt.title(f"Losses")

plt.legend()
plt.show()'''
##################################################################################
