import os
from itertools import product
import shutil
import time

#################################
optimizers = ["RRM", "RRM + ADAM (0.1)", "RRM + ADAM (0.1) + lookahead", "RRM + ADAM (0.1) + learned lookahead", "RRM + ADAM (0.9)", "RRM + ADAM (0.9) + lookahead", "RRM + ADAM (0.9) + learned lookahead"]
trans = [["tran3", {"eps": [0.1, 1, 10, 100, 1000, 10000]}]]
seeds = list(range(1234,1244))

result_folder = "results"

files_to_check = ["theta_diffs.txt", "step_losses.txt", "optimal_diffs.txt", "perf_risks.txt"]
#################################

while True:

    try:

        currents = {(optimizer, tran): 0 for optimizer, (tran, metas) in list(product(optimizers, trans))}
        totals = {(optimizer, tran): 0 for optimizer, (tran, metas) in list(product(optimizers, trans))}

        #print("All Problems:")
        #print("#"*50)
        total = 0
        count = 0
        for optimizer in optimizers:
            for ((tran, metas), seed) in list(product(trans, seeds)):
                keys = metas.keys()
                values = metas.values()
                all_values = list(product(*values))
                for values in all_values:
                    meta = {key:val for key,val in zip(keys,values)}
                    meta_name = ""
                    for key,val in zip(keys,values):
                        meta_name += f"{key} {float(val)} , "
                    meta_name = meta_name[:-3]

                    exp_name = f"{optimizer}__{tran}__{meta_name}__{seed}"

                    exp_folder = f"{result_folder}/{optimizer}/{tran}/{meta_name}/{seed}"
                    
                    problem = False
                    if not os.path.isdir(exp_folder):
                        problem = True
                    for file_to_check in files_to_check:
                        if not os.path.isfile(f"{exp_folder}/{file_to_check}"):
                            problem = True
                    
                    if problem:
                        #print(exp_name)
                        None
                    else:
                        count += 1
                        currents[(optimizer, tran)] += 1
                    
                    total += 1
                    totals[(optimizer, tran)] += 1
        
    except KeyboardInterrupt:
        exit()
    os.system('clear')
    print("#"*50)
    for key in currents.keys():
        print(f"{key}: {currents[key]}/{totals[key]}")
    print("#"*50)
    print(f"Progress: {count}/{total} ({(count/total*100):.2f}%)")
    time.sleep(1)
