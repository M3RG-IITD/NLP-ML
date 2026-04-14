import sys
agrs = sys.argv
from functools import partial
import pandas as pd
import numpy as np
import MLPipeline as MLP
import os

optuna_NN = MLP.tunning.optuna_NN
def NN_params(trial, batch_size=[1], minN=1, maxN=10):
    params = {
    "epochs": 500,
    "batch_size": trial.suggest_categorical("batch_size", batch_size),
    "n_layers": trial.suggest_int("n_layers", 2,6),
    "drop": trial.suggest_categorical("drop", [True, False]),
    "drate": trial.suggest_categorical("drate", [i for i in np.arange(0.1,0.4,0.1)]),
    "norm": trial.suggest_categorical("norm", [False, True]),
    "activation":  trial.suggest_categorical("activation", ["LeakyReLU", "ReLU"]),
    "opt": trial.suggest_categorical("opt", ["Adam", "SGD"]),
    "opt_params": {},
    }
    params["layers"] = [trial.suggest_int("L{}".format(i), minN, maxN) for i in range(params["n_layers"])]
    if params["opt"]=="SGD":
        params["opt_params"] = {
            "lr": trial.suggest_float("lr", 1e-5, 0.1, log=True),
            "momentum": trial.suggest_float("momentum", 9e-5, 0.9, log=True),
            }
    if params["opt"]=="Adam":
        params["opt_params"] = {
            "lr": trial.suggest_float("lr", 1e-3, 0.1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 0.001, log=True),
            }
    return params
    
os.makedirs('../MLPipeline_results_NN_dissolution_composition', exist_ok=True)
name = "NN_dissolution_composition_NO_NLP.csv"
import numpy as np
if '.csv' in name:
    
        for i in range(100):
            if not(os.path.exists('../MLPipeline_results_NN_dissolution_composition/%s_%s'%(name[:-4],i))):
                os.mkdir("../MLPipeline_results_NN_dissolution_composition/%s_%s"%(name[:-4],i))
                break

        X_ = pd.read_csv('/home/civil/staff/sidm3rg.cstaff/Documents/PU_Learning/glass_setup/maxpooled_processed_output_edited_composition/maxpooled_processed_output_edited_composition/01_Pipeline_umap_dissolution_maxpooled_comp_Node_train_test_split_train_split_X.csv')
        y_ = pd.read_csv('/home/civil/staff/sidm3rg.cstaff/Documents/PU_Learning/glass_setup/maxpooled_processed_output_edited_composition/maxpooled_processed_output_edited_composition/01_Pipeline_umap_dissolution_maxpooled_comp_Node_train_test_split_train_split_y.csv')
        X_ = X_.iloc[:, :-10]
        
        N = X_.shape[0]
        D_in = X_.shape[1]
        if N<100:
            batch_size = [8]
        else:
            batch_size = [int(i) for i in list(2**np.arange(np.floor(np.log2(0.02*N)),np.floor(np.log2(0.04*N))+1))]
        minN = int(np.floor(D_in/2))
        maxN = D_in*2
        P2 = MLP.Pipe(name="%s"%name[:-4], output="../MLPipeline_results_NN_dissolution_composition/%s_%s"%(name[:-4],i))
        P2.add(MLP.data_cleaning.normalize_data())  ##(x-mean)/(std)
        P2.add(optuna_NN(partial(NN_params, batch_size=batch_size, minN=minN, maxN=maxN), CV=True, n_trials=150))
        P2((X_, y_))