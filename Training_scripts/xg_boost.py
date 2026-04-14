import sys
agrs = sys.argv
import os
import pandas as pd
import MLPipeline as MLP
optuna_XGBoost = MLP.tunning.optuna_XGBoost
 
def xgboost_params(trial):
    param = {
        "random_state": trial.suggest_int("random_state", 1, 1000),
        "objective": "reg:squarederror",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "n_estimators" : 400,
        "subsample": trial.suggest_float("subsample",0.7,1),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.7,1),
        "reg_alpha": trial.suggest_float("reg_alpha",1e-4,1e-1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda",1e-4,1e-1, log=True)
    }
    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 3, 8)
        param["eta"] = trial.suggest_float("eta", 1e-8, 0.1, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 0.1, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    return param
 


os.makedirs('../MLPipeline_results_dissolution_XGB_composition', exist_ok=True)
name = "xgb_dissolution_composition.csv"
import numpy as np
if '.csv' in name:
    
        for i in range(100):
            if not(os.path.exists('../MLPipeline_results_dissolution_XGB_composition/%s_%s'%(name[:-4],i))):
                os.mkdir("../MLPipeline_results_dissolution_XGB_composition/%s_%s"%(name[:-4],i))
                break

        X_ = pd.read_csv('/home/civil/staff/sidm3rg.cstaff/Documents/PU_Learning/glass_setup/maxpooled_processed_output_edited_composition/maxpooled_processed_output_edited_composition/01_Pipeline_umap_dissolution_maxpooled_comp_Node_train_test_split_train_split_X.csv')
        y_ = pd.read_csv('/home/civil/staff/sidm3rg.cstaff/Documents/PU_Learning/glass_setup/maxpooled_processed_output_edited_composition/maxpooled_processed_output_edited_composition/01_Pipeline_umap_dissolution_maxpooled_comp_Node_train_test_split_train_split_y.csv')
#         X_ = X_.iloc[:, :-10]
        
        P2 = MLP.Pipe(name="%s"%name[:-4], output="../MLPipeline_results_dissolution_XGB_composition/%s_%s"%(name[:-4],i))
        P2.add(MLP.data_cleaning.normalize_data())
        P2.add(optuna_XGBoost(xgboost_params, CV=True, n_trials=150))
        P2((X_, y_))