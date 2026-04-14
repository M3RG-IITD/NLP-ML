from MLPipeline import Node
from functools import partial
import numpy as np
import pickle

from .models.NN import NN

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor
import json

class optuna_Base(Node):
    def __init__(self, regressor, params, name="Optuna Base", split=0.25, n_trials=10, CV=False, save_random_states=True,use_trial = None):
        super().__init__(name)
        self.params = params
        self.split = split
        self.n_trials = n_trials
        self.CV = CV
        self.regressor = regressor
        self.model = None
        self.X = None
        self.y = None
        self.study = None
        self.save_random_states = save_random_states

        def run_study(data, filename, self=None):
            self.X, self.y = data
            self.filename = filename

            try:
                self.X = self.X.values
                self.y = self.y.values
            except:
                pass

            def set_data(input_data, self):
                (data, target) = input_data

                if self.save_random_states:
                    with open("{}_random.pkl".format(filename), "wb") as f:
                        pickle.dump(np.random.get_state(), f)
                else:
                    with open("{}_random.pkl".format(filename), "rb") as f:
                        np.random.set_state(pickle.load(f))

                self.train_x, self.valid_x, self.train_y, self.valid_y = train_test_split(data, target, test_size=self.split)

            def objective(trial, self=None):
                params = self.params(trial)
                with open("{}_Trial_{}_params.json".format(filename, trial.number), "w+") as file:
                    json.dump(params, file)

                if not(CV):
                    eval_set = [(self.train_x, self.train_y), (self.valid_x, self.valid_y)]
                    self.model,full_model = self.regressor(eval_set, params, "{}_Trial_{}".format(filename, trial.number))
                    R2_train = r2_score(self.train_y, self.model(self.train_x))
                    R2_val = r2_score(self.valid_y, self.model(self.valid_x))
                    with open("{}_Trial_{}_scores.txt".format(filename, trial.number), "w+") as file:
                            file.write("#Table\nTrain\tValidation\n")
                            file.write("{}\t{}\n".format(R2_train, R2_val))
                    return R2_val
                else:
                    kf = KFold(n_splits=10,shuffle=True)
                    scores = []
                    ind = 0

                    with open("{}_Trial_{}_scores.txt".format(filename, trial.number), "w+") as file:
                        file.write("#KFold Table:\n")
                        file.write("Fold\tTrain\tValidation\n")

                        if self.save_random_states:
                            with open("{}_Trial_{}_random.pkl".format(filename, trial.number), "wb") as f:
                                pickle.dump(np.random.get_state(), f)
                        else:
                            if use_trial!=None:
                                with open("{}_Trial_{}_random.pkl".format(filename, use_trial), "rb") as f:
                                    np.random.set_state(pickle.load(f))
                            else:
                                with open("{}_Trial_{}_random.pkl".format(filename, trial.number), "rb") as f:
                                    np.random.set_state(pickle.load(f))

                        for train_index, val_index in kf.split(self.X):
                            ind += 1
                            train_x, valid_x = self.X[train_index,:], self.X[val_index,:]
                            train_y, valid_y = self.y[train_index], self.y[val_index]
                            eval_set = [(train_x, train_y), (valid_x, valid_y)]
                            self.model,full_model = self.regressor(eval_set, params,"{}_Trial_{}_{}".format(filename, trial.number, ind),fn_use_trial="{}__Trial__{}__{}__{}".format(filename, trial.number, ind, use_trial))
                            try:
                                R2_train = r2_score(train_y, self.model(train_x))
                                R2_val = r2_score(valid_y, self.model(valid_x))
                                scores += [R2_val]
                                file.write("{}\t{}\t{}\n".format(ind, R2_train, R2_val))
                            except:
                                scores += [0.0]
                                file.write("{}\t{}\t{}\n".format(ind, 0.0, 0.0))


                    return np.mean(scores)


            self.study = optuna.create_study(study_name="Study {}".format(self.name), direction="maximize", sampler=TPESampler(seed=63))

            if not(CV):
                set_data(data, self)
            self.study.optimize(partial(objective,self=self), self.n_trials)
            with open("{}_Trial_best_{}".format(filename, self.study.best_trial.number), "w") as file:
                pass
            return self.study

        self.steps = [partial(run_study, self=self)]

class optuna_XGBoost(optuna_Base):
    def __init__(self, params, name="Optuna XGBoost", split=0.25, n_trials=10, CV=False, early_stopping_rounds=20, save_random_states=True,use_trial = None):

        def regressor(eval_set, params, filename, fn_use_trial=None):
            model = XGBRegressor(**params)
            X, y = eval_set[0]
            model.fit(X, y, eval_metric=["rmse"], eval_set=eval_set, verbose=False, early_stopping_rounds = early_stopping_rounds)
            results = model.evals_result()
            with open(filename+"_model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open(filename+"_trainingcurve.json", "w") as file:
                json.dump(results['validation_0'], file)
            with open(filename+"_validationcurve.json", "w") as file:
                json.dump(results['validation_1'], file)
            if params["booster"]=="gblinear":
                return partial(model.predict, ntree_limit=0), model
            else:
                return partial(model.predict, ntree_limit=model.best_ntree_limit), model

        super().__init__(regressor, params, name=name, split=split, n_trials=n_trials, CV=CV,save_random_states=save_random_states,use_trial = use_trial)


class optuna_NN(optuna_Base):
    def __init__(self, params, regressor=None, name="Optuna NN", split=0.25, n_trials=10, CV=False, early_stopping_rounds=20, save_random_states=True,use_trial = None):

        if regressor==None:
            regressor = NN

        super().__init__(regressor, params, name=name, split=split, n_trials=n_trials, CV=CV,save_random_states=save_random_states,use_trial = use_trial)

#
