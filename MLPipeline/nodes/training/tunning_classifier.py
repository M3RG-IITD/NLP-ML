from MLPipeline import Node
from functools import partial
import numpy as np
import pickle

from .models.NN_Classifier import NN_classifier

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, f1_score

from tqdm.auto import tqdm
import optuna
# from optuna.integration import ProgressBar
from xgboost import XGBClassifier
import json

class optuna_Base(Node):
    def __init__(self, classifier, params, name="Optuna Base", split=0.25, n_trials=10, CV=False, save_random_states=True,use_trial = None):
        super().__init__(name)
        self.params = params
        self.split = split
        self.n_trials = n_trials
        self.CV = CV
        self.classifier = classifier
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
                    self.model,full_model = self.classifier(eval_set, params, "{}_Trial_{}".format(filename, trial.number))
#                     probs_train = full_model.predict_proba(self.train_x)[:, 1]
#                     probs_val = full_model.predict_proba(self.valid_x)[:, 1]
                    f1_train = f1_score(self.train_y, self.model(self.train_x))
                    f1_val = f1_score(self.valid_y, self.model(self.valid_x))
                    with open("{}_Trial_{}_scores.txt".format(filename, trial.number), "w+") as file:
                            file.write("#Table\nTrain\tValidation\n")
                            file.write("{}\t{}\n".format(f1_train, f1_val))
                    return f1_val
                else:
                    kf = KFold(n_splits=5,shuffle=True)
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
                            print(train_y)
                            eval_set = [(train_x, train_y), (valid_x, valid_y)]
                            self.model,full_model = self.classifier(eval_set, params,"{}_Trial_{}_{}".format(filename, trial.number, ind),fn_use_trial="{}__Trial__{}__{}__{}".format(filename, trial.number, ind, use_trial))
                            
#                                 probs_train = full_model.predict_proba(self.train_x)[:, 1]
#                                 probs_val = full_model.predict_proba(self.valid_x)[:, 1]
#                             oned_train_y = self.train_y.ravel()
#                             oned_valid_y = self.valid_y.ravel()
                            f1_train = f1_score(train_y, self.model(train_x))
                            f1_val = f1_score(valid_y, self.model(valid_x))
                            scores += [f1_val]
                            file.write("{}\t{}\t{}\n".format(ind, f1_train, f1_val))
#                             except:
#                                 scores += [0.0]
#                                 file.write("{}\t{}\t{}\n".format(ind, 'no_score', 'no_score'))


                    return np.mean(scores)


            self.study = optuna.create_study(study_name="Study {}".format(self.name), direction="maximize")

            if not(CV):
                set_data(data, self)
            with tqdm(total=self.n_trials, desc="Optimizing Trials") as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    
                self.study.optimize(partial(objective,self=self), self.n_trials, callbacks=[callback])
            with open("{}_Trial_best_{}".format(filename, self.study.best_trial.number), "w") as file:
                pass
            return self.study

        self.steps = [partial(run_study, self=self)]

class optuna_XGBoost(optuna_Base):
    def __init__(self, params, name="Optuna XGBoost", split=0.25, n_trials=10, CV=False, early_stopping_rounds=30, save_random_states=True,use_trial = None):

        def classifier(eval_set, params, filename, fn_use_trial=None):
            model = XGBClassifier(**params)
            X, y = eval_set[0]
            model.fit(X, y, eval_metric=["logloss"], eval_set=eval_set, verbose=False, early_stopping_rounds = early_stopping_rounds)
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

        super().__init__(classifier, params, name=name, split=split, n_trials=n_trials, CV=CV,save_random_states=save_random_states,use_trial = use_trial)


class optuna_NN(optuna_Base):
    def __init__(self, params, classifier=None, name="Optuna NN", split=0.25, n_trials=10, CV=False, early_stopping_rounds=30, save_random_states=True,use_trial = None):

        if classifier==None:
            classifier = NN_classifier

        super().__init__(classifier, params, name=name, split=split, n_trials=n_trials, CV=CV,save_random_states=save_random_states,use_trial = use_trial)

#
