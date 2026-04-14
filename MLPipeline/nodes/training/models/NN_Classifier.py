import json
import numpy as np
import pickle
import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.utils import setup_logger
from ignite.metrics import Loss


class NN_seq_classifier(nn.Module):
    def __init__(self, D_in, D_out, activation=nn.ReLU, layers=[1], dropout_rate=[0.2], batch_norm=True, dropout=True):
        super(NN_seq_classifier, self).__init__()

        if len(layers)!=len(dropout_rate):
            dropout_rate = dropout_rate*len(layers)

        self.seq = nn.Sequential()

        for a, b, p, n in zip([D_in]+layers[:-1], layers, dropout_rate, range(1+len(layers))):

            self.seq.add_module("Linear {}".format(n), nn.Linear(a, b))

            if dropout:
                self.seq.add_module("Dropout {}".format(n), nn.Dropout(p=p))

            self.seq.add_module("Activation {}".format(n), activation())

            if batch_norm:
                self.seq.add_module("Batch Norm {}".format(n), nn.BatchNorm1d(b))

        self.seq.add_module("Linear", nn.Linear(layers[-1], D_out))

    def forward(self,x):
        return self.seq(x)


def get_data_loader_classifier(data, batch_size):
    (X, y) = data
    dataset = TensorDataset(Tensor(X), Tensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def NN_classifier(eval_set, in_params, filename,fn_use_trial=None):
    # fn_use_trial = "01_Pipeline_Electrical_conduct_at_RT_Node_Optuna NN__Trial__0__1__None"
    [filename_,_,trial_number_,ind_,use_trial] = fn_use_trial.split('__')
    params = {
    "epochs": 100,
    "batch_size": 1,
    "layers":[4,4,4],
    "drop": False,
    "drate": 0.2,
    "norm": False,
    "activation": "ReLU",
    "opt": "SGD",
    "opt_params": {},
    }

    log_interval = 100
    params.update(in_params)
    params["opt_params"].update(in_params["opt_params"])

    train, val = eval_set

    train_x, train_y = train
    val_x, val_y = val

    # extract params
    epochs = params["epochs"]
    batch_size = params["batch_size"]
    layers = params["layers"]
    drop = params["drop"]
    drate = params["drate"]
    norm = params["norm"]
    act = params["activation"]

    train_loader = get_data_loader_classifier((train_x, train_y), batch_size)
    val_loader = get_data_loader_classifier((val_x, val_y), batch_size)

    D_in = train_x.shape[1]
    D_out = train_y.shape[1]
    if use_trial == 'None':
        rs = torch.random.get_rng_state()
        with open("{}_Trial_{}_{}_seed.pkl".format(filename_, trial_number_, ind_),'wb') as f:
            pickle.dump(rs,f)
    else:
        with open("{}_Trial_{}_{}_seed.pkl".format(filename_, use_trial, ind_),'rb') as f:
            rs = pickle.load(f)
        torch.random.set_rng_state(rs)

    model = NN_seq_classifier(D_in, D_out, getattr(nn, act), layers=layers, dropout_rate=[drate], batch_norm=norm, dropout=drop)

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    model.to(device)  # Move model before creating optimizer
    optimizer = getattr(optim, params["opt"])(model.parameters(), **params["opt_params"])

    criterion = nn.BCEWithLogitsLoss()
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer.logger = setup_logger("trainer")

    val_metrics = {"BCELoss": Loss(criterion)}
    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    evaluator.logger = setup_logger("evaluator")
    evaluator.training_curve = []
    evaluator.validation_curve = []

    desc = "ITERATION - loss: {:.2f}"

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss_classifier(engine):
        print("Loss : ", engine.state.output)
        if np.isnan(engine.state.output):
            print("Stopped due to NaNs.")
            engine.state.epoch = params["epochs"]

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results_classifier(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        bce = metrics["BCELoss"]
        evaluator.training_curve += [bce]
        print(
            "Training Results - Epoch: {}  Loss: {:.2f}".format(
                engine.state.epoch, bce
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_valation_results_classifier(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        bce = metrics["BCELoss"]
        evaluator.validation_curve += [bce]
        print(
            "Training Results - Epoch: {}  Loss: {:.2f}".format(
                engine.state.epoch, bce
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
    def log_time_classifier(engine):
        print(
            "{} took {} seconds".format(trainer.last_event_name.name, trainer.state.times[trainer.last_event_name.name])
        )

    trainer.run(train_loader, max_epochs=epochs)
    with open(filename+"_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(filename+"_trainingcurve.json", "w") as file:
        json.dump({"bceloss":evaluator.training_curve}, file)
    with open(filename+"_validationcurve.json", "w") as file:
        json.dump({"bceloss":evaluator.validation_curve}, file)

    def predict_classifier(x):
        model.eval()
        with torch.no_grad():
            x_processed = x.astype(np.float32) # Ensure float for PyTorch
            x_tensor = Tensor(x_processed)
        
            logits = model(x_tensor) # model is from NN_classifier's scope
            probs = torch.sigmoid(logits).cpu().detach().numpy()
            preds = (probs > 0.5).astype(int)
        
        return preds.ravel()



    return predict_classifier, model
