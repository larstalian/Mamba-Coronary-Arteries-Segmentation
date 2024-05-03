import optuna
import torch
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from model_segmamba.segmamba import SegMamba
from monai.losses import DiceLoss
from object_oriented_with_monai import Blackmamba


def objective(trial, directory):
    # Hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    lr_step_size = trial.suggest_int('lr_step_size', 20, 100, step=10)
    batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8])

    # Initialize Blackmamba with suggested parameters
    experiment = Blackmamba(root_dir=directory,
                            version_name=f"trial_{trial.number}",
                            epochs=10,
                            batch_size=batch_size,
                            lr=lr,
                            lr_step_size=lr_step_size)

    metric = experiment.run()
    return metric


def run_study(directory):
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, directory), n_trials=50)
    best_trial = study.best_trial

    print(f"Best trial: {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")


if __name__ == "__main__":
    run_study('/datasets/tdt4265/mic/asoca')
