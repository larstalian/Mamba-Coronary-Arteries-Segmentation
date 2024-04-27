import optuna
import torch
from main import get_transforms, load_data, create_data_loaders, train_epoch, validate
from model_segmamba.segmamba import SegMamba
from monai.losses import DiceLoss
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete


def objective(trial, root_dir):
    # Suggesting hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = 1
    lr_step_size = trial.suggest_int('lr_step_size', 20, 100, step=10)

    # Using main.py functions to set up data loaders and transforms
    train_transforms, val_transforms = get_transforms()
    train_ds, val_ds = load_data(root_dir, train_transforms, val_transforms)
    train_loader, val_loader = create_data_loaders(train_ds, val_ds, batch_size)

    # Model and training configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegMamba(in_chans=1, out_chans=1, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = DiceLoss(sigmoid=True)
    writer = SummaryWriter()  # Tensorboard summary writer
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Training and validation using main.py functions
    for epoch in range(10):  # Typically, a reduced number of epochs for hyperparameter tuning
        train_epoch(model, train_loader, device, optimizer, loss_function, writer, epoch, len(train_loader))
        if (epoch + 1) % 2 == 0:  # Validation frequency
            metric = validate(model, val_loader, device, dice_metric, post_trans, writer, epoch)
            trial.report(metric, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return metric


def run_study(root_dir):
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, root_dir), n_trials=50)
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")


if __name__ == "__main__":
    run_study('/datasets/tdt4265/mic/asoca')
