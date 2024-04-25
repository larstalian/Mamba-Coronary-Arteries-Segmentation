import os

import numpy as np
import torch
from monai.data import DataLoader, partition_dataset, SmartCacheDataset
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import EarlyStopHandler, CheckpointSaver, LrScheduleHandler, ValidationHandler, MeanDice
from monai.losses import DiceLoss
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityd, RandFlipd, RandRotated,
    RandShiftIntensityd, CenterSpatialCropd, EnsureTyped
)
from torch.optim import Adam

from model_segmamba.segmamba import SegMamba

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Configuration
root_dir = '/datasets/tdt4265/mic/asoca'
# root_dir = '/cluster/projects/vc/data/mic/open/Heart/ASOCA'
num_epochs = 100
batch_size = 1
learning_rate = 0.0001

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2]),
    RandRotated(
        keys=["image", "label"],
        range_x=(0.0, 10.0),
        range_y=(0.0, 10.0),
        range_z=(0.0, 10.0),
        prob=0.1,
        keep_size=True,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
        dtype=np.float32
    ),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128]),  # Center padding for cybele computer
    # SpatialPadd(keys=["image", "label"], spatial_size=(512, 512, 224)),  # Padding (already included)
    EnsureTyped(keys=["image", "label"]),
])

# Prepare dataset and dataloader
data_dicts = [{
    'image': os.path.join(root_dir, 'Diseased/CTCA', f'Diseased_{i}.nrrd'),
    'label': os.path.join(root_dir, 'Diseased/Annotations', f'Diseased_{i}.nrrd')
} for i in range(1, 20)]

train_files, val_files = partition_dataset(data_dicts, ratios=[0.8, 0.2], shuffle=True)

# Using SmartCacheDataset for efficient caching
cache_num_train = min(10 * batch_size, len(train_files) - 1)
cache_num_val = min(5 * batch_size, len(val_files) - 1)

train_ds = SmartCacheDataset(data=train_files, transform=transforms, replace_rate=0.2, cache_num=cache_num_train,
                             num_init_workers=4, num_replace_workers=4)
val_ds = SmartCacheDataset(data=val_files, transform=transforms, replace_rate=0.2, cache_num=cache_num_val,
                           num_init_workers=2, num_replace_workers=2)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=6)

model = SegMamba(in_chans=1, out_chans=4, depths=[2,2,2,2], feat_size=[48, 96, 192, 384]).cuda()
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = Adam(model.parameters(), lr=0.0001)

# Metrics
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

# Handlers for evaluation
val_handlers = [
    EarlyStopHandler(patience=10, score_function=lambda x: -x['Mean_Dice'], trainer=None)  # Placeholder for trainer
]

# Define the evaluator first
val_evaluator = SupervisedEvaluator(
    device=torch.device("cuda:0"),
    val_data_loader=val_loader,
    network=model,
    key_val_metric={"Mean_Dice": MeanDice(dice_metric)},  # Correctly wrapped metric
    inferer=SlidingWindowInferer(),
    val_handlers=val_handlers
)

# Update trainer placeholder in EarlyStopHandler

# Train handlers
train_handlers = [
    LrScheduleHandler(lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1), print_lr=True),
    ValidationHandler(validator=val_evaluator, interval=1, epoch_level=True),
    CheckpointSaver(save_dir="./", save_dict={"model": model}, save_key_metric=True)
]

# Create the trainer with the correct reference to the evaluator
trainer = SupervisedTrainer(
    device=torch.device("cuda:0"),
    max_epochs=num_epochs,
    train_data_loader=train_loader,
    network=model,
    optimizer=optimizer,
    loss_function=loss_function,
    inferer=SlidingWindowInferer(),
    key_train_metric={"Mean_Dice": MeanDice(dice_metric)},  # Correctly wrapped metric
    train_handlers=train_handlers
)
val_handlers[0].trainer = trainer

# Start training
trainer.run()