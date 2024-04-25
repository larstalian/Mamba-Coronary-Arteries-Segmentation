import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, EnsureTyped, EnsureChannelFirstd,
    Orientationd, Spacingd, CenterSpatialCropd,SpatialPadd, RandFlipd, RandRotated, RandSpatialCropd, RandShiftIntensityd
)
from monai.data import DataLoader, Dataset, partition_dataset
from monai.config import print_config
from model_segmamba.segmamba import SegMamba
from torch.cuda.amp import GradScaler, autocast
from monai.data import SmartCacheDataset
from blackmamba.early_stopping import EarlyStopping

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
    CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128]), # Center padding for cybele computer
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

train_ds = SmartCacheDataset(data=train_files, transform=transforms, replace_rate=0.2, cache_num=cache_num_train, num_init_workers=4, num_replace_workers=4)
val_ds = SmartCacheDataset(data=val_files, transform=transforms, replace_rate=0.2, cache_num=cache_num_val, num_init_workers=2, num_replace_workers=2)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=6)

# Model setup
model = SegMamba(in_chans=1, out_chans=4, depths=[2,2,2,2], feat_size=[48, 96, 192, 384]).cuda()

warmup_epochs = 5
warmup_start_lr = 1e-6  # Start with a very low learning rate
base_lr = learning_rate  # Target learning rate after warm-up

writer = SummaryWriter()

# Modify the optimizer setup to start with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=warmup_start_lr)
scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

early_stopping = EarlyStopping(patience=10, verbose=True)


def log_memory():
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

def adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr, warmup_start_lr):
    if epoch < warmup_epochs:
        lr = warmup_start_lr + (base_lr - warmup_start_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def sigmoid_activation(x):
    return torch.sigmoid(x)

def dice_coefficient(pred, target, smooth=1.):
    pred = sigmoid_activation(pred)  # Convert logits to probabilities
    pred = (pred > 0.5).float()  # Threshold probabilities to get binary predictions
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def dice_loss(pred, target, smooth=1.):
    pred = sigmoid_activation(pred)  # Convert logits to probabilities
    pred = (pred > 0.5).float()  # Threshold probabilities to get binary predictions
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename="checkpoint.pth.tar"):
    state = torch.load(filename)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    return state['epoch'], state['best_val_loss']

# Helper function to perform a training or validation epoch

def run_epoch(loader, is_training=True):
    epoch_loss = 0.0
    epoch_dice = 0.0
    scaler = GradScaler()  # Initialize the gradient scaler

    if is_training:
        model.train()
    else:
        model.eval()

    for data in loader:
        inputs, labels = data['image'], data['label']
        inputs, labels = inputs.cuda(), labels.cuda()
        labels = labels.to(dtype=torch.long)

        if is_training:
            optimizer.zero_grad()

        # Using automatic mixed precision
        with autocast():
            outputs = model(inputs)
            dice_loss_val = dice_loss(outputs, labels)  # Calculate Dice loss
            ce_loss = F.cross_entropy(outputs, labels.squeeze(1))  # Calculate cross-entropy loss
            loss = dice_loss_val + ce_loss  # Combine losses

        if not is_training:
            with autocast():
                dice = dice_coefficient(outputs, labels)  # Compute the Dice coefficient
            epoch_dice += dice.item()

        if is_training:
            # Backward pass with automatic mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log gradient norms to TensorBoard
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
            writer.add_scalar('Gradient_norm', total_norm, epoch)

        epoch_loss += loss.item()

    # Calculate average loss and Dice for the epoch
    num_batches = len(loader)
    avg_epoch_loss = epoch_loss / num_batches
    if not is_training:
        avg_epoch_dice = epoch_dice / num_batches
        return avg_epoch_loss, avg_epoch_dice
    else:
        return avg_epoch_loss


# Training and Validation Loop
best_val_loss = float('inf')
best_val_dice = 0.0

for epoch in range(num_epochs):
    # Adjust learning rate during the warm-up phase
    adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr, warmup_start_lr)

    train_loss = run_epoch(train_loader, is_training=True)
    val_loss, val_dice = run_epoch(val_loader, is_training=False)
    
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Val', val_loss, epoch)
    writer.add_scalar('Dice/Val', val_dice, epoch)

    # Only use the scheduler after warm-up
    if epoch >= warmup_epochs:
        scheduler_lr.step(val_loss)  # Update the learning rate based on the validation loss

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model2.pth')
        print(f'Saved Best Model at Epoch {epoch+1}')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'optimizer': optimizer.state_dict(),
        }, filename='best_model_checkpoint.pth.tar')

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), 'best_model_dice.pth')

    if val_dice > best_val_dice:
        best_val_dice = val_dice
    

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Dice: {val_dice}')

    # Early stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break


writer.close()
print('Finished Training')