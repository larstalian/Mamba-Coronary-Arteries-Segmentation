import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.optim as optim
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, EnsureTyped, EnsureChannelFirstd,
    Orientationd, Spacingd, CenterSpatialCropd
)
from monai.data import DataLoader, Dataset, partition_dataset
from monai.config import print_config
from model_segmamba.segmamba import SegMamba
from torch.optim import Adam
import monai


# Configuration
# root_dir = '/datasets/tdt4265/mic/asoca'
root_dir = '/cluster/projects/vc/data/mic/open/Heart/ASOCA'
num_epochs = 100
batch_size = 4
learning_rate = 0.0002

# Define MONAI transforms with additional transformations for 3D images
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    # CenterSpatialCropd(keys=["image", "label"], roi_size=[32, 64, 64]),  # Crop the image
    EnsureTyped(keys=["image", "label"]),
])
# Prepare dataset and dataloader
data_dicts = [{
    'image': os.path.join(root_dir, 'Diseased/CTCA', f'Diseased_{i}.nrrd'),
    'label': os.path.join(root_dir, 'Diseased/Annotations', f'Diseased_{i}.nrrd')
} for i in range(1, 20)]

train_files, val_files = partition_dataset(data_dicts, ratios=[0.8, 0.2], shuffle=True)

train_ds = Dataset(data=train_files, transform=transforms)
val_ds = Dataset(data=val_files, transform=transforms)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# Model setup
model = SegMamba(in_chans=1, out_chans=4, depths=[2,2,2,2], feat_size=[48, 96, 192, 384]).cuda()
optimizer = Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter()

# Loss Function
def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Helper function to perform a training or validation epoch
def run_epoch(loader, is_training=True):
    epoch_loss = 0.0
    if is_training:
        model.train()
    else:
        model.eval()
    
    for data in loader:
        inputs, labels = data
      #  print("Batch data keys:", data.keys())  # Should print: dict_keys(['image', 'label'])
        inputs, labels = data['image'], data['label']
      #  print("Inputs type before cuda:", type(inputs))  # Should be <class 'torch.Tensor'>
       # print("Labels type before cuda:", type(labels))  # Should be <class 'torch.Tensor'>

        #print(inputs)
        inputs, labels = inputs.cuda(), labels.cuda()
      #  if isinstance(labels, monai.data.meta_tensor.MetaTensor):
       #     labels = labels.tensor

        if is_training:
            optimizer.zero_grad()

      #  labels = labels.long()  # Cast labels to long
        labels = labels.to(dtype=torch.long)  # Cast labels to long

       # print('s')
        outputs = model(inputs)
        # print('s')
        loss = dice_loss(outputs, labels)
        labels = labels.squeeze(1)
        F.cross_entropy(outputs, labels)
        loss += F.cross_entropy(outputs, labels)
        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

# Training Loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_loss = run_epoch(train_loader, is_training=True)
    val_loss = run_epoch(val_loader, is_training=False)

    # Logging to TensorBoard
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Val', val_loss, epoch)

    # Save the model if validation loss has decreased
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model2.pth')
        print(f'Saved Best Model at Epoch {epoch+1}')

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

writer.close()
print('Finished Training')
