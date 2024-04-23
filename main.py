import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from model_segmamba.segmamba import SegMamba
from blackmamba.asoca_dataset import MedicalImageDataset
import os

# Configuration
root_dir = '/datasets/tdt4265/mic/asoca'
num_epochs = 50
batch_size = 4
learning_rate = 0.001

# Setup TensorBoard
writer = SummaryWriter()

# Dataset and DataLoader
dataset = MedicalImageDataset(root_dir=root_dir)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model
model = SegMamba(in_chans=4, out_chans=4, depths=[2,2,2,2], feat_size=[48, 96, 192, 384]).cuda()

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
        inputs, labels = inputs.cuda(), labels.cuda()

        if is_training:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = dice_loss(outputs, labels) + F.cross_entropy(outputs, labels)
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
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Saved Best Model at Epoch {epoch+1}')

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

writer.close()
print('Finished Training')
