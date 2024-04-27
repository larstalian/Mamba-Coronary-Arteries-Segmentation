import logging
import os
import sys

import monai
import torch
from monai.data import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    SpatialCropd,
    RandFlipd,
    RandRotated,
    RandShiftIntensityd,
    EnsureTyped,
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np

from model_segmamba.segmamba import SegMamba


def get_transforms():
    spacial_crop = SpatialCropd(keys=["img", "seg"], roi_start=[50, 50, 50], roi_end=[178, 178, 178])
    train_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        spacial_crop,
        RandFlipd(keys=["img", "seg"], spatial_axis=[0, 1, 2]),
        RandRotated(
            keys=["img", "seg"],
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
        RandShiftIntensityd(keys=["img"], offsets=0.1, prob=0.5),
        ScaleIntensityd(keys="img"),
        EnsureTyped(keys=["img", "seg"]),

    ])
    val_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        RandFlipd(keys=["img", "seg"], spatial_axis=[0, 1, 2]),
        RandRotated(
            keys=["img", "seg"],
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
        RandShiftIntensityd(keys=["img"], offsets=0.1, prob=0.5),
        ScaleIntensityd(keys="img"),
        EnsureTyped(keys=["img", "seg"]),
    ])
    return train_transforms, val_transforms


def load_data(root_dir, train_transforms, val_transforms):
    data_dicts = [{
        'img': os.path.join(root_dir, 'Diseased/CTCA', f'Diseased_{i}.nrrd'),
        'seg': os.path.join(root_dir, 'Diseased/Annotations', f'Diseased_{i}.nrrd')
    } for i in range(1, 20)]
    train_files, val_files = monai.data.utils.partition_dataset(data_dicts, ratios=[0.8, 0.2], shuffle=True)
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    return train_ds, val_ds


def create_data_loaders(train_ds, val_ds, batch_size=1):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=6, collate_fn=list_data_collate)
    return train_loader, val_loader


def train_epoch(model, train_loader, device, optimizer, loss_function, writer, epoch, epoch_len):
    model.train()
    epoch_loss = 0
    for step, batch_data in enumerate(train_loader):
        inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    return epoch_loss / len(train_loader)


def validate(model, val_loader, device, dice_metric, post_trans, writer, epoch):
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs, y=val_labels)
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        writer.add_scalar("val_mean_dice", metric, epoch)
        plot_2d_or_3d_image(val_images, epoch, writer, index=0, tag="image")
        plot_2d_or_3d_image(val_labels, epoch, writer, index=0, tag="label")
        plot_2d_or_3d_image(val_outputs, epoch, writer, index=0, tag="output")
    return metric


def main(root_dir, epochs, lr_step_size, version_name):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    train_transforms, val_transforms = get_transforms()
    train_ds, val_ds = load_data(root_dir, train_transforms, val_transforms)
    train_loader, val_loader = create_data_loaders(train_ds, val_ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegMamba(in_chans=1, out_chans=1, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384]).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)

    best_lr = 0.01065901858877665
    best_lr_step_size = 70

    optimizer = torch.optim.SGD(model.parameters(), best_lr, weight_decay=1e-5, momentum=0.99, nesterov=True)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, best_lr_step_size, 1e-6)

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    writer = SummaryWriter()

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Update the learning rate
        scheduler.step()

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    roi_size = (96, 96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), f"run_{version_name}_epochs_{epochs}_lr_step_{lr_step_size}.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the coronary arteries segmentation model")
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train')
    parser.add_argument('--idun', action='store_true', help='Use the Idun cluster path for datasets')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    root_dir = '/datasets/tdt4265/mic/asoca'
    if args.idun:
        root_dir = '/cluster/projects/vc/data/mic/open/Heart/ASOCA'

    if args.tune:
        from hyperparam_tuning import run_study

        run_study(root_dir)
    else:
        iteration_version_name = 1.3
        lr_step_size = 50
        main(root_dir, args.epochs, lr_step_size, iteration_version_name)
