import argparse
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
    RandSpatialCropd,
    Orientationd,
    EnsureTyped,
    RandHistogramShiftd,
    SpatialCropd,
    RandShiftIntensityd
)
from monai.visualize import plot_2d_or_3d_image
from monai.data import SmartCacheDataset

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_segmamba.segmamba import SegMamba
from torch.cuda.amp import GradScaler, autocast


class CombinedLoss(torch.nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5, sigmoid=True):
        super(CombinedLoss, self).__init__()
        self.dice_loss = monai.losses.DiceLoss(sigmoid=sigmoid)
        self.cross_entropy_loss = torch.nn.BCEWithLogitsLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, outputs, targets):
        dice_loss = self.dice_loss(outputs, targets)
        ce_loss = self.cross_entropy_loss(outputs, targets)
        return self.weight_dice * dice_loss + self.weight_ce * ce_loss


class Blackmamba:
    def __init__(self, root_dir, version_name, epochs, batch_size, lr, lr_step_size):
        self.root_dir = root_dir
        self.version_name = version_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.initialize_components()
        self.scaler = GradScaler()

    def initialize_components(self):
        self.train_transforms, self.val_transforms = self.get_transforms()
        self.train_ds, self.val_ds = self.load_data()
        self.train_loader, self.val_loader = self.create_data_loaders()
        self.setup_model()

    def setup_model(self):
        self.model = SegMamba(in_chans=1, out_chans=1, depths=[2, 2, 2, 2], feat_size=[48, 96, 132, 224]).cuda()
        self.loss_function = CombinedLoss(weight_dice=0.8, weight_ce=0.2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, weight_decay=1e-5, momentum=0.99,
                                         nesterov=True)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5,
                                                                    verbose=True)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.writer = SummaryWriter()
        monai.config.print_config()
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    @staticmethod
    def get_transforms():
        crop_size = (256, 256, 128)
        # sp_crop = SpatialCropd(keys=["img", "seg"], roi_start=[128, 128, 64], roi_end=[256, 256, 128])

        train_transforms = Compose([
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Orientationd(keys=["img", "seg"], axcodes="RAS"),
            # sp_crop,
            RandSpatialCropd(keys=["img", "seg"], roi_size=crop_size, random_size=False),
            # RandFlipd(keys=["img", "seg"], spatial_axis=[0, 1, 2]),
            # RandRotated(keys=["img", "seg"],
            #             range_x=(0.0, 10.0),
            #             range_y=(0.0, 10.0),
            #             range_z=(0.0, 10.0),
            #             prob=0.1,
            #             keep_size=True),
            ScaleIntensityd(keys="img"),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["img", "seg"]),
            RandHistogramShiftd(keys=["img"], num_control_points=3, prob=0.1),
        ])

        val_transforms = Compose([
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            Orientationd(keys=["img", "seg"], axcodes="RAS"),
            # sp_crop,
            ScaleIntensityd(keys="img"),
            EnsureTyped(keys=["img", "seg"]),
        ])

        return train_transforms, val_transforms

    def load_data(self):
        data_dicts = [{
            'img': os.path.join(self.root_dir, 'Diseased/CTCA', f'Diseased_{i}.nrrd'),
            'seg': os.path.join(self.root_dir, 'Diseased/Annotations', f'Diseased_{i}.nrrd')
        } for i in range(1, 20)]

        cache_num_train = min(10 * self.batch_size, len(data_dicts) - 1)
        cache_num_val = min(5 * self.batch_size, len(data_dicts) - 1)
        train_files, val_files = monai.data.utils.partition_dataset(data_dicts, ratios=[0.8, 0.2], shuffle=True)
        train_ds = SmartCacheDataset(data=train_files, transform=self.train_transforms, replace_rate=0.2,
                                     cache_num=cache_num_train, num_init_workers=4, num_replace_workers=6)
        val_ds = SmartCacheDataset(data=val_files, transform=self.val_transforms, replace_rate=0.2,
                                   cache_num=cache_num_val, num_init_workers=2, num_replace_workers=4)

        return train_ds, val_ds

    def create_data_loaders(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=6, collate_fn=list_data_collate)
        return train_loader, val_loader

    def train_epoch(self, train_loader, device, optimizer, loss_function, writer, epoch, epoch_len):
        self.model.train()
        epoch_loss = 0
        for step, batch_data in enumerate(train_loader):
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()

            # Runs the forward pass with autocasting.
            with autocast():
                outputs = self.model(inputs)
                loss = loss_function(outputs, labels)

            # Scales the loss, and calls backward() to create scaled gradients
            self.scaler.scale(loss).backward()

            # Log gradient norms
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            writer.add_scalar("gradient_norm", total_norm, epoch_len * epoch + step)

            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.step(optimizer)

            # Updates the scale for next iteration
            self.scaler.update()

            epoch_loss += loss.item()
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        return epoch_loss / len(train_loader)

    def validate(self, val_loader, device, dice_metric, post_trans, writer, epoch):
        self.model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, self.model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            writer.add_scalar("val_mean_dice", metric, epoch)
            plot_2d_or_3d_image(val_images, epoch, writer, index=0, tag="image")
            plot_2d_or_3d_image(val_labels, epoch, writer, index=0, tag="label")
            plot_2d_or_3d_image(val_outputs, epoch, writer, index=0, tag="output")
        return metric

    def run(self):
        val_interval = 2
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = list()
        metric_values = list()
        writer = SummaryWriter()
        for epoch in range(self.epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{self.epochs}")
            self.model.train()
            epoch_loss = 0
            step = 0
            for batch_data in self.train_loader:
                step += 1
                inputs, labels = batch_data["img"].cuda(), batch_data["seg"].cuda()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()

                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(self.train_ds) // self.train_loader.batch_size
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            # Update the learning rate
            self.scheduler.step(epoch_loss)

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            writer.add_scalar("learning_rate", current_lr, epoch)

            if (epoch + 1) % val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    val_images = None
                    val_labels = None
                    val_outputs = None
                    for val_data in self.val_loader:
                        val_images, val_labels = val_data["img"].cuda(), val_data["seg"].cuda()
                        roi_size = (96, 96, 96)
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, self.model)
                        val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]
                        # compute metric for current iteration
                        self.dice_metric(y_pred=val_outputs, y=val_labels)
                    # aggregate the final mean dice result
                    metric = self.dice_metric.aggregate().item()
                    # reset the status for next validation round
                    self.dice_metric.reset()

                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(self.model.state_dict(),
                                   f"run_{self.version_name}_epochs_{self.epochs}_lr_step_{self.lr_step_size}.pth")
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

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Run the coronary arteries segmentation model")
        parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
        parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train')
        parser.add_argument('--idun', action='store_true', help='Use the Idun cluster path for datasets')
        parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
        parser.add_argument('--lr_step', type=int, default=100, help='Step size for learning rate scheduler')
        parser.add_argument('--it', type=float, default=2.5, help='Version name for the iteration')
        parser.add_argument('--batch', type=int, default=1, help='Batch size')
        return parser.parse_args()


if __name__ == "__main__":
    args = Blackmamba.parse_arguments()

    training_directory = '/datasets/tdt4265/mic/asoca'
    if args.idun:
        training_directory = '/cluster/projects/vc/data/mic/open/Heart/ASOCA'

    if args.tune:
        from hyperparam_tuning import run_study

        run_study(training_directory)
    else:
        Blackmamba(root_dir=training_directory,
                   version_name=args.it,
                   epochs=args.epochs,
                   lr=args.lr,
                   lr_step_size=args.lr_step,
                   batch_size=args.batch
                   ).run()
