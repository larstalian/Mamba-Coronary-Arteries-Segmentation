import torch
from torch.utils.data import Dataset, DataLoader
import nrrd
import os
import numpy as np


class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the data (should end at .../mic/asoca).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()
        self.target_shape = (512, 512, 224)
        
    def _load_samples(self):
        samples = []
        categories = ['Diseased', 'Normal']
        for category in categories:
            ctca_path = os.path.join(self.root_dir, category, 'CTCA')
            annotations_path = os.path.join(self.root_dir, category, 'Annotations')

            image_files = sorted([os.path.join(ctca_path, f) for f in os.listdir(ctca_path) if f.endswith('.nrrd')])
            label_files = sorted([os.path.join(annotations_path, f) for f in os.listdir(annotations_path) if f.endswith('.nrrd')])
            
            samples.extend(zip(image_files, label_files))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_path = self.samples[idx]
        image, _ = nrrd.read(image_path)
        print(image.shape)
        label, _ = nrrd.read(label_path)

        image = pad_image(image, self.target_shape)
        label = pad_image(label, self.target_shape)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        image = torch.from_numpy(image).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.FloatTensor)

        return image, label

def pad_image(img, target_shape):
    pad_width = [(0, max(target_shape[dim] - img.shape[dim], 0)) for dim in range(len(target_shape))]
    img_padded = np.pad(img, pad_width, mode='constant', constant_values=0)
    return img_padded

