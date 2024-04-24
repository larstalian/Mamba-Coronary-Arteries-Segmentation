import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import LoadImaged, AddChanneld, Compose, SpatialPadd
from monai.data import Dataset
import os


class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = Compose([
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            SpatialPadd(keys=['image', 'label'], spatial_size=(512, 512, 224), mode='constant')
        ]) if transform is None else transform

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        categories = ['Diseased', 'Normal']
        for category in categories:
            ctca_path = os.path.join(self.root_dir, category, 'CTCA')
            annotations_path = os.path.join(self.root_dir, category, 'Annotations')

            image_files = sorted([os.path.join(ctca_path, f) for f in os.listdir(ctca_path) if f.endswith('.nrrd')])
            label_files = sorted([os.path.join(annotations_path, f) for f in os.listdir(annotations_path) if f.endswith('.nrrd')])
            
            samples.extend([{'image': img_file, 'label': lbl_file} for img_file, lbl_file in zip(image_files, label_files)])

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
<<<<<<< Updated upstream
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

=======
        data = self.samples[idx]
        data = self.transform(data)
        return data['image'], data['label']


def pad_image(tensor, target_shape):
    _, d, h, w = tensor.size()  # Assuming tensor has shape [C, D, H, W]
    padding = [0, max(target_shape[2] - w, 0),  # pad width
               0, max(target_shape[1] - h, 0),  # pad height
               0, max(target_shape[0] - d, 0)]  # pad depth
    tensor_padded = F.pad(tensor, padding, "constant", 0)
    return tensor_padded
>>>>>>> Stashed changes
