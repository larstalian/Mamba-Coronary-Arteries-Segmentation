

import torch 
from model_segmamba.segmamba import SegMamba

# t1 = torch.rand(1, 4, 128, 128, 128).cuda()


# model = SegMamba(in_chans=4,
#                  out_chans=4,
#                  depths=[2,2,2,2],
#                  feat_size=[48, 96, 192, 384]).cuda()

# out = model(t1)

# print(out.shape)

import nibabel as nib
import numpy as np




import numpy as np

def extract_patches(volume, patch_size, stride):
    z, y, x = volume.shape
    patches = []
    start_points = [
        list(range(0, max(d - s, 1), s)) + [d - s] for d, s in zip(volume.shape, stride)
    ]  # Handle the end case by ensuring the last patch aligns with the edge of the volume

    for start_z in start_points[0]:
        for start_y in start_points[1]:
            for start_x in start_points[2]:
                end_z = min(start_z + patch_size[0], z)
                end_y = min(start_y + patch_size[1], y)
                end_x = min(start_x + patch_size[2], x)
                patch = volume[start_z:end_z, start_y:end_y, start_x:end_x]
                if patch.shape == tuple(patch_size):
                    patches.append((patch, (start_z, start_y, start_x)))
    return patches

def infer_full_volume(model, full_volume, patch_size, stride, device='cuda'):
    model.eval()
    output_volume = np.zeros(full_volume.shape)
    count_map = np.zeros(full_volume.shape)  # To average the overlaps

    patches = extract_patches(full_volume, patch_size, stride)
    
    with torch.no_grad():
        for patch, start_coords in patches:
            start_z, start_y, start_x = start_coords
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimension
            patch_tensor = patch_tensor.float() 
            
            prediction = model(patch_tensor)
            prediction = torch.sigmoid(prediction).cpu().numpy()  # Convert to probabilities
            
            output_volume[start_z:start_z+patch_size[0], start_y:start_y+patch_size[1], start_x:start_x+patch_size[2]] += prediction[0, 0]
            count_map[start_z:start_z+patch_size[0], start_y:start_y+patch_size[1], start_x:start_x+patch_size[2]] += 1

    output_volume /= count_map
    return output_volume

file_path = '/datasets/tdt4265/mic/asoca/Diseased/CTCA/Diseased_1.nrrd'

# Load the NIfTI file
#nifti_file = nib.load(file_path)
import nrrd

# Convert it to a numpy array
#full_image = nifti_file.get_fdata()
full_image, header = nrrd.read(file_path)

# Assuming 'full_image' is your (512, 512, 224) numpy array loaded from a file or similar source.
full_image_normalized = (full_image - np.mean(full_image)) / np.std(full_image)  # Example normalization

# Define patch size and stride
patch_size = (128, 128, 32)
stride = (64, 64, 16)

# Load model
model = SegMamba(in_chans=1, out_chans=1, depths=[2,2,2,2], feat_size=[48, 96, 192, 384])
model.load_state_dict(torch.load('epoch_200_model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



predicted_volume = infer_full_volume(model, full_image_normalized, patch_size, stride)


output_path = 'prediction3.nrrd'  # Specify the full path where you want to save the file

# Save the numpy array as a NRRD file
nrrd.write(output_path, predicted_volume)

print(f"Saved predicted volume as NRRD file at: {output_path}")



# predicted_image = nib.Nifti1Image(predicted_volume, affine=None)  # Provide affine if available

# nib.save(predicted_volume, 'prediction.nii.gz')

nifti_img = nib.Nifti1Image(predicted_volume.astype(np.float32), affine=np.eye(4))

# Specify the path where you want to save the Nifti file
output_path = 'prediction2.nii.gz'  # Change to .nii.gz extension

# Save the Nifti image
nib.save(nifti_img, output_path)

print(f"Saved predicted volume as NIfTI file at: {output_path}")



def dice_coefficient(preds, targets, smooth=1e-5):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()  # Convert probabilities to binary values
    intersection = (preds * targets).sum(dim=[2, 3, 4])  # Intersection
    union = preds.sum(dim=[2, 3, 4]) + targets.sum(dim=[2, 3, 4])  # Union
    dice = (2. * intersection + smooth) / (union + smooth)  # Dice coefficient
    return dice.mean()  # Average over all batches

file_path = '/datasets/tdt4265/mic/asoca/Diseased/Annotations/Diseased_1.nrrd'
label, header = nrrd.read(file_path)

print(dice_coefficient(torch.tensor(predicted_volume), torch.tensor(label)))