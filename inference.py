import torch
import nrrd
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityd, CenterSpatialCropd, EnsureTyped, SpatialCropd
from model_segmamba.segmamba import SegMamba  # Import SegMamba from your module
name = '256x256x16'
# Model initialization
model = SegMamba(in_chans=1, out_chans=1, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384]).cuda()

# Load the model weights
model.load_state_dict(torch.load('/work/larststa/blackmamba/blackmamba/best_model_dice.pth'))

# Model in evaluation mode
model.eval()

# Define the transforms
transforms2 = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    SpatialCropd(keys=["image"], roi_start=[300, 220, 80], roi_end=[428, 348, 208]), 
    EnsureTyped(keys=["image"]),
])

transforms_labels = Compose([
    LoadImaged(keys=["label"]),
    EnsureChannelFirstd(keys=["label"]),
    Orientationd(keys=["label"], axcodes="RAS"),
    SpatialCropd(keys=["label"], roi_start=[300, 220, 80], roi_end=[428, 348, 208]), 
    EnsureTyped(keys=["label"]),
])

# Paths to the image and label files
test_image_path = '/datasets/tdt4265/mic/asoca/Diseased/CTCA/Diseased_1.nrrd'
test_label_path = '/datasets/tdt4265/mic/asoca/Diseased/Annotations/Diseased_1.nrrd'

# Prepare the image and label data for the transform
test_data = {'image': test_image_path}
label_data = {'label': test_label_path}

# Apply the transforms
test_data = transforms2(test_data)
label_data = transforms_labels(label_data)

# Get the test and label tensors, ensuring they're on GPU
test_tensor = test_data['image'].unsqueeze(0).cuda()  # Add a batch dimension
label_tensor = label_data['label'].unsqueeze(0).cuda()  # Add a batch dimension for consistency

# Run inference
output = model(test_tensor)

# Convert the output to a numpy array and move it to CPU memory
output_data = output.cpu().detach().numpy()

# Select one channel of the output for simplicity
output_channel_data = output_data[0, 0, :, :, :]  # Select the first channel

# Save the inference output as an NRRD file
nrrd.write(f'inference{name}.nrrd', output_channel_data)

# Extract and save the label data
label_array = label_tensor.cpu().numpy()
label_channel_data = label_array[0, 0, :, :, :]
nrrd.write(f'labels1{name}.nrrd', label_channel_data)

print('Inference and label extraction completed.')
