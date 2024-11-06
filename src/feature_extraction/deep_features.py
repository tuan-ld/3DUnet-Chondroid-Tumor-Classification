import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import nibabel as nib
import pandas as pd

# Load pre-trained EfficientNet model
def load_efficientnet_model():
    model = models.efficientnet_b0(pretrained=True)
    model.eval()  # Set model to evaluation mode
    return model

# Function to preprocess the image
def preprocess_image(image_array):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match EfficientNet input size
        transforms.Grayscale(num_output_channels=3),  # Convert single-channel to three-channel (RGB)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(image_array)
    return preprocess(image)

# Function to extract deep features without ROI mask
def extract_deep_features(image_path, model):
    nii_img = nib.load(image_path)
    nii_data = nii_img.get_fdata()
    all_features = []

    for i in range(nii_data.shape[-1]):
        image_array = nii_data[:, :, i]
        image = preprocess_image(image_array).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            features = model(image)

        features = features.cpu().numpy()
        all_features.append(features)

    mean_features = np.mean(np.stack(all_features, axis=-1), axis=-1)
    feature_dict = {f'efficientnet_mean_feature_{i+1}': mean_features[0][i] for i in range(mean_features.shape[-1])}
    return feature_dict

# Function to extract deep features with ROI mask
def extract_deep_features_roi_mask(image_path, mask_path, model):
    nii_img = nib.load(image_path)
    nii_data = nii_img.get_fdata()
    nii_mask = nib.load(mask_path)
    mask_data = nii_mask.get_fdata()
    all_features = []

    for i in range(nii_data.shape[-1]):
        image_array = nii_data[:, :, i]
        mask_array = mask_data[:, :, i]
        roi_image_array = image_array * mask_array

        roi_image = preprocess_image(roi_image_array).unsqueeze(0)

        with torch.no_grad():
            features = model(roi_image)

        features = features.cpu().numpy()
        all_features.append(features)

    mean_features = np.mean(np.stack(all_features, axis=-1), axis=-1)
    feature_dict = {f'efficientnet_mean_feature_{i+1}': mean_features[0][i] for i in range(mean_features.shape[-1])}
    return feature_dict

# Main function to save deep features to separate files
def save_deep_features(radiomics_features_file, dataset_dir, deep_features_file, roi_deep_features_file):
    model = load_efficientnet_model()
    radiomics_df = pd.read_csv(radiomics_features_file)
    deep_features_dict = {}
    roi_deep_features_dict = {}

    for index, row in radiomics_df.iterrows():
        image_path = os.path.join(dataset_dir, row['Image_Path'])
        mask_path = os.path.join(dataset_dir, row['Mask_Path']) if 'Mask_Path' in row else None

        try:
            # Extract and save global deep features
            deep_features = extract_deep_features(image_path, model)
            deep_features_dict[index] = deep_features
            
            # Extract and save ROI-based deep features if mask path is available
            if mask_path:
                roi_deep_features = extract_deep_features_roi_mask(image_path, mask_path, model)
                roi_deep_features_dict[index] = roi_deep_features

        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")

    # Convert dictionaries to DataFrames and save
    pd.DataFrame.from_dict(deep_features_dict, orient='index').to_csv(deep_features_file, index=False)
    pd.DataFrame.from_dict(roi_deep_features_dict, orient='index').to_csv(roi_deep_features_file, index=False)
    print(f"Deep features saved to {deep_features_file}")
    print(f"ROI-based deep features saved to {roi_deep_features_file}")
