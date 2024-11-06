# data_preprocessing.py

import os
import numpy as np
import nibabel as nib  # For MRI NIfTI file handling
import cv2  # For resizing images
from sklearn.preprocessing import MinMaxScaler

# Directory paths
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"

# Image parameters
TARGET_SIZE = (128, 128)  # Resize images to 128x128 (adjust as needed)

def load_mri_image(file_path):
    """
    Load MRI image from a NIfTI file (.nii or .nii.gz format).
    """
    image = nib.load(file_path)
    image_data = image.get_fdata()
    return image_data

def resize_image(image, target_size=TARGET_SIZE):
    """
    Resize each slice of the 3D MRI image to the target size.
    """
    resized_slices = [cv2.resize(slice_, target_size, interpolation=cv2.INTER_AREA) for slice_ in image]
    return np.array(resized_slices)

def normalize_image(image):
    """
    Normalize image intensities to the range [0, 1].
    """
    scaler = MinMaxScaler()
    image_flat = image.flatten().reshape(-1, 1)
    normalized_flat = scaler.fit_transform(image_flat)
    normalized_image = normalized_flat.reshape(image.shape)
    return normalized_image

def preprocess_and_save_images():
    """
    Main function to preprocess and save MRI images.
    """
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    
    for file_name in os.listdir(RAW_DATA_DIR):
        if file_name.endswith(".nii") or file_name.endswith(".nii.gz"):
            print(f"Processing {file_name}...")
            
            # Load and preprocess image
            file_path = os.path.join(RAW_DATA_DIR, file_name)
            image = load_mri_image(file_path)
            resized_image = resize_image(image)
            normalized_image = normalize_image(resized_image)
            
            # Save processed image
            processed_file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
            nib.save(nib.Nifti1Image(normalized_image, affine=np.eye(4)), processed_file_path)
            print(f"Saved processed image: {processed_file_path}")

if __name__ == "__main__":
    preprocess_and_save_images()
