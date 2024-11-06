import os
import csv
import pandas as pd
from radiomics import featureextractor

# Define default paths
DATASET_DIR = './data'
PARAMS_FILE = './Params.yaml'
OUTPUT_FEATURES_FILE = './radiomics_features.csv'
MISMATCH_FILE = './image_mask_mismatch.csv'

def extract_radiomics_features(dataset_dir=DATASET_DIR, params_file=PARAMS_FILE, output_file=OUTPUT_FEATURES_FILE, mismatch_file=MISMATCH_FILE):
    """
    Extracts radiomics features from images and masks in the specified dataset directory, 
    saves them to a CSV file, and returns a DataFrame of the extracted features.
    Mismatched image and mask paths are recorded in a separate CSV file.
    """
    # Initialize feature extractor with specified YAML parameters
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    extractor.settings['geometryTolerance'] = 1e-3  # Adjust tolerance to handle geometry mismatches

    # Initialize lists for features and mismatches
    features_list = []
    mismatch_paths = []
    fieldnames = None

    # Loop through train and test subsets
    for subset in ['train', 'test']:
        subset_dir = os.path.join(dataset_dir, subset)
        for class_label in os.listdir(subset_dir):
            class_dir = os.path.join(subset_dir, class_label)
            for image_name_and_type in os.listdir(class_dir):
                image_dir = os.path.join(class_dir, image_name_and_type)

                # Define paths for image and mask
                image_path = os.path.join(image_dir, f"{image_name_and_type.split('_')[-1]}.nii.gz")
                mask_path = os.path.join(image_dir, f"{image_name_and_type.split('_')[-1]}-seg.nii.gz")

                # Check if both image and mask exist
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    try:
                        # Extract features
                        result = extractor.execute(image_path, mask_path)
                        
                        # Prepare feature row for DataFrame and CSV
                        row = {
                            'Subset': subset,
                            'Class_Label': class_label,
                            'Image_Path': image_path,
                            'Mask_Path': mask_path
                        }
                        row.update(result)
                        features_list.append(row)

                        # Initialize CSV headers based on result keys (for writing to file)
                        if fieldnames is None:
                            fieldnames = list(row.keys())
                            with open(output_file, 'w', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()

                        # Write each row to the CSV file
                        with open(output_file, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writerow(row)

                    except Exception as e:
                        print(f"Error processing image {image_path} and mask {mask_path}: {str(e)}")
                        mismatch_paths.append((image_path, mask_path))
                else:
                    print(f"Image or mask missing for {image_path} or {mask_path}")
                    mismatch_paths.append((image_path, mask_path))

    # Save mismatched paths to a CSV file
    if mismatch_paths:
        with open(mismatch_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image_Path', 'Mask_Path'])
            writer.writerows(mismatch_paths)
        print(f"Mismatched paths saved to {mismatch_file}")

    # Convert features list to DataFrame and return
    features_df = pd.DataFrame(features_list)
    return features_df

# For direct script execution
if __name__ == "__main__":
    extract_radiomics_features()
