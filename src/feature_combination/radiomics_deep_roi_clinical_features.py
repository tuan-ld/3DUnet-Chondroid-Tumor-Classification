import pandas as pd

# Define file paths
radiomics_file = './radiomics_features.csv'
roi_deep_features_file = './roi_deep_features.csv'
clinical_file = './clinical_features.csv'
output_file = './clinical_radiomics_deep_roi_features.csv'

# Load data
radiomics_df = pd.read_csv(radiomics_file)
roi_deep_features_df = pd.read_csv(roi_deep_features_file)
clinical_df = pd.read_csv(clinical_file)

# Ensure 'patient_id' is of the same type in all DataFrames
radiomics_df['patient_id'] = radiomics_df['patient_id'].astype(int)
roi_deep_features_df['patient_id'] = roi_deep_features_df['patient_id'].astype(int)
clinical_df['patient_id'] = clinical_df['patient_id'].astype(int)

# Merge radiomics, clinical, and ROI-based deep learning features on 'patient_id'
combined_df = pd.merge(radiomics_df, clinical_df, on='patient_id')
combined_df = pd.merge(combined_df, roi_deep_features_df, on='patient_id')

# Save combined data to CSV
combined_df.to_csv(output_file, index=False)
print(f"Merged data exported to {output_file}")
