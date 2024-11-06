import pandas as pd

# Define file paths
radiomics_file = './radiomics_features.csv'
deep_features_file = './deep_features.csv'
output_file = './radiomics_deep_learning_features.csv'

# Load data
radiomics_df = pd.read_csv(radiomics_file)
deep_features_df = pd.read_csv(deep_features_file)

# Ensure 'patient_id' is of the same type in both DataFrames
radiomics_df['patient_id'] = radiomics_df['patient_id'].astype(int)
deep_features_df['patient_id'] = deep_features_df['patient_id'].astype(int)

# Merge radiomics and deep learning features on 'patient_id'
combined_df = pd.merge(radiomics_df, deep_features_df, on='patient_id')

# Save combined data to CSV
combined_df.to_csv(output_file, index=False)
print(f"Merged data exported to {output_file}")
