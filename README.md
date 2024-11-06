# 3D Unet Assisted Radiomics and Deep Features for Chondroid Tumor Classification

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Data Preparation](#data-preparation)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Exploration](#1-data-exploration)
  - [2. Tumor Segmentation](#2-tumor-segmentation)
  - [3. Feature Extraction](#3-feature-extraction)
  - [4. Model Training and Evaluation](#4-model-training-and-evaluation)
- [Results](#results)
- [References](#references)
- [Contact](#contact)

## Overview
This project leverages a hybrid approach combining **3D Unet**-assisted segmentation, **radiomics**, **deep learning features**, and **clinical data** for the classification of chondroid tumors (enchondroma and chondrosarcoma) from MRI images. The goal is to achieve higher diagnostic accuracy and reliability, assisting clinicians in differentiating between benign and malignant tumor types.

## Features
- **3D U-Net Segmentation**: Automated tumor segmentation using nnUNet for high-resolution MRI data.
- **Multi-modal Feature Extraction**:
  - **Radiomics**: Quantitative features capturing tumor shape, texture, and intensity.
  - **Deep Learning**: High-dimensional features from CNN-based models.
  - **Clinical Data**: Patient demographics and tumor-specific clinical information.
- **Machine Learning Classification**: Models like Random Forest, XGBoost, Gradient Boosting, LightGBM, and CatBoost to classify tumor types based on integrated features.
- **Evaluation Metrics**: Accuracy, Weighted Kappa, and AUC for comprehensive performance assessment.

## Data Preparation
1. **Download the MRI Dataset**: Place the raw MRI sequences in `data/raw/`.
2. **Data Preprocessing**: Run `src/data_preprocessing.py` to normalize, resize, and prepare data for segmentation.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/tuan-ld/3DUnet-Chondroid-Tumor-Classification.git
   cd 3DUnet-Chondroid-Tumor-Classification
2. Install required packages:
   ```bash
   pip install -r requirements.txt
3. Install [nnUnetv2](https://github.com/MIC-DKFZ/nnUNet/tree/master)

## Usage

### 1. Data Exploration

Open the `notebooks/01_data_exploration.ipynb` notebook to explore and visualize the dataset. This notebook includes basic statistics, visual checks, and distribution analysis for age, sex, and tumor types.

### 2. Tumor Segmentation

Train and apply the 3D Unet segmentation model using the `notebooks/02_segmentation_training.ipynb`. This notebook guides you through:

- Loading MRI data
- Configuring and training the 3D Unet model
- Saving segmented outputs in `data/nnUnetv2/nnUnet_results/`

### 3. Feature Extraction

Extract features from segmented MRI images using `notebooks/03_feature_extraction.ipynb`. This notebook performs:

- **Radiomics Feature Extraction**: Based on shape, intensity, and texture.
- **Deep Learning Feature Extraction**: Using a pre-trained CNN to capture complex features.
- **Clinical Data Integration**: Adding patient demographics and clinical characteristics.

### 4. Features combination

Combining features models using the extracted features in `notebooks/04_feature_combination.ipynb`. This notebook includes:

- Building five different feature combinations (radiomics-only, deep learning-only, etc.)

### 5. Model Training and Evaluation
Train and evaluate classification models using the extracted features in `notebooks/model_fitting`. This folder includes notebooks:
- Training classifiers (Random Forest, XGBoost, Gradient Boosting, LightGBM, CatBoost)
- Evaluating models with Accuracy, Weighted Kappa, and AUC metrics
- Saving results in `results/`

## Results

Results from the classification models, including accuracy, weighted kappa, and AUC, are saved in the `results/` folder. Visualizations, such as confusion matrices and ROC curves, can be found in `results/figures/`. The best model, which combines radiomics, deep ROI, and clinical features, achieved an accuracy of 0.90, a weighted kappa of 0.85, and an AUC of 0.91 with the CatBoost classifier.

## References

A list of references for the methodologies and tools used in this study is included in the `docs/references.md` file.

## Contact

For any questions or collaboration inquiries, please contact the corresponding authors:

- **Author One**: Le Dinh Tuan (ldt.uet@gmail.com)
- **Author Four**: Joon-Yong Jung (jy.jung@songeui.ac.kr)

Department of Radiology, Seoul St. Mary's Hospital, College of Medicine, The Catholic University of Korea, Seoul, Republic of Korea