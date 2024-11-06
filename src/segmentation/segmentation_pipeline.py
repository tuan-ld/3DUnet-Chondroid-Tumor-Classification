import subprocess
import os

class NnUNetPipeline:
    def __init__(self, dataset_dir, output_dir, task_name):
        """
        Initialize nnU-Net pipeline settings.
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.task_name = task_name

    def prepare_data(self):
        """
        Prepares the dataset for nnU-Net training by using nnU-Net's preprocessing functions.
        """
        print("Preparing dataset for nnU-Net...")
        subprocess.run([
            "nnUNetv2_convert_dataset",
            "--dataset_dir", self.dataset_dir,
            "--task_name", self.task_name,
            "--output_dir", os.path.join(self.dataset_dir, "nnUNet_preprocessed")
        ])

    def train_model(self, fold=0):
        """
        Trains nnU-Net on the specified dataset and fold.
        """
        print("Training nnU-Net model...")
        subprocess.run([
            "nnUNetv2_train",
            self.task_name,
            "3d_fullres",
            str(fold),
            "--data_dir", os.path.join(self.dataset_dir, "nnUNet_preprocessed"),
            "--output_dir", self.output_dir
        ])

    def predict(self, input_dir, output_dir):
        """
        Runs inference using the trained nnU-Net model.
        """
        print("Running inference with nnU-Net model...")
        subprocess.run([
            "nnUNetv2_predict",
            "--task_name", self.task_name,
            "--fold", "ensemble",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--data_dir", os.path.join(self.dataset_dir, "nnUNet_preprocessed"),
            "--output_format", "nifti"
        ])
