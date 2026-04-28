import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from utils.preprocessing import preprocess_image

class DRDataset(Dataset):
    """
    Custom dataset for Diabetic Retinopathy
    CSV format:
    image_name, label
    """

    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 1]

        image = preprocess_image(img_path)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label)
