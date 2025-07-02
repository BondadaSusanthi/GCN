import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class ODIRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, embeddings=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.embeddings = embeddings

        self.labels = self.data.columns[1:]  # Assuming 1st column is 'image_id'
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(self.data.iloc[idx, 1:].values.astype('float32'))

        return image, labels, self.embeddings
