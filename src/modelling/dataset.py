from typing import Tuple
import pandas as pd
import os
from torch.utils.data import Dataset
from pydantic import BaseModel

class CustomDataDataset(BaseModel, Dataset):
    annotation_file: str
    labels: pd.DataFrame = pd.read_csv(annotation_file)
    transform: callable = None
    target_transform: callable = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[]:
        label = self.labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label