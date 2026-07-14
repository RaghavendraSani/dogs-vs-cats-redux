from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class DogsVsCatsDataset(Dataset):

    def __init__(self, image_paths, transform=None, train=True):
        self.image_paths = image_paths
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        #image = Image.open(image_path).convert("RGB")
        image = np.array(Image.open(image_path).convert("RGB"))

        #if self.transform:
            #image = self.transform(image)
        if self.transform:
            image = self.transform(image=image)["image"]

        if not self.train:
            image_id = Path(image_path).stem
            return image, image_id

        label = 1.0 if Path(image_path).name.startswith("dog") else 0.0
        label = torch.tensor(label, dtype=torch.float32)
        return image, label