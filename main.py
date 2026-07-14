"""
from src.utils.config import load_config

config = load_config()

print("Experiment:", config.experiment.name)
print("Model:", config.model.name)
print("Image Size:", config.dataset.image_size)
print("Batch Size:", config.dataloader.batch_size)
print("Epochs:", config.training.epochs)
"""

"""
from pathlib import Path
from src.datasets.dataset import DogsVsCatsDataset

train_dir = Path("data/raw/train")
image_paths = list(train_dir.glob("*.jpg"))
dataset = DogsVsCatsDataset(image_paths)
print("Dataset Size:", len(dataset))
image, label = dataset[0]
print("Image Size:", image.size)
print("Label:", label)
"""

from pathlib import Path
from src.datasets.dataset import DogsVsCatsDataset
from src.datasets.transforms import train_transform

train_dir = Path("data/raw/train")
image_paths = list(train_dir.glob("*.jpg"))
dataset = DogsVsCatsDataset(
    image_paths=image_paths,
    transform=train_transform
)
image, label = dataset[0]
print("Image Shape:", image.shape)
print("Image Type:", image.dtype)
print("Label:", label)