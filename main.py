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
"""

"""
import torch
from src.models.model import DogsVsCatsModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DogsVsCatsModel().to(device)
dummy = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    output = model(dummy)
print("Device:", device)
print("Output Shape:", output.shape)
"""

"""
from src.training.dataloader import create_dataloaders

def main():
    train_loader, valid_loader = create_dataloaders(batch_size=32)
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(valid_loader))
    images, labels = next(iter(train_loader))
    print("Image Batch Shape:", images.shape)
    print("Label Batch Shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)
    
if __name__ == "__main__":
    main()
"""

from src.models.model import DogsVsCatsModel
from src.training.dataloader import create_dataloaders
from src.training.trainer import Trainer

def main():
    train_loader, valid_loader = create_dataloaders(batch_size=32)
    model = DogsVsCatsModel()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=30,
        learning_rate=1e-4,
    )
    trainer.train()

if __name__ == "__main__":
    main()