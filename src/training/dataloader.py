from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.datasets.dataset import DogsVsCatsDataset
from src.datasets.transforms import train_transform, valid_transform


def create_dataloaders(batch_size=32):

    train_dir = Path("data/raw/train")

    image_paths = list(train_dir.glob("*.jpg"))

    labels = [
        1 if path.name.startswith("dog") else 0
        for path in image_paths
    ]

    train_paths, valid_paths = train_test_split(
        image_paths,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    train_dataset = DogsVsCatsDataset(
        train_paths,
        transform=train_transform,
    )

    valid_dataset = DogsVsCatsDataset(
        valid_paths,
        transform=valid_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, valid_loader