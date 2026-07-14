from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.datasets.dataset import DogsVsCatsDataset
from src.datasets.transforms import valid_transform
from src.models.model import DogsVsCatsModel

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DogsVsCatsModel()
    model.load_state_dict(torch.load("checkpoints/best_model.pth",map_location=device,weights_only=True))
    model.to(device)
    model.eval()

    # Test dataset
    test_dir = Path("data/raw/test")

    image_paths = sorted(test_dir.glob("*.jpg"),key=lambda x: int(x.stem))

    test_dataset = DogsVsCatsDataset(
        image_paths=image_paths,
        transform=valid_transform,
        train=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    predictions = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1)
            probs = probs.cpu().numpy()
            for image_id, prob in zip(image_ids, probs):
                predictions.append(
                    (
                        int(image_id),
                        float(prob),
                    )
                )

    predictions.sort(key=lambda x: x[0])
    submission = pd.DataFrame(
        predictions,
        columns=["id", "label"],
    )

    output_dir = Path("data/submissions")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission.csv"
    submission.to_csv(
        output_path,
        index=False,
    )
    print(f"\nSubmission saved to: {output_path}")
    print(submission.head())