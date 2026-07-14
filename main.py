from src.utils.config import load_config

config = load_config()

print("Experiment:", config.experiment.name)
print("Model:", config.model.name)
print("Image Size:", config.dataset.image_size)
print("Batch Size:", config.dataloader.batch_size)
print("Epochs:", config.training.epochs)