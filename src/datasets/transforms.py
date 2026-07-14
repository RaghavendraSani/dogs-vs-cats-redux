import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


train_transform = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
    ToTensorV2(),
])


valid_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
    ToTensorV2(),
])