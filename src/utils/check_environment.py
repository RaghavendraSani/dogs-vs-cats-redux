import torch
import torchvision
import timm
import albumentations
import numpy as np
import pandas as pd
import cv2
import PIL

print("Dogs vs Cats environment check: ")
print("pytorch version: ", torch.__version__)
print("torchvision version: ", torchvision.__version__)
print("albumentations version: ", albumentations.__version__)
print("timming version: ", timm.__version__)
print("pillow version: ", PIL.__version__)
print("numpy version: ", np.__version__)
print("pandas version: ", pd.__version__)
print("open cv version: ", cv2.__version__)

print("\nCUDA available: ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU: ", torch.cuda.get_device_name(0))
    print("GPU count: ", torch.cuda.device_count())
    print("CUDA version: ", torch.version.cuda)

