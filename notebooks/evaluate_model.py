import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import numpy as np

classes = [
    "deepfake",
    "manipulated",
    "real"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

test_dataset = ImageFolder(
    "datasets/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

model = resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)

model.load_state_dict(
    torch.load(
        "models/truthlens_resnet.pth",
        map_location="cpu"
    )
)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, preds = torch.max(outputs,1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

print(
    confusion_matrix(
        all_labels,
        all_preds
    )
)

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=classes
    )
)