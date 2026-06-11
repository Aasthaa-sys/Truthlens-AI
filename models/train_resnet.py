import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.RandomCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASETS
# =========================

train_dataset = ImageFolder(
    "datasets/train",
    transform=train_transform
)

val_dataset = ImageFolder(
    "datasets/val",
    transform=val_transform
)

test_dataset = ImageFolder(
    "datasets/test",
    transform=val_transform
)

# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Train classes:", train_dataset.classes)
print("Train size:", len(train_dataset))

print("Val classes:", val_dataset.classes)
print("Val size:", len(val_dataset))

print("Test classes:", test_dataset.classes)
print("Test size:", len(test_dataset))

# =========================
# MODEL
# =========================

model = resnet50(
    weights=ResNet50_Weights.DEFAULT
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    3
)

print(model.fc)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

print("Loss and Optimizer Ready")

# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

model = model.to(device)

# =========================
# TRAINING
# =========================

best_val_acc = 0

for epoch in range(20):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    # =====================
    # VALIDATION
    # =====================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1}/20 | "
        f"Loss: {running_loss/len(train_loader):.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # =====================
    # SAVE BEST MODEL
    # =====================

    if val_accuracy > best_val_acc:

        best_val_acc = val_accuracy

        torch.save(
            model.state_dict(),
            "models/truthlens_resnet.pth"
        )

        print(
            f"Best Model Saved "
            f"(Val Acc = {val_accuracy:.2f}%)"
        )

# =========================
# LOAD BEST MODEL
# =========================

model.load_state_dict(
    torch.load(
        "models/truthlens_resnet.pth",
        map_location=device
    )
)

# =========================
# TEST ACCURACY
# =========================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

test_accuracy = 100 * correct / total

print("\n=========================")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("=========================")
print("Training Completed")