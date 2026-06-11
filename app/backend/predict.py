import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

classes = [
    "deepfake",
    "manipulated",
    "real"
]

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

print("Model Loaded Successfully")
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
image_path = r"C:\Users\study\internproj\datasets\test\manipulated\Tp_D_NRN_M_N_cha00027_cha00028_11785.jpg"
print("Testing:", image_path)
image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)
with torch.no_grad():

    output = model(image)

    print("Raw Output:")
    print(output)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    print("\n--- Prediction Results ---")

    for i, cls in enumerate(classes):
        print(
            f"{cls}: {probabilities[0][i].item()*100:.2f}%"
        )

    confidence, predicted = torch.max(
        probabilities,
        1
    )

    print(
        "\nFinal Prediction:",
        classes[predicted.item()]
    )

    print(
        "Confidence:",
        f"{confidence.item()*100:.2f}%"
    )