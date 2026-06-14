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


def predict_image(image_path):

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

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    prediction = classes[
        predicted.item()
    ]

    confidence = round(
        confidence.item()*100,
        2
    )

    return prediction, confidence


if __name__ == "__main__":

    image_path = r"C:\Users\study\internproj\datasets\test\deepfake\syn_0.5_12_00e0200bf1f248c0ab26e105314fbefa.jpg"

    prediction, confidence = predict_image(
        image_path
    )

    print(
        "Prediction:",
        prediction
    )

    print(
        "Confidence:",
        confidence,
        "%"
    )