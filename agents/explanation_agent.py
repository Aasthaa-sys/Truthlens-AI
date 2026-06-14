import torch
import torch.nn as nn
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_heatmap(image_path):

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

    input_tensor = transform(
        image
    ).unsqueeze(0)

    target_layers = [
        model.layer4[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    rgb_img = cv2.imread(
        image_path
    )

    rgb_img = cv2.cvtColor(
        rgb_img,
        cv2.COLOR_BGR2RGB
    )

    rgb_img = cv2.resize(
        rgb_img,
        (224,224)
    )

    rgb_img = np.float32(
        rgb_img
    ) / 255

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    save_path = (
        "outputs/heatmaps/gradcam_result.jpg"
    )

    cv2.imwrite(
        save_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    return save_path


if __name__ == "__main__":

    image_path = r"C:\Users\study\internproj\datasets\test\real\000000058350.jpg"

    result = generate_heatmap(
        image_path
    )

    print(
        "Heatmap Saved:",
        result
    )