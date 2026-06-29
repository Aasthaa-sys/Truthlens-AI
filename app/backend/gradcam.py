import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================
# CREATE OUTPUT DIRECTORY
# ==========================

OUTPUT_DIR = os.path.join("outputs", "gradcam")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# LOAD MODEL
# ==========================

model = resnet50(weights=None)

# Change to 2 if your model was trained on Real vs Deepfake
model.fc = nn.Linear(
    model.fc.in_features,
    3
)

model.load_state_dict(
    torch.load(
        "models/truthlens_resnet.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# ==========================
# GENERATE GRAD-CAM
# ==========================

def generate_gradcam(image_path):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    target_layers = [model.layer4[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    rgb_img = cv2.imread(image_path)

    if rgb_img is None:
        raise FileNotFoundError(
            f"Unable to read image: {image_path}"
        )

    rgb_img = cv2.cvtColor(
        rgb_img,
        cv2.COLOR_BGR2RGB
    )

    rgb_img = cv2.resize(
        rgb_img,
        (224, 224)
    )

    rgb_img = np.float32(rgb_img) / 255.0

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        "gradcam_result.jpg"
    )

    success = cv2.imwrite(
        save_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    if not success:
        raise RuntimeError(
            f"Failed to save Grad-CAM image: {save_path}"
        )

    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"Grad-CAM image was not created: {save_path}"
        )

    return save_path


if __name__ == "__main__":

    test_image = "sample.jpg"  # Replace with your image path

    output = generate_gradcam(test_image)

    print("Grad-CAM saved at:", output)