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
# PROJECT ROOT
# ==========================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

# ==========================
# PATHS
# ==========================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "truthlens_resnet.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "gradcam"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# LOAD MODEL
# ==========================

model = resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    3
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=torch.device("cpu")
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

clean_checkpoint = {}

for key, value in checkpoint.items():
    clean_key = key.replace("module.", "")
    clean_checkpoint[clean_key] = value

model.load_state_dict(clean_checkpoint)
model.eval()

# ==========================
# GENERATE GRAD-CAM
# ==========================

def generate_gradcam(image_path):
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

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

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor
        )[0]

    rgb_img = cv2.imread(image_path)

    if rgb_img is None:
        raise FileNotFoundError(
            f"Unable to read image with OpenCV: {image_path}"
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

    saved = cv2.imwrite(
        save_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    if not saved:
        raise RuntimeError(
            f"OpenCV failed to save Grad-CAM image: {save_path}"
        )

    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"Grad-CAM image was not created: {save_path}"
        )

    return save_path


if __name__ == "__main__":
    test_image = os.path.join(
        BASE_DIR,
        "sample.jpg"
    )

    output = generate_gradcam(test_image)

    print("Grad-CAM saved at:", output)