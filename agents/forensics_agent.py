import os
from PIL import Image, ImageChops, ImageEnhance


def perform_ela(image_path):
    """
    Perform Error Level Analysis (ELA) on an image.

    Args:
        image_path (str): Path to input image.

    Returns:
        str: Path to generated ELA image.
    """

    # Create output directory if it doesn't exist
    output_dir = os.path.join("outputs", "forensics")
    os.makedirs(output_dir, exist_ok=True)

    # Output file paths
    temp_path = os.path.join(output_dir, "temp_ela.jpg")
    ela_path = os.path.join(output_dir, "ela_result.jpg")

    # Open original image
    image = Image.open(image_path).convert("RGB")

    # Save temporary compressed image
    image.save(temp_path, "JPEG", quality=90)

    # Reopen compressed image
    resaved = Image.open(temp_path)

    # Compute difference
    diff = ImageChops.difference(image, resaved)

    # Find maximum difference
    extrema = diff.getextrema()
    max_diff = max(channel[1] for channel in extrema)

    if max_diff == 0:
        max_diff = 1

    # Enhance brightness
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(diff).enhance(scale)

    # Save ELA image
    ela_image.save(ela_path)

    return ela_path


if __name__ == "__main__":
    image_path = r"C:\Users\study\internproj\datasets\test\real\000000045550.jpg"

    result = perform_ela(image_path)

    print("ELA image saved at:", result)