from PIL import Image, ImageChops, ImageEnhance

def perform_ela(image_path):

    temp_path = "outputs/forensics/temp_ela.jpg"

    ela_path = "outputs/forensics/ela_result2.jpg"

    image = Image.open(image_path).convert("RGB")

    image.save(
        temp_path,
        "JPEG",
        quality=90
    )

    resaved = Image.open(temp_path)

    diff = ImageChops.difference(
        image,
        resaved
    )

    extrema = diff.getextrema()

    max_diff = max(
        ex[1]
        for ex in extrema
    )

    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff

    diff = ImageEnhance.Brightness(
        diff
    ).enhance(scale)

    diff.save(ela_path)

    return ela_path


if __name__ == "__main__":

    image_path = r"C:\Users\study\internproj\datasets\test\real\000000045550.jpg"

    result = perform_ela(
        image_path
    )

    print(
        "ELA Saved:",
        result
    )