from PIL import Image
import os

folders = [
    "datasets/real",
    "datasets/manipulated",
    "datasets/deepfake"
]

for folder in folders:

    bad = 0

    for file in os.listdir(folder):

        path = os.path.join(folder,file)

        try:
            img = Image.open(path)
            img.verify()

        except:
            bad += 1

    print(folder, "Bad Images:", bad)