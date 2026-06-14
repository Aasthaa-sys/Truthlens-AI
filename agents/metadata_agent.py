from PIL import Image
from PIL.ExifTags import TAGS

def extract_metadata(image_path):

    metadata = {}

    try:

        image = Image.open(image_path)

        exif_data = image.getexif()

        for tag_id, value in exif_data.items():

            tag = TAGS.get(tag_id, tag_id)

            metadata[tag] = value

    except Exception as e:

        print("Metadata Error:", e)

    return metadata


if __name__ == "__main__":

    image_path = r"C:\Users\study\internproj\datasets\test\real\000000000872.jpg"

    metadata = extract_metadata(image_path)

    print("\n--- Metadata ---")

    if len(metadata) == 0:

        print("No metadata found")

    else:

        for key, value in metadata.items():

            print(key, ":", value)