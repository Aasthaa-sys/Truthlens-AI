import os
import random
import shutil

random.seed(42)

DATASET_DIR = "datasets"

CLASSES = [
    "real",
    "manipulated",
    "deepfake"
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

for cls in CLASSES:

    source_dir = os.path.join(DATASET_DIR, cls)

    files = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(
    (".jpg", ".jpeg", ".png", ".tif", ".tiff")
)
    ]

    random.shuffle(files)

    total = len(files)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    for split_name, split_files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files)
    ]:

        target_dir = os.path.join(
            DATASET_DIR,
            split_name,
            cls
        )

        os.makedirs(target_dir, exist_ok=True)

        for file in split_files:

            src = os.path.join(source_dir, file)
            dst = os.path.join(target_dir, file)

            shutil.copy2(src, dst)

    print(f"\n{cls}")
    print(f"Total: {total}")
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")