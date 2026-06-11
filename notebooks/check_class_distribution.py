from torchvision.datasets import ImageFolder

dataset = ImageFolder("datasets/test")

print(dataset.classes)

counts = {}

for _, label in dataset.samples:
    class_name = dataset.classes[label]

    if class_name not in counts:
        counts[class_name] = 0

    counts[class_name] += 1

print(counts)