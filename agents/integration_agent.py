from deepfake_agent import predict_image
from explanation_agent import generate_heatmap
from metadata_agent import extract_metadata
from risk_agent import assess_risk
from forensics_agent import perform_ela

image_path = r"C:\Users\study\internproj\datasets\test\real\000000058350.jpg"

print("\n=================================")
print("TruthLens AI Integration Agent")
print("=================================\n")

# Detection Agent
prediction, confidence = predict_image(
    image_path
)

# Explanation Agent
heatmap_path = generate_heatmap(
    image_path
)

# Forensics Agent
ela_path = perform_ela(
    image_path
)

# Metadata Agent
metadata = extract_metadata(
    image_path
)

metadata_found = len(metadata) > 0

# Risk Agent
risk_level, reasons = assess_risk(
    prediction,
    confidence,
    metadata_found
)

# Final Report
print("Prediction:", prediction)

print("Confidence:", confidence, "%")

print("\nHeatmap:")
print(heatmap_path)

print("\nELA Result:")
print(ela_path)

print("\nMetadata:")

if metadata_found:

    for key, value in metadata.items():

        print(key, ":", value)

else:

    print("No metadata found")

print("\nRisk Level:", risk_level)

print("\nReasons:")

for reason in reasons:

    print("-", reason)