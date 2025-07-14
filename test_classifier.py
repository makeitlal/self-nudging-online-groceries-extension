import requests
from tempfile import NamedTemporaryFile
from EcommerceClassifier.model import load_model, predict

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

model = load_model()

image_url = "https://target.scene7.com/is/image/Target/GUEST_ce4ac41d-c124-49db-8f0f-2f472ee51815"
title = "Strawberries - 2lb"

# Download the image and get local path
image_path = download_image(image_url)

# Predict
result = predict(model, image_path, title)

# Print result
print("Prediction result:", result)
# for label, score in result.items():
#     print(f"{label}: {score:.4f}")

# Optional: delete the temp image file after prediction
import os
os.unlink(image_path)
