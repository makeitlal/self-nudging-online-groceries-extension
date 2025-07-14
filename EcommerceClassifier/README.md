---
license: apache-2.0
library_name: PyTorch
tags:
- text-classification
- emcommerce
base_model:
- jinaai/jina-embeddings-v2-base-en
- microsoft/resnet-50
pipeline_tag: image-text-to-text
---
<br><br>

<p align="center">
<img src="https://huggingface.co/Maverick98/EcommerceClassifier/resolve/main/1.png" alt="EcommerceClassifier is a multi-modal deep learning model developed to enhance product categorization in e-commerce settings" width="150px">
</p>

<p align="center">
<b>Ecommerce Classifier trained by <b>Maverick AI</b>.</b>
</p>

# EcommerceClassifier

**EcommerceClassifier** is a fine-grained product classifier specifically designed for e-commerce platforms. This model leverages both product images and titles to classify items into one of 434 categories across two primary e-commerce domains: Grocery & Gourmet and Health & Household. All the training classes can be seen the label_to_class.json file

## Model Details

### Model Description

EcommerceClassifier is a multi-modal deep learning model developed to enhance product categorization in e-commerce settings. It integrates image and text data to provide accurate classifications, ensuring that products are correctly placed in their respective categories. This model is particularly useful in automating the product categorization process, optimizing search results, and improving recommendation systems.

- **Developed by:** [Mohit Dhawan]
- **Model type:** Multi-modal classification model
- **Language(s) (NLP):** English (product titles)
- **License:** Apache 2.0
- **Finetuned from model:** ResNet50 for image encoding, Jina's embeddings for text encoding

### Model Sources

- **Repository:** https://huggingface.co/Maverick98/EcommerceClassifier/
- **Demo:** https://huggingface.co/spaces/Maverick98/ECommerceClassify

## Uses

### Direct Use

EcommerceClassifier is intended for direct use in e-commerce platforms to automate and improve the accuracy of product classification. It can be integrated into existing systems to classify new products, enhance search functionality, and improve the relevancy of recommendations.

### Downstream Use

EcommerceClassifier can be fine-tuned for specific e-commerce categories or extended to include additional product domains. It can also be integrated into larger e-commerce systems for fraud detection, where misclassified or counterfeit products are flagged.

### Out-of-Scope Use

EcommerceClassifier is not intended for use outside of e-commerce product classification, particularly in contexts where the input data is significantly different from the domains it was trained on. Misuse includes attempts to classify non-e-commerce-related images or texts.

## Bias, Risks, and Limitations

While EcommerceClassifier is trained on a diverse dataset, it may still exhibit biases inherent in the training data, particularly if certain categories are underrepresented. There is also a risk of overfitting to specific visual or textual features, which may reduce its effectiveness on new, unseen data.

### Recommendations

Users should be aware of the potential biases in the model and consider re-training or fine-tuning EcommerceClassifier with more diverse or updated data as needed. Regular evaluation of the model's performance on new data is recommended to ensure it continues to perform accurately.

## How to Get Started with the Model

Use the code below to get started with EcommerceClassifier:

```python
import torch
from transformers import AutoTokenizer, AutoModel
import json
import requests
from PIL import Image
from torchvision import transforms
import urllib.request
import torch.nn as nn

# --- Define the Model ---
class FineGrainedClassifier(nn.Module):
    def __init__(self, num_classes=434):  # Updated to 434 classes
        super(FineGrainedClassifier, self).__init__()
        self.image_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.image_encoder.fc = nn.Identity()
        self.text_encoder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en')
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # Updated to 434 classes
        )
    
    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_encoder(image)
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]
        combined_features = torch.cat((image_features, text_features), dim=1)
        output = self.classifier(combined_features)
        return output

# Load the label-to-class mapping from Hugging Face
label_map_url = "https://huggingface.co/Maverick98/EcommerceClassifier/resolve/main/label_to_class.json"
label_to_class = requests.get(label_map_url).json()

# Load the custom model
model = FineGrainedClassifier(num_classes=len(label_to_class))
checkpoint_url = f"https://huggingface.co/Maverick98/EcommerceClassifier/resolve/main/model_checkpoint.pth"
checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location=torch.device('cpu'))

# Clean up the state dictionary
state_dict = checkpoint.get('model_state_dict', checkpoint)
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_key = k[7:]  # Remove "module." prefix
    else:
        new_key = k

    # Check if the new_key exists in the model's state_dict, only add if it does
    if new_key in model.state_dict():
        new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)

# Load the tokenizer from Jina
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en")

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_path_or_url):
    if image_path_or_url.startswith("http"):
        with urllib.request.urlopen(image_path_or_url) as url:
            image = Image.open(url).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')
    
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(image_path_or_url, title, threshold=0.7):
    # Preprocess the image
    image = load_image(image_path_or_url)
    
    # Tokenize title
    title_encoding = tokenizer(title, padding='max_length', max_length=200, truncation=True, return_tensors='pt')
    input_ids = title_encoding['input_ids']
    attention_mask = title_encoding['attention_mask']

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image, input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top3_probabilities, top3_indices = torch.topk(probabilities, 3, dim=1)

    # Map the top 3 indices to class names
    top3_classes = [label_to_class[str(idx.item())] for idx in top3_indices[0]]

    # Check if the highest probability is below the threshold
    if top3_probabilities[0][0].item() < threshold:
        top3_classes.insert(0, "Others")
        top3_probabilities = torch.cat((torch.tensor([[1.0 - top3_probabilities[0][0].item()]]), top3_probabilities), dim=1)

    # Output the class names and their probabilities
    results = {}
    for i in range(len(top3_classes)):
        results[top3_classes[i]] = top3_probabilities[0][i].item()
    
    return results

# Example usage
image_url = "https://example.com/path_to_your_image.jpg"  # Replace with actual image URL or local path
title = "Organic Green Tea"
results = predict(image_url, title)

print("Prediction Results:")
for class_name, prob in results.items():
    print(f"Class: {class_name}, Probability: {prob}")

```

# Training Details

## Training Data

EcommerceClassifier was trained on a dataset scraped from Amazon, focusing on two primary product nodes:

- **Grocery & Gourmet**
- **Health & Household**

The dataset includes over 434 categories with product images and titles, providing a comprehensive basis for training the model.

## Training Procedure

### Preprocessing:

- Images were resized to 224x224 pixels.
- Titles were tokenized using Jina’s embedding model.
- Data augmentation techniques such as random horizontal flip, random rotation, and color jitter were applied to images during training.

### Training Hyperparameters:

- **Training regime:** Mixed precision (fp16)
- **Optimizer:** AdamW
- **Learning Rate:** 1e-4
- **Epochs:** 20
- **Batch Size:** 8
- **Accumulation Steps:** 4

### Speeds, Sizes, Times:

The model was trained over 20 epochs using an NVIDIA A10 GPU, with each epoch taking approximately 30 minutes.

# Evaluation

## Testing Data, Factors & Metrics

### Testing Data

The model was evaluated on a validation dataset held out from the training data. The testing data includes a balanced representation of all 434 categories.

### Factors

Evaluation factors include subpopulations within the Grocery & Gourmet and Health & Household categories.

### Metrics

The model was evaluated using the following metrics:

- **Accuracy:** The overall correctness of the model's predictions.
- **Precision and Recall:** Evaluated per class to ensure balanced performance across all categories.

## Results

The model achieved an overall accuracy of 83%, with a balanced precision and recall across most categories. Precision and recall tend to be low in the aggregated classes such as assortments, gift pack etc. The "others" category effectively captured instances where the model's confidence in the top predictions was low.

## Summary

EcommerceClassifier demonstrated strong performance across the majority of categories, with particular strengths in well-represented classes. Future work may focus on enhancing performance in categories with fewer training examples.

# Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in Lacoste et al. (2019).

- **Hardware Type:** NVIDIA A10 GPUs
- **Hours used:** ~10 hours total training time

# Technical Specifications

## Model Architecture and Objective

The model consists of a ResNet50-based image encoder and a Jina embeddings-based text encoder, combined through fully connected layers to classify into 434 categories.

## Compute Infrastructure

- **Hardware:** NVIDIA A10 GPUs
- **Software:** The model was implemented using PyTorch and Hugging Face Transformers libraries.


# Model Card Authors

Mohit Dhawan

# Model Card Contact

For inquiries, please contact [mohit.dhawan2510@gmail.com]