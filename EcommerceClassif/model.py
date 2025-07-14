# model.py
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import json
from PIL import Image
from torchvision import transforms, models
import os
import gdown


current_dir = os.path.dirname(__file__)
json_path = os.path.join(current_dir, "label_to_npm.json")
checkpoint_path = os.path.join(current_dir, "model_checkpoint.pth")

# --- Label map ---
with open(json_path, "r") as f:
    label_to_class = json.load(f)

# --- Model Definition ---
class FineGrainedClassifier(nn.Module):
    def __init__(self, num_classes=434):
        super(FineGrainedClassifier, self).__init__()
        self.image_encoder = models.resnet50(pretrained=True)
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
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_encoder(image)
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]
        combined = torch.cat((image_features, text_features), dim=1)
        return self.classifier(combined)

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)

# --- Load model and tokenizer ---
def load_model(checkpoint_path=checkpoint_path):
    model = FineGrainedClassifier(num_classes=len(label_to_class))

    if not os.path.exists(checkpoint_path):
        gdown.download("https://drive.google.com/file/d/1SR95PUMKe1xCaVWI5VI7hvtSq5zCygVx", checkpoint_path, quiet=False)
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        if new_k in model.state_dict():
            cleaned_state_dict[new_k] = v
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model

tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en")

# def predict(model, image_path, title, threshold=0.4):
#     image = load_image(image_path)
#     encoding = tokenizer(title, padding='max_length', max_length=200, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         output = model(image, input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
#         probs = torch.nn.functional.softmax(output, dim=1)
#         top3_prob, top3_idx = torch.topk(probs, 3, dim=1)

#     labels = [label_to_class[str(i.item())] for i in top3_idx[0]]
#     results = {labels[i]: top3_prob[0][i].item() for i in range(len(labels))}

#     if top3_prob[0][0] < threshold:
#         results = {"Others": 1.0 - top3_prob[0][0].item(), **results}

#     return results

# def predict(model, image_path, title):
#     image = load_image(image_path)
#     encoding = tokenizer(title, padding='max_length', max_length=200, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         output = model(image, input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
#         probs = torch.nn.functional.softmax(output, dim=1)
#         top_prob, top_idx = torch.max(probs, dim=1)

#     top_label = label_to_class[str(top_idx.item())]
#     top_score = top_prob.item()
#     return top_label, top_score

def predict(model, image_path, title):
    image = load_image(image_path)
    encoding = tokenizer(title, padding='max_length', max_length=200, truncation=True, return_tensors='pt')

    with torch.no_grad():
        output = model(image, input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
        probs = torch.nn.functional.softmax(output, dim=1).squeeze()

    # Aggregate probs per NPM category
    npm_scores = {}
    for i, prob in enumerate(probs):
        npm_category = label_to_class[str(i)]
        npm_scores[npm_category] = npm_scores.get(npm_category, 0) + prob.item()

    # Sort categories by total probability
    sorted_npm = sorted(npm_scores.items(), key=lambda x: x[1], reverse=True)

    top_npm_category, top_score = sorted_npm[0]
    return top_npm_category, top_score