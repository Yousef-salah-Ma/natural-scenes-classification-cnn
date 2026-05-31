"""
# 🌍 Natural Scenes Image Classification Dataset

## Context
This dataset contains images of natural scenes from around the world.

## Content
The dataset contains around 25,000 images of size 150x150 distributed across 6 classes:

{
    'buildings' -> 0,
    'forest'    -> 1,
    'glacier'   -> 2,
    'mountain'  -> 3,
    'sea'       -> 4,
    'street'    -> 5
}

## Splits
- Train: ~14,000 images  
- Test: ~3,000 images  
- Prediction: ~7,000 images  

## Source
Originally published by Intel via Analytics Vidhya:
https://datahack.analyticsvidhya.com

## Goal
Train a deep learning model capable of accurately classifying natural scenes into their categories.
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
from torchvision.transforms import v2

# ====== Classes (from folders) ======
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# ====== Model ======
# Deep CNN model

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel , self ).__init__()
        
        self.train_transform = v2.Compose([
            # Randomly flip image horizontally (left ↔ right)
            # Helps model become invariant to left-right orientation
            v2.RandomHorizontalFlip(p=0.5),

            # Randomly flip image vertically (top ↕ bottom)
            # Useful for some datasets, but can be harmful for natural scenes
            v2.RandomVerticalFlip(p=0.2),

            # Randomly rotate image by up to ±20 degrees
            # Improves robustness to object orientation changes
            v2.RandomRotation(degrees=20),

            # Randomly change image brightness, contrast, saturation, and hue
            # Helps model generalize under different lighting conditions
            v2.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        ])
        
        # Block 1: 32 feature maps
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
            
         
        )

        # Block 2: 64 feature maps
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        )

        # Block 3: 128 feature maps
        self.b3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )

        # Block 4: 256 feature maps
        self.b4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
      
            nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        )
         # Block 5: 256 feature maps
        self.b5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        )

        # Global pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 6)  # 6 classes
        )
        
        
    def forward(self, x):
        # Apply augmentation only in training mode
        if self.training:
            x = self.train_transform(x)

        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.global_pool(x)
        x = self.flatten(x)

        return self.fc(x)

# ====== Device ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load Model ======
model = MyModel()
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

# ====== Transform  ======
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# ====== UI ======
st.set_page_config(page_title="AI Classifier", page_icon="🔥")
st.title("🌍 Natural Scenes Classifier AI")

st.write("Upload a natural scene image and the model will classify it into one of 6 categories.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()

        st.success(f"🌟 Predicted Class: {classes[pred]}")

        st.info(f"""
        📊 Prediction Confidence: {confidence:.2f}

        🏷️ Classes:
        - buildings
        - forest
        - glacier
        - mountain
        - sea
        - street
        """)
