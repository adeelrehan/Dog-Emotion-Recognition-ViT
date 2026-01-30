# Automated Dog Emotion Recognition for Veterinary Applications Using Vision Transformers

## Domain
Veterinary

## Introduction
Animals cannot verbally communicate their emotional states, making veterinary diagnosis and handling challenging. 
This project aims to develop an automated system that recognizes dog emotional states using image-based deep learning techniques.

## Problem Statement
Veterinarians often rely on subjective interpretation of canine behavior, which may lead to misclassification of emotions such as fear or aggression.
This project proposes a Vision Transformer–based approach to objectively classify dog emotions from images.

## Dataset
Dog Emotion Dataset obtained from Kaggle.
The dataset contains approximately 4,000 images of dogs categorized into four emotion classes:
Angry, Happy, Relaxed, and Sad.
- **Source:** Kaggle – Dog Emotion Dataset  
- **Link:** https://www.kaggle.com/datasets/danielshanbalico/dog-emotion  
- **Domain:** Veterinary  
- **Emotion Classes:** Angry, Happy, Relaxed, Sad  
- **Image Type:** RGB dog facial images  

### Preprocessing Steps
- Images resized to 224×224 pixels  
- Pixel normalization applied  
- Strong data augmentation used to handle small dataset size  
- Stratified train–validation split (85% training, 15% validation)


## Model
Vision Transformer (ViT-B/16) using transfer learning.
The model processes image patches using self-attention to capture global visual context.

## Evaluation Metrics
Accuracy  
Precision  
Recall  
F1-score  
Confusion Matrix  

## Sustainable Development Goals (SDGs)
SDG 3 – Good Health and Well-being  
SDG 15 – Life on Land  

## Tools & Technologies
Python  
PyTorch  
Vision Transformer  
TIMM  
Jupyter Notebook  
Google Colab  

