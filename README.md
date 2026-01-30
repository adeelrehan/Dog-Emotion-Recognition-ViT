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


## 🧠 Model Architecture

This project uses a **Vision Transformer (ViT)** model implemented completely from scratch.  
The model works by dividing each input image into fixed-size patches, which are then flattened and embedded into token representations.

These tokens are passed through Transformer encoder layers consisting of multi-head self-attention and feed-forward networks.  
The self-attention mechanism enables the model to capture global contextual relationships across different regions of the image, which is essential for recognizing subtle emotional expressions in dogs.

The final representation is passed through a fully connected classification head to predict one of four emotion classes: Angry, Happy, Relaxed, or Sad.

⚠️ No pretrained models or transfer learning techniques were used.


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

## 📊 Results and Evaluation

### Validation Performance
- **Accuracy:** 75.67%
- **Precision (weighted):** 0.7621
- **Recall (weighted):** 0.7567
- **F1-score (weighted):** 0.7587
- 
- ![Confusion Matrix](results/confusion_matrix.png)
- 
![ROC Curve](results/roc_curve.png)


## 📓 Notebook

The complete implementation, training, and evaluation of the Vision Transformer model is available in the following notebook:

- `notebooks/Dog_Emotion_Recognition_ML2026_final.ipynb`



