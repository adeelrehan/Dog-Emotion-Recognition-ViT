# Dataset Description

## Dataset Name
Dog Emotion Dataset

## Source
Kaggle  
https://www.kaggle.com/datasets/danielshanbalico/dog-emotion

## Domain
Veterinary and Animal Welfare

## Dataset Overview
The Dog Emotion Dataset consists of RGB images of dogs labeled according to their emotional state. The dataset is designed to support supervised image classification tasks and is suitable for studying animal emotion recognition in veterinary applications.

The emotional states represented in the dataset act as indicators of stress, discomfort, and potential pain, which are important factors in veterinary diagnosis and animal welfare assessment.

## Classes
The dataset contains four emotion classes:
- Angry
- Happy
- Relaxed
- Sad

Each image belongs to exactly one of the above categories.

## Dataset Size
The dataset contains approximately 4,000 labeled images distributed across the four emotion classes.

## Data Organization
The dataset is organized in a directory-based structure, where each emotion class is stored in a separate folder. This structure enables efficient loading and preprocessing for deep learning models.

Example structure:
dataset/
├── Angry/
├── Happy/
├── Relaxed/
└── Sad/

## Data Split
For model training and evaluation, the dataset is divided into:
- Training set: 70%
- Validation set: 15%
- Test set: 15%

This split ensures unbiased evaluation and helps prevent data leakage.

## Data Preprocessing
The following preprocessing steps are applied before model training:
- Image resizing to a fixed resolution
- Normalization of pixel values
- Data augmentation techniques such as rotation and horizontal flipping to improve generalization

## Challenges
Key challenges associated with the dataset include:
- Variations in lighting and background conditions
- Subtle visual differences between certain emotional states
- Potential class imbalance

These challenges are addressed through preprocessing, data augmentation, and careful model design.
