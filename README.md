# Binary-Change-Detection-EO-SAR-Image-Pairs


## Project Overview

This project performs binary change detection on EO-SAR image pairs using a U-Net based deep learning architecture.

The model predicts pixel-level change masks between pre-event and post-event satellite imagery.

---

# Dataset Structure

data/

- ├── train/
- https://huggingface.co/datasets/doron333/change-detection-dataset/blob/main/train.zip
- ├── val/  
- https://huggingface.co/datasets/doron333/change-detection-dataset/blob/main/val.zip
- └── test/
- https://huggingface.co/datasets/doron333/change-detection-dataset/blob/main/test.zip

Each split contains:
- pre-event images
- post-event images
- target masks

---

# Installation

pip install -r requirements.txt

---

# Training

python train.py

---

# Evaluation

python eval.py

---

# Model Architecture

- U-Net
- ResNet34 encoder
- 4-channel input:
  - RGB EO image
  - SAR image

---

# Loss Function

Combined:
- BCEWithLogitsLoss
- Dice Loss

---

# Metrics

- IoU
- F1 Score
- Precision
- Recall

---

# Results

| Split | IoU | F1 |
|---|---|---|

(Add your values)

---

# Future Improvements

- Attention-based fusion
- Transformer architectures
- Multi-scale learning
- Better SAR denoising

---

# References

- U-Net
- Change Detection papers
- segmentation-models-pytorch
