import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from dataset import ChangeDetectionDataset
from model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = get_model().to(DEVICE)

model.load_state_dict(
    torch.load("best_model.pth")
)
model.eval()


val_dataset = ChangeDetectionDataset(
    "data/val"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)



def f1_score(preds, targets, threshold=0.5):

    preds = torch.sigmoid(preds)

    preds = (preds > threshold).float()

    tp = (preds * targets).sum()

    precision = tp / (preds.sum() + 1e-6)

    recall = tp / (targets.sum() + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return f1.item()


sample_image, sample_mask = val_dataset[0]

input_tensor = sample_image.unsqueeze(0).to(DEVICE)

with torch.no_grad():

    pred_mask = model(input_tensor)

    pred_mask = torch.sigmoid(pred_mask)

    pred_mask = (pred_mask > 0.5).float()

pred_mask = pred_mask.squeeze().cpu().numpy()

sample_image = sample_image.cpu().numpy()

sample_mask = sample_mask.squeeze().cpu().numpy()

pre_rgb = np.transpose(sample_image[:3], (1,2,0))


plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
plt.imshow(pre_rgb)
plt.title("Pre Event")

plt.subplot(1,3,2)
plt.imshow(sample_mask, cmap='gray')
plt.title("Ground Truth")

plt.subplot(1,3,3)
plt.imshow(pred_mask, cmap='gray')
plt.title("Prediction")

plt.show()