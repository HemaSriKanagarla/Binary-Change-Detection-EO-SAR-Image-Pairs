import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from dataset import ChangeDetectionDataset
from model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


train_dataset = ChangeDetectionDataset(
    "data/train"
)

val_dataset = ChangeDetectionDataset(
    "data/val"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)


model = get_model().to(DEVICE)


bce_loss = nn.BCEWithLogitsLoss()

dice_loss = smp.losses.DiceLoss(mode='binary')


def combined_loss(pred, target):

    bce = bce_loss(pred, target)

    dice = dice_loss(pred, target)

    return bce + dice
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4
)



def iou_score(preds, targets, threshold=0.5):

    preds = torch.sigmoid(preds)

    preds = (preds > threshold).float()

    intersection = (preds * targets).sum()

    union = preds.sum() + targets.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.item()


EPOCHS = 10

best_iou = 0
for epoch in range(EPOCHS):

    print(f"Epoch {epoch+1}/{EPOCHS}")

    model.train()

    total_train_loss = 0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = combined_loss(outputs, masks)

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    model.eval()

    total_iou = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            total_iou += iou_score(outputs, masks)

    avg_iou = total_iou / len(val_loader)

    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Validation IoU: {avg_iou:.4f}")

    if avg_iou > best_iou:

        best_iou = avg_iou

        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")