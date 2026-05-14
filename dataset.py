import os
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from glob import glob

IMAGE_SIZE = 256

class ChangeDetectionDataset(Dataset):

    def __init__(self, root_dir):

        self.pre_paths = sorted(
            glob(os.path.join(root_dir, "pre-event", "*.tif"))
        )

        self.post_paths = sorted(
            glob(os.path.join(root_dir, "post-event", "*.tif"))
        )

        self.mask_paths = sorted(
            glob(os.path.join(root_dir, "target", "*.tif"))
        )
        def __len__(self):
            return len(self.pre_paths)

    def __getitem__(self, idx):

        pre_img = cv2.imread(self.pre_paths[idx], cv2.IMREAD_UNCHANGED)
        post_img = cv2.imread(self.post_paths[idx], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)

        pre_img = cv2.resize(pre_img, (IMAGE_SIZE, IMAGE_SIZE))
        post_img = cv2.resize(post_img, (IMAGE_SIZE, IMAGE_SIZE))
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))

        pre_img = pre_img.astype(np.float32) / 255.0
        post_img = post_img.astype(np.float32) / 255.0

        post_img = np.expand_dims(post_img, axis=-1)

        image = np.concatenate([pre_img, post_img], axis=-1)

        mask = mask.astype(np.float32)

        image = torch.tensor(image).permute(2,0,1)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask