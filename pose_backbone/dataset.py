import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class JointTransform:
    def __init__(self, crop_size=224, is_train=True, num_keypoints=42):
        self.crop_size = crop_size
        self.is_train = is_train
        self.num_keypoints = num_keypoints
        self.rgb_transforms = self._get_rgb_transforms(is_train=is_train)

    def _get_rgb_transforms(self, is_train, defer_transform=False):
        img_transforms = []
        if is_train:
            img_transforms.append(transforms.RandomHorizontalFlip())
            if not defer_transform:
                img_transforms.extend([
                    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25),
                    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(saturation=(0.7, 1.2))]), p=0.25),
                    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(brightness=(0.7, 1.2))]), p=0.25),
                    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(contrast=(0.7, 1.2))]), p=0.25),
                    transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25)
                ])
        if not defer_transform:
            img_transforms.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        return transforms.Compose(img_transforms)

    def __call__(self, image, keypoints):
        keypoints = np.array(keypoints).reshape(-1, 2)
        w, h = image.size

        # Random crop for training
        if self.is_train:
            # Random crop to self.crop_size
            i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                image, output_size=(self.crop_size, self.crop_size))
            image = F.crop(image, i, j, h_crop, w_crop)

            # Adjust keypoints to match crop
            for idx in range(len(keypoints)):
                if keypoints[idx][0] != -1 and keypoints[idx][1] != -1:
                    # Adjust x coordinate
                    keypoints[idx][0] -= j
                    if keypoints[idx][0] < 0 or keypoints[idx][0] >= w_crop:
                        keypoints[idx] = [-1, -1]
                        continue

                    # Adjust y coordinate
                    keypoints[idx][1] -= i
                    if keypoints[idx][1] < 0 or keypoints[idx][1] >= h_crop:
                        keypoints[idx] = [-1, -1]
                        continue

            # Random horizontal flip
            if random.random() > 0.5:
                image = F.hflip(image)
                for idx in range(len(keypoints)):
                    if keypoints[idx][0] != -1:
                        keypoints[idx][0] = w_crop - keypoints[idx][0]

            w, h = w_crop, h_crop
        else:
            # For validation, center crop
            image = F.center_crop(image, (self.crop_size, self.crop_size))

            # Adjust keypoints to match center crop
            left = (w - self.crop_size) // 2
            top = (h - self.crop_size) // 2
            for idx in range(len(keypoints)):
                if keypoints[idx][0] != -1 and keypoints[idx][1] != -1:
                    # Adjust coordinates
                    keypoints[idx][0] -= left
                    keypoints[idx][1] -= top

                    # Check if keypoint is within crop
                    if (keypoints[idx][0] < 0 or keypoints[idx][0] >= self.crop_size or
                        keypoints[idx][1] < 0 or keypoints[idx][1] >= self.crop_size):
                        keypoints[idx] = [-1, -1]

            w, h = self.crop_size, self.crop_size

        # Convert image to tensor
        image = F.to_tensor(image)
        image = self.rgb_transforms(image)

        # Generate keypoint heatmaps
        heatmaps = self._generate_heatmaps(keypoints, w, h)

        # Convert keypoints to normalized coordinates
        normalized_keypoints = np.zeros_like(keypoints, dtype=np.float32)
        for i in range(len(keypoints)):
            if keypoints[i][0] != -1 and keypoints[i][1] != -1:
                normalized_keypoints[i][0] = keypoints[i][0] / w
                normalized_keypoints[i][1] = keypoints[i][1] / h
            else:
                normalized_keypoints[i][0] = -1
                normalized_keypoints[i][1] = -1

        return image, torch.FloatTensor(normalized_keypoints), heatmaps

    def _generate_heatmaps(self, keypoints, width, height, sigma=3):
        heatmap_h = height // 4  # Output stride of 4
        heatmap_w = width // 4

        heatmaps = np.zeros((self.num_keypoints, heatmap_h, heatmap_w), dtype=np.float32)

        for i in range(len(keypoints)):
            if keypoints[i][0] == -1 or keypoints[i][1] == -1:
                continue

            x = int(keypoints[i][0] * heatmap_w / width)
            y = int(keypoints[i][1] * heatmap_h / height)

            if x < 0 or y < 0 or x >= heatmap_w or y >= heatmap_h:
                continue

            size = 6 * sigma + 3
            x0 = max(0, x - size // 2)
            y0 = max(0, y - size // 2)
            x1 = min(heatmap_w, x + size // 2 + 1)
            y1 = min(heatmap_h, y + size // 2 + 1)

            for map_y in range(y0, y1):
                for map_x in range(x0, x1):
                    d2 = (map_x - x) ** 2 + (map_y - y) ** 2
                    heatmaps[i, map_y, map_x] = np.exp(-d2 / (2 * sigma ** 2))

        return torch.FloatTensor(heatmaps)


class PoseDataset(Dataset):
    def __init__(self, json_file, img_dir, is_train=True, crop_size=224, num_keypoints=42):
        self.img_dir = img_dir
        self.is_train = is_train
        self.crop_size = crop_size
        self.num_keypoints = num_keypoints

        # Load annotations - format is a dictionary where keys are image names
        # and values are lists of keypoints
        with open(json_file, 'r') as f:
            self.annotations = json.load(f)

        # Convert dictionary to list for indexing
        self.image_names = list(self.annotations.keys())

        self.transform = JointTransform(crop_size=crop_size, is_train=is_train,
                                        num_keypoints=num_keypoints)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get image name and keypoints
        image_name = self.image_names[idx]
        keypoints = self.annotations[image_name]

        # Load image
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        image_tensor, keypoint_tensor, heatmap_tensor = self.transform(image, keypoints)

        sample = {
            'image': image_tensor,
            'keypoints': keypoint_tensor,
            'heatmaps': heatmap_tensor,
            'image_id': image_name
        }

        return sample


def create_dataloaders(train_json, val_json, train_img_dir, val_img_dir,
                       batch_size=32, num_workers=4, crop_size=224, num_keypoints=42):

    train_dataset = PoseDataset(
        json_file=train_json,
        img_dir=train_img_dir,
        is_train=True,
        crop_size=crop_size,
        num_keypoints=num_keypoints
    )

    val_dataset = PoseDataset(
        json_file=val_json,
        img_dir=val_img_dir,
        is_train=False,
        crop_size=crop_size,
        num_keypoints=num_keypoints
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Example paths to your JSON files and image directories
    train_json = "/home/huyanh/Projects/TouchEvent/Data/Ego4DExo/pose_imgs/train.json"
    val_json = "/home/huyanh/Projects/TouchEvent/Data/Ego4DExo/pose_imgs/val.json"
    train_img_dir = "/home/huyanh/Projects/TouchEvent/Data/Ego4DExo/pose_imgs/train"
    val_img_dir = "/home/huyanh/Projects/TouchEvent/Data/Ego4DExo/pose_imgs/val"

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_json=train_json,
        val_json=val_json,
        train_img_dir=train_img_dir,
        val_img_dir=val_img_dir,
        batch_size=32,
        num_workers=4,
        crop_size=224,
        num_keypoints=42
    )

    # Get a batch from train_loader
    for batch in train_loader:
        images = batch['image']  # Shape: [batch_size, 3, crop_size, crop_size]
        keypoints = batch['keypoints']  # Shape: [batch_size, num_keypoints, 2]
        heatmaps = batch['heatmaps']  # Shape: [batch_size, num_keypoints, H/4, W/4]

        print(f"Batch images shape: {images.shape}")
        print(f"Batch keypoints shape: {keypoints.shape}")
        print(f"Batch heatmaps shape: {heatmaps.shape}")

        # DEBUG >>>
        import os

        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        # Create output directory
        os.makedirs('debug_output', exist_ok=True)

        # Process batch elements
        for b in range(min(images.shape[0], 4)):  # Limit to first 4 samples
            # 1. Save the RGB image
            img = images[b].cpu().detach().numpy().transpose(1, 2, 0)
            # Denormalize image
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)

            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"Image {b}")
            plt.axis('off')
            plt.savefig(f'debug_output/image_{b}.jpg')
            plt.close()

            # 2. Create a single combined heatmap from all 42 keypoints
            hmap = heatmaps[b].cpu().detach().numpy()
            combined_heatmap = np.max(hmap, axis=0)  # Max across all keypoints

            # Save the combined heatmap
            plt.figure(figsize=(6, 6))
            plt.imshow(combined_heatmap, cmap='jet')
            plt.title(f"All Keypoints Combined {b}")
            plt.colorbar()
            plt.axis('off')
            plt.savefig(f'debug_output/heatmap_all_{b}.jpg')
            plt.close()

            # 3. Save overlay of combined heatmap on image
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.imshow(combined_heatmap, cmap='jet', alpha=0.6)
            plt.title(f"Overlay of All Keypoints {b}")
            plt.axis('off')
            plt.savefig(f'debug_output/heatmap_overlay_{b}.jpg')
            plt.close()

        print(f"Debug images saved to 'debug_output/' directory")
        # ENDDEBUG >>>
        break  # Just check one batch
