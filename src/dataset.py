import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

from src.config import (
    DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    VAL_SPLIT, TEST_SPLIT, RANDOM_SEED,
)


def get_image_paths_and_labels(data_dir: Path):
    paths, labels = [], []
    for img_path in sorted((data_dir / "good").glob("*.[jJ][pP][gG]")):
        paths.append(img_path)
        labels.append(0)
    for img_path in sorted((data_dir / "bad").glob("*.[jJ][pP][gG]")):
        paths.append(img_path)
        labels.append(1)
    return paths, labels


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DefectDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def build_loaders(data_dir: Path = DATA_DIR):
    paths, labels = get_image_paths_and_labels(data_dir)
    paths, labels = np.array(paths), np.array(labels)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=labels,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=train_labels,
    )

    train_ds = DefectDataset(train_paths.tolist(), train_labels.tolist(), TRAIN_TRANSFORM)
    val_ds = DefectDataset(val_paths.tolist(), val_labels.tolist(), VAL_TRANSFORM)
    test_ds = DefectDataset(test_paths.tolist(), test_labels.tolist(), VAL_TRANSFORM)

    class_counts = np.bincount(train_labels.astype(int))
    weights = 1.0 / class_counts
    sample_weights = [weights[int(l)] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader
