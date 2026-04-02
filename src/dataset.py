import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch

# ─── CONFIG ───────────────────────────────────────────────
PROCESSED_DIR = r"data\processed"
CSV_PATH      = r"data\raw\train.csv"
IMG_SIZE      = 224
BATCH_SIZE    = 16
RANDOM_SEED   = 42
# ──────────────────────────────────────────────────────────

class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['id_code'] + '.png')
        image    = Image.open(img_path).convert('RGB')
        label    = int(row['diagnosis'])
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def get_dataloaders():
    df = pd.read_csv(CSV_PATH)

    # Split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.30,
        stratify=df['diagnosis'],
        random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50,
        stratify=temp_df['diagnosis'],
        random_state=RANDOM_SEED
    )

    print(f"✅ Train : {len(train_df)} images")
    print(f"✅ Val   : {len(val_df)} images")
    print(f"✅ Test  : {len(test_df)} images")

    # Save splits
    train_df.to_csv('data/train_split.csv', index=False)
    val_df.to_csv('data/val_split.csv',   index=False)
    test_df.to_csv('data/test_split.csv',  index=False)
    print("💾 Splits saved to data/")

    train_transform, val_transform = get_transforms()

    train_dataset = DRDataset(train_df, PROCESSED_DIR, train_transform)
    val_dataset   = DRDataset(val_df,   PROCESSED_DIR, val_transform)
    test_dataset  = DRDataset(test_df,  PROCESSED_DIR, val_transform)

    # Class weights for imbalance handling
    class_counts  = train_df['diagnosis'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    class_weights = torch.FloatTensor(class_weights)
    print(f"\n⚖️  Class weights: {class_weights.numpy().round(4)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    print("="*50)
    print("  DR SCREENING — DATASET SETUP")
    print("="*50)
    train_loader, val_loader, test_loader, weights = get_dataloaders()

    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\n✅ Batch shape : {images.shape}")
    print(f"✅ Labels      : {labels.tolist()}")
    print(f"\n🚀 Dataset ready for training!")