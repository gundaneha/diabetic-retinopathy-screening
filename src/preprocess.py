import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ─── CONFIG ───────────────────────────────────────────────
RAW_DIR      = r"data\raw\train_images"
CSV_PATH     = r"data\raw\train.csv"
OUTPUT_DIR   = r"data\processed"
IMG_SIZE     = 224
# ──────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def ben_graham(image, sigma=10):
    """Enhance blood vessel visibility using Ben Graham preprocessing."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    enhanced = cv2.addWeighted(image, 4, blurred, -4, 128)
    mask = np.zeros(enhanced.shape, dtype=np.uint8)
    h, w = enhanced.shape[:2]
    cv2.circle(mask, (w//2, h//2), int(min(h,w) * 0.45), (1,1,1), -1)
    enhanced = enhanced * mask + 128 * (1 - mask)
    return enhanced.astype(np.uint8)

def preprocess_and_save(df):
    """Preprocess all images and save to output directory."""
    print(f"\n📁 Processing {len(df)} images...")
    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(RAW_DIR, row['id_code'] + '.png')

        if not os.path.exists(img_path):
            failed.append(row['id_code'])
            continue

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            failed.append(row['id_code'])
            continue

        # Apply Ben Graham preprocessing
        img = ben_graham(img)

        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Save
        out_path = os.path.join(OUTPUT_DIR, row['id_code'] + '.png')
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"✅ Done! Processed images saved to: {OUTPUT_DIR}")
    if failed:
        print(f"⚠️  Failed to process {len(failed)} images: {failed[:5]}")

def show_eda(df):
    """Show basic exploratory data analysis."""
    print("\n📊 Dataset Overview:")
    print(df.head())
    print(f"\nTotal images : {len(df)}")
    print(f"Label counts :\n{df['diagnosis'].value_counts().sort_index()}")

    labels = {0:'No DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferative'}

    plt.figure(figsize=(8,4))
    sns.countplot(x='diagnosis', data=df, palette='viridis')
    plt.xticks(ticks=[0,1,2,3,4], labels=labels.values())
    plt.title('Class Distribution in APTOS 2019 Dataset')
    plt.xlabel('DR Severity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('data/class_distribution.png')
    plt.show()
    print("📈 Chart saved to data/class_distribution.png")

def show_samples(df):
    """Show sample images from each class."""
    labels = {0:'No DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferative'}
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Sample Images — Original vs Preprocessed', fontsize=14)

    for i in range(5):
        sample = df[df['diagnosis'] == i].iloc[0]
        img_path = os.path.join(RAW_DIR, sample['id_code'] + '.png')
        img = cv2.imread(img_path)

        # Original
        axes[0][i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0][i].set_title(f'Original\n{labels[i]}')
        axes[0][i].axis('off')

        # Preprocessed
        processed = ben_graham(img)
        processed = cv2.resize(processed, (IMG_SIZE, IMG_SIZE))
        axes[1][i].imshow(processed)
        axes[1][i].set_title(f'Preprocessed\n{labels[i]}')
        axes[1][i].axis('off')

    plt.tight_layout()
    plt.savefig('data/sample_images.png')
    plt.show()
    print("🖼️  Sample images saved to data/sample_images.png")

# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    print("="*50)
    print("  DR SCREENING — PREPROCESSING PIPELINE")
    print("="*50)

    # Step 1 - EDA
    show_eda(df)

    # Step 2 - Sample visualization
    show_samples(df)

    # Step 3 - Preprocess and save all images
    preprocess_and_save(df)

    print("\n🚀 Preprocessing complete! Ready for model training.")