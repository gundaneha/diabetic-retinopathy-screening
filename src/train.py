import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import sys
sys.path.append('src')
from dataset import get_dataloaders
from model import get_model

# ─── CONFIG ───────────────────────────────────────────────
EPOCHS      = 5
LR          = 1e-4
SAVE_PATH   = 'data/best_model.pth'
# ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    avg_loss = total_loss / len(loader)
    kappa    = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    acc      = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, kappa, acc


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    avg_loss = total_loss / len(loader)
    kappa    = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    acc      = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, kappa, acc


def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_kappa'], label='Train')
    axes[1].plot(history['val_kappa'],   label='Val')
    axes[1].set_title('Quadratic Weighted Kappa')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    axes[2].plot(history['train_acc'], label='Train')
    axes[2].plot(history['val_acc'],   label='Val')
    axes[2].set_title('Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('data/training_history.png')
    plt.show()
    print("📈 Training history saved to data/training_history.png")


def train():
    print("="*50)
    print("  DR SCREENING — TRAINING")
    print("="*50)

    # Load data and model
    train_loader, val_loader, test_loader, class_weights = get_dataloaders()
    model     = get_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_kappa = -1
    history = {
        'train_loss': [], 'val_loss': [],
        'train_kappa': [], 'val_kappa': [],
        'train_acc': [], 'val_acc': []
    }

    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print(f"⚠️  CPU training — each epoch ~15-20 mins\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*40}")
        print(f"  EPOCH {epoch} / {EPOCHS}")
        print(f"{'='*40}")

        train_loss, train_kappa, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_kappa, val_acc = validate(
            model, val_loader, criterion
        )
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_kappa'].append(train_kappa)
        history['val_kappa'].append(val_kappa)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Train  — Loss: {train_loss:.4f} | Kappa: {train_kappa:.4f} | Acc: {train_acc:.4f}")
        print(f"   Val    — Loss: {val_loss:.4f} | Kappa: {val_kappa:.4f} | Acc: {val_acc:.4f}")

        # Save best model
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"   💾 Best model saved! (Kappa: {best_kappa:.4f})")

    print(f"\n🏆 Training Complete!")
    print(f"🏆 Best Val Kappa: {best_kappa:.4f}")

    plot_history(history)
    return model


if __name__ == "__main__":
    train()