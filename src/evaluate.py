import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize
import torchvision.transforms as transforms
import sys
sys.path.append('src')
from dataset import get_dataloaders
from model import get_model

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH    = 'data/best_model.pth'
PROCESSED_DIR = r"data\processed"
LABELS        = {0:'No DR', 1:'Mild', 2:'Moderate',
                 3:'Severe', 4:'Proliferative'}
# ──────────────────────────────────────────────────────────


def load_model():
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print("✅ Best model loaded!")
    return model


def evaluate_model(model, test_loader):
    all_preds   = []
    all_labels  = []
    all_probs   = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS.values(),
                yticklabels=LABELS.values())
    plt.title('Confusion Matrix — DR Classification')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('data/confusion_matrix.png')
    plt.show()
    print("📊 Confusion matrix saved!")


def grad_cam(model, img_tensor, target_class=None):
    """Generate Grad-CAM heatmap for a single image."""
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    # Hook into last conv layer
    handle = model.backbone.features[-1].register_forward_hook(forward_hook)

    # Forward pass
    output = model(img_tensor.unsqueeze(0))
    if target_class is None:
        target_class = output.argmax().item()

    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()

    handle.remove()

    # Compute Grad-CAM
    grad    = gradients[0].squeeze(0).mean(dim=[1, 2])
    act     = activations[0].squeeze(0)
    weights = grad

    cam = torch.zeros(act.shape[1:])
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = torch.relu(cam)
    cam = cam.detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, target_class


def show_gradcam_samples(model, test_df):
    """Show Grad-CAM for one sample from each DR class."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle('Grad-CAM — What the Model Looks At', fontsize=14)

    for i in range(5):
        sample   = test_df[test_df['diagnosis'] == i]
        if len(sample) == 0:
            continue
        sample   = sample.iloc[0]
        img_path = os.path.join(PROCESSED_DIR, sample['id_code'] + '.png')
        img      = Image.open(img_path).convert('RGB')
        img_t    = transform(img)

        # Grad-CAM
        cam, pred = grad_cam(model, img_t, target_class=i)
        cam_resized = np.array(Image.fromarray(
            (cam * 255).astype(np.uint8)).resize((224, 224)))

        # Original
        axes[0][i].imshow(img.resize((224, 224)))
        axes[0][i].set_title(f'Actual: {LABELS[i]}')
        axes[0][i].axis('off')

        # Grad-CAM overlay
        axes[1][i].imshow(img.resize((224, 224)))
        axes[1][i].imshow(cam_resized, cmap='jet', alpha=0.45)
        axes[1][i].set_title(f'Predicted: {LABELS[pred]}')
        axes[1][i].axis('off')

    plt.tight_layout()
    plt.savefig('data/gradcam.png')
    plt.show()
    print("🔥 Grad-CAM saved to data/gradcam.png")


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*50)
    print("  DR SCREENING — EVALUATION")
    print("="*50)

    # Load data and model
    _, _, test_loader, _ = get_dataloaders()
    model = load_model()

    # Evaluate
    print("\n📊 Running evaluation on test set...")
    labels, preds, probs = evaluate_model(model, test_loader)

    # Metrics
    from sklearn.metrics import cohen_kappa_score, accuracy_score
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    acc   = accuracy_score(labels, preds)

    print(f"\n🏆 Final Test Results:")
    print(f"   Quadratic Weighted Kappa : {kappa:.4f}")
    print(f"   Accuracy                 : {acc:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(labels, preds,
                                target_names=list(LABELS.values())))

    # Confusion matrix
    plot_confusion_matrix(labels, preds)

    # Grad-CAM
    print("\n🔥 Generating Grad-CAM visualizations...")
    test_df = pd.read_csv('data/test_split.csv')
    show_gradcam_samples(model, test_df)

    print("\n✅ Evaluation complete!")
    print("📁 Check data/ folder for all saved outputs!")