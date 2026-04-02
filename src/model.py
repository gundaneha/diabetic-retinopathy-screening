import torch
import torch.nn as nn
import torchvision.models as models

# ─── CONFIG ───────────────────────────────────────────────
NUM_CLASSES = 5
DROPOUT     = 0.4
# ──────────────────────────────────────────────────────────

class DRModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super(DRModel, self).__init__()

        # Load pretrained EfficientNet B3
        self.backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )

        # Freeze early layers — only train last 30%
        total_layers = len(list(self.backbone.features.children()))
        freeze_until = int(total_layers * 0.70)
        for i, layer in enumerate(self.backbone.features.children()):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def get_model():
    model = DRModel()

    # Count trainable parameters
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ Model     : EfficientNet-B3 + Custom Head")
    print(f"✅ Total params     : {total:,}")
    print(f"✅ Trainable params : {trainable:,}")
    print(f"✅ Frozen params    : {total - trainable:,}")

    return model


if __name__ == "__main__":
    print("="*50)
    print("  DR SCREENING — MODEL SETUP")
    print("="*50)

    model = get_model()

    # Test forward pass
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)

    print(f"\n✅ Input shape  : {dummy.shape}")
    print(f"✅ Output shape : {out.shape}")
    print(f"✅ Output sample: {out[0].detach().numpy().round(3)}")
    print(f"\n🚀 Model ready for training!")