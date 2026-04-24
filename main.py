import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "celeba",
    "img_align_celeba",
    "img_align_celeba"
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEMO_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "demo_outputs")

MODEL_PATH = os.path.join(MODELS_DIR, "simple_srcnn_20epochs_final.pth")
COMPARISON_PATH = os.path.join(DEMO_OUTPUTS_DIR, "lr_sr_hr_comparison.png")
LOSS_CURVE_PATH = os.path.join(DEMO_OUTPUTS_DIR, "loss_curve.png")
PSNR_PATH = os.path.join(DEMO_OUTPUTS_DIR, "psnr_results.txt")

HR_SIZE = 128
LR_SIZE = 32
MAX_IMAGES = 2000
BATCH_SIZE = 16
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET
# ============================================================

class FaceSRDataset(Dataset):
    def __init__(self, img_dir, hr_size=128, lr_size=32, max_images=None):
        self.img_dir = img_dir
        self.hr_size = hr_size
        self.lr_size = lr_size

        self.files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if max_images is not None:
            self.files = self.files[:max_images]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")

        hr_img = img.resize((self.hr_size, self.hr_size), Image.BICUBIC)

        lr_small = hr_img.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        lr_img = lr_small.resize((self.hr_size, self.hr_size), Image.BICUBIC)

        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)

        return lr_tensor, hr_tensor


# ============================================================
# MODEL
# ============================================================

class SimpleSRCNN(nn.Module):
    def __init__(self):
        super(SimpleSRCNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dataset folder not found:\n{DATA_DIR}")

    full_dataset = FaceSRDataset(
        img_dir=DATA_DIR,
        hr_size=HR_SIZE,
        lr_size=LR_SIZE,
        max_images=MAX_IMAGES
    )

    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return full_dataset, train_dataset, val_dataset, train_loader, val_loader


# ============================================================
# LOAD MODEL
# ============================================================

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Saved model not found:\n{MODEL_PATH}")

    model = SimpleSRCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return model


# ============================================================
# SHOW SAVED OUTPUTS
# ============================================================

def show_saved_outputs():
    print("\n===== SAVED PROJECT OUTPUTS =====")

    if os.path.exists(COMPARISON_PATH):
        comparison_img = Image.open(COMPARISON_PATH)

        plt.figure(figsize=(12, 4))
        plt.imshow(comparison_img)
        plt.title("Saved LR vs SR vs HR Comparison")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print("Comparison image not found:", COMPARISON_PATH)

    if os.path.exists(LOSS_CURVE_PATH):
        loss_curve_img = Image.open(LOSS_CURVE_PATH)

        plt.figure(figsize=(8, 5))
        plt.imshow(loss_curve_img)
        plt.title("Saved Training vs Validation Loss Curve")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        print("Loss curve image not found:", LOSS_CURVE_PATH)

    if os.path.exists(PSNR_PATH):
        print("\n===== SAVED PSNR RESULTS =====")
        with open(PSNR_PATH, "r") as f:
            print(f.read())
    else:
        print("\nPSNR results file not found.")


# ============================================================
# EVALUATION METRICS
# ============================================================

def evaluate_metrics(model, val_loader):
    model.eval()

    mse_list = []
    psnr_list = []
    ssim_list = []

    sample_lr = None
    sample_sr = None
    sample_hr = None

    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)

            sr_imgs = model(lr_imgs)
            sr_imgs = torch.clamp(sr_imgs, 0.0, 1.0)

            batch_mse = F.mse_loss(sr_imgs, hr_imgs, reduction="mean").item()
            mse_list.append(batch_mse)

            if batch_mse == 0:
                batch_psnr = 100.0
            else:
                batch_psnr = 10 * math.log10(1.0 / batch_mse)

            psnr_list.append(batch_psnr)

            sr_np = sr_imgs.cpu().permute(0, 2, 3, 1).numpy()
            hr_np = hr_imgs.cpu().permute(0, 2, 3, 1).numpy()

            for i in range(sr_np.shape[0]):
                ssim_val = ssim(
                    hr_np[i],
                    sr_np[i],
                    channel_axis=2,
                    data_range=1.0
                )
                ssim_list.append(ssim_val)

            if sample_lr is None:
                sample_lr = lr_imgs[0].cpu().permute(1, 2, 0).numpy()
                sample_sr = sr_imgs[0].cpu().permute(1, 2, 0).numpy()
                sample_hr = hr_imgs[0].cpu().permute(1, 2, 0).numpy()

    avg_mse = np.mean(mse_list)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print("\n===== EVALUATION METRICS =====")
    print(f"Average MSE  : {avg_mse:.6f}")
    print(f"Average PSNR : {avg_psnr:.4f} dB")
    print(f"Average SSIM : {avg_ssim:.4f}")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(sample_lr)
    plt.title("LR Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(sample_sr)
    plt.title("SR Output")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(sample_hr)
    plt.title("HR Target")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# CUSTOM DEMO
# ============================================================

def run_custom_demo(model, image_path):
    if not os.path.exists(image_path):
        print("\nNo custom test image found.")
        print("Add a face image named test_face.jpg inside the project folder.")
        return

    img = Image.open(image_path).convert("RGB")
    orig_img = img.resize((HR_SIZE, HR_SIZE), Image.BICUBIC)

    lr_small = orig_img.resize((LR_SIZE, LR_SIZE), Image.BICUBIC)
    lr_input = lr_small.resize((HR_SIZE, HR_SIZE), Image.BICUBIC)

    to_tensor = transforms.ToTensor()
    input_tensor = to_tensor(lr_input).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

    orig_np = to_tensor(orig_img).permute(1, 2, 0).numpy()
    lr_np = input_tensor[0].cpu().permute(1, 2, 0).numpy()
    out_np = output_tensor[0].cpu().permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(orig_np)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(lr_np)
    plt.title("Blurred Input")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(out_np)
    plt.title("Enhanced Output")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("Using device:", DEVICE)
    print("Project root:", PROJECT_ROOT)
    print("Dataset directory:", DATA_DIR)

    full_dataset, train_dataset, val_dataset, train_loader, val_loader = create_dataloaders()

    print("\n===== DATA READY =====")
    print("Full dataset size:", len(full_dataset))
    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    model = load_trained_model()

    print("\n===== MODEL READY =====")
    print("Loaded model from:", MODEL_PATH)

    show_saved_outputs()

    evaluate_metrics(model, val_loader)

    print("\n===== CUSTOM IMAGE DEMO =====")
    test_image_path = os.path.join(PROJECT_ROOT, "test_face.jpg")
    run_custom_demo(model, test_image_path)


if __name__ == "__main__":
    main()