# Face Super-Resolution using SRCNN

This project implements a simple face super-resolution pipeline using a CNN-based SRCNN model on the CelebA dataset.

## Project Objective
The goal of this project is to enhance low-resolution facial images and reconstruct a higher-quality version with improved visual detail.

## Methodology
- Used the **CelebA dataset** for face images
- Generated **low-resolution (LR)** inputs by downsampling and bicubic upscaling
- Used the original resized face image as the **high-resolution (HR)** target
- Trained a **Simple SRCNN model** to learn the mapping:
  - **Blurred LR face → Enhanced SR face**

## Model Architecture
The model is a simple CNN with:
- Conv layer: 3 → 64 channels
- Conv layer: 64 → 32 channels
- Conv layer: 32 → 3 channels

This follows the classical SRCNN-style approach for image super-resolution.

## Evaluation Metrics
The project uses:
- **MSE (Mean Squared Error)**
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**

### Reported Results
- **Bicubic PSNR (LR vs HR):** 25.2415 dB
- **Model PSNR (SR vs HR):** 26.1526 dB
- **PSNR Improvement:** +0.9111 dB
- **Approximate reconstruction error reduction:** ~19%

## Key Observation
The model performs better than standard bicubic interpolation, but since it is trained with pixel-wise loss (MSE), the output can still appear slightly smooth or blurry.

## Future Scope / Novelty Direction
The next improvement direction is to make the model more **face-aware** by incorporating:
- **Perceptual loss**
- **Facial structure / landmark guidance**
- Better preservation of important facial details like eyes, nose, and lips

## Included Files
- `main.py` → main project script
- `requirements.txt` → dependencies
- `test_face.jpg` → sample demo image
- `lr_sr_hr_comparison.png` → visual comparison output
- `training vs validation loss curve.png` → loss curve

## Note
The full CelebA dataset is not uploaded to this repository due to size limitations.
