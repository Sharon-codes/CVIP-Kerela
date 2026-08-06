#!/usr/bin/env python
"""
Experiment 3: Physical Sensor Robustness Suite (V2)
Applies Gaussian noise, rotation, brightness/contrast shifts, and JPEG compression
perturbations to external validation images and measures ROC-AUC degradation
for Hybrid CNN-TDA-SVM vs. EfficientNet-B0 (CNN Only).

Author: Lead Biomedical ML Engineer (IEEE Q1 Submission Suite)
"""

import os
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from core_pipeline import load_real_images, extract_cnn, extract_tda
from exp_baselines import EfficientNetB0FeatureExtractor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# Perturbation functions
def apply_gaussian_noise(images, sigma):
    noisy = []
    for img in images:
        noise = np.random.normal(0, sigma, img.shape)
        n_img = np.clip(img + noise, 0.0, 1.0)
        noisy.append(n_img)
    return np.array(noisy)


def apply_rotation(images, angle):
    rotated = []
    for img in images:
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        rotated.append(rot)
    return np.array(rotated)


def apply_brightness_contrast(images, shift_pct):
    # shift_pct e.g. -0.20, -0.10, +0.10, +0.20
    shifted = []
    for img in images:
        # Contrast scale = 1 + shift_pct, brightness shift = shift_pct
        s_img = np.clip(img * (1.0 + shift_pct) + shift_pct * 0.1, 0.0, 1.0)
        shifted.append(s_img)
    return np.array(shifted)


def apply_jpeg_compression(images, quality):
    compressed = []
    for img in images:
        img_8u = (img * 255.0).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        result, enc = cv2.imencode('.jpg', img_8u, encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        compressed.append(dec.astype(np.float32) / 255.0)
    return np.array(compressed)


def run_robustness_v2_experiment(saved_probs):
    print("=" * 80)
    print("EXPERIMENT 3: PHYSICAL SENSOR ROBUSTNESS SUITE (NOISE, ROTATION, CONTRAST, JPEG)")
    print("=" * 80)

    X_imgs_ext, y_ext, _, _, _ = load_real_images("data/external")

    hybrid_pipe = saved_probs["Hybrid CNN-TDA (Final Pipeline)"]["pipe"]
    effnet_pipe = saved_probs["EfficientNet-B0 (CNN Only)"]["pipe"]

    effnet_extractor = EfficientNetB0FeatureExtractor()

    robustness_results = {
        "gaussian": {},
        "rotation": {},
        "contrast": {},
        "jpeg": {}
    }

    # 1. Gaussian Noise (sigma = 0.01, 0.05, 0.10)
    print("\n[+] Testing Gaussian Noise Degradation...")
    for sigma in [0.01, 0.05, 0.10]:
        imgs_p = apply_gaussian_noise(X_imgs_ext, sigma)
        
        # Hybrid features
        cnn_p = extract_cnn(imgs_p)
        tda_p = extract_tda(imgs_p)
        hyb_p = np.hstack([tda_p, cnn_p])
        prob_hyb = hybrid_pipe.predict_proba(hyb_p)[:, 1]
        auc_hyb = roc_auc_score(y_ext, prob_hyb)

        # EffNet features
        eff_p = effnet_extractor.extract(imgs_p)
        prob_eff = effnet_pipe.predict_proba(eff_p)[:, 1]
        auc_eff = roc_auc_score(y_ext, prob_eff)

        print(f"  Gaussian sigma={sigma:.2f} | Hybrid AUC: {auc_hyb:.4f} | EffNet-B0 AUC: {auc_eff:.4f} | Delta: +{auc_hyb - auc_eff:.4f}")
        robustness_results["gaussian"][sigma] = {"hybrid": auc_hyb, "effnet": auc_eff}

    # 2. Rotation (15, 45, 90 deg)
    print("\n[+] Testing Rotation Perturbation...")
    for angle in [15, 45, 90]:
        imgs_p = apply_rotation(X_imgs_ext, angle)

        cnn_p = extract_cnn(imgs_p)
        tda_p = extract_tda(imgs_p)
        hyb_p = np.hstack([tda_p, cnn_p])
        prob_hyb = hybrid_pipe.predict_proba(hyb_p)[:, 1]
        auc_hyb = roc_auc_score(y_ext, prob_hyb)

        eff_p = effnet_extractor.extract(imgs_p)
        prob_eff = effnet_pipe.predict_proba(eff_p)[:, 1]
        auc_eff = roc_auc_score(y_ext, prob_eff)

        print(f"  Rotation {angle}° | Hybrid AUC: {auc_hyb:.4f} | EffNet-B0 AUC: {auc_eff:.4f} | Delta: +{auc_hyb - auc_eff:.4f}")
        robustness_results["rotation"][angle] = {"hybrid": auc_hyb, "effnet": auc_eff}

    # 3. Brightness/Contrast Shift (-20%, -10%, +10%, +20%)
    print("\n[+] Testing Brightness/Contrast Shift...")
    for shift in [-0.20, -0.10, 0.10, 0.20]:
        imgs_p = apply_brightness_contrast(X_imgs_ext, shift)

        cnn_p = extract_cnn(imgs_p)
        tda_p = extract_tda(imgs_p)
        hyb_p = np.hstack([tda_p, cnn_p])
        prob_hyb = hybrid_pipe.predict_proba(hyb_p)[:, 1]
        auc_hyb = roc_auc_score(y_ext, prob_hyb)

        eff_p = effnet_extractor.extract(imgs_p)
        prob_eff = effnet_pipe.predict_proba(eff_p)[:, 1]
        auc_eff = roc_auc_score(y_ext, prob_eff)

        print(f"  Shift {shift*100:+.0f}% | Hybrid AUC: {auc_hyb:.4f} | EffNet-B0 AUC: {auc_eff:.4f} | Delta: +{auc_hyb - auc_eff:.4f}")
        robustness_results["contrast"][shift] = {"hybrid": auc_hyb, "effnet": auc_eff}

    # 4. JPEG Compression (Q = 90, 70, 50, 30)
    print("\n[+] Testing JPEG Compression Loss...")
    for q in [90, 70, 50, 30]:
        imgs_p = apply_jpeg_compression(X_imgs_ext, q)

        cnn_p = extract_cnn(imgs_p)
        tda_p = extract_tda(imgs_p)
        hyb_p = np.hstack([tda_p, cnn_p])
        prob_hyb = hybrid_pipe.predict_proba(hyb_p)[:, 1]
        auc_hyb = roc_auc_score(y_ext, prob_hyb)

        eff_p = effnet_extractor.extract(imgs_p)
        prob_eff = effnet_pipe.predict_proba(eff_p)[:, 1]
        auc_eff = roc_auc_score(y_ext, prob_eff)

        print(f"  JPEG Q={q:02d} | Hybrid AUC: {auc_hyb:.4f} | EffNet-B0 AUC: {auc_eff:.4f} | Delta: +{auc_hyb - auc_eff:.4f}")
        robustness_results["jpeg"][q] = {"hybrid": auc_hyb, "effnet": auc_eff}

    return robustness_results


if __name__ == "__main__":
    from exp_baselines import run_baselines_experiment
    _, saved_probs = run_baselines_experiment()
    run_robustness_v2_experiment(saved_probs)
