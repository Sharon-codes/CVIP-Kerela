#!/usr/bin/env python
"""
Robustness Experiments Module:
- Exp 1: Noise Degradation (Gaussian, Salt & Pepper, Poisson)
- Exp 2: JPEG Lossy Compression (Quality 100, 90, 80, 70, 50)
- Exp 4: ROI Bounding Box Sensitivity (Perturbations 0%, 5%, 10%, 20%)
- Exp 9: Biomechanical Elastic Deformation (Alpha 0, 5, 10, 20)

Generates 300 DPI Seaborn whitegrid plots proving the topological hardware shield.
Author: Lead Biomedical ML Engineer (Q1 Journal Submission Suite)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

from physics_engine import (
    apply_gaussian_noise,
    apply_salt_pepper_noise,
    apply_poisson_noise,
    apply_jpeg_compression,
    apply_elastic_deformation,
    apply_roi_perturbation
)


# =====================================================================
# Experiment 1: Noise Degradation
# =====================================================================
def run_experiment_1_noise(X_val, y_val, cnn_model, hybrid_pipeline, extract_cnn_fn, extract_tda_fn, save_dir="images", dpi=300):
    print("\n[+] Running Experiment 1: Sensor Noise Degradation Suite...")

    noise_conditions = [
        ("Clean", None, None),
        ("Gauss 0.01", apply_gaussian_noise, {"sigma": 0.01}),
        ("Gauss 0.03", apply_gaussian_noise, {"sigma": 0.03}),
        ("Gauss 0.05", apply_gaussian_noise, {"sigma": 0.05}),
        ("S&P 1%", apply_salt_pepper_noise, {"amount": 0.01}),
        ("S&P 3%", apply_salt_pepper_noise, {"amount": 0.03}),
        ("S&P 5%", apply_salt_pepper_noise, {"amount": 0.05}),
        ("Poisson", apply_poisson_noise, {})
    ]

    labels = []
    cnn_aucs = []
    hybrid_aucs = []

    for name, fn, kwargs in noise_conditions:
        labels.append(name)
        if fn is None:
            X_deg = X_val.copy()
        else:
            X_deg = fn(X_val, **kwargs)

        X_cnn_deg = extract_cnn_fn(X_deg)
        X_tda_deg = extract_tda_fn(X_deg)
        X_hybrid_deg = np.hstack([X_tda_deg, X_cnn_deg])

        cnn_prob = cnn_model.predict_proba(X_cnn_deg)[:, 1]
        hybrid_prob = hybrid_pipeline.predict_proba(X_hybrid_deg)[:, 1]

        auc_cnn = roc_auc_score(y_val, cnn_prob)
        auc_hybrid = roc_auc_score(y_val, hybrid_prob)

        cnn_aucs.append(auc_cnn)
        hybrid_aucs.append(auc_hybrid)

        print(f"  Condition: {name:12s} | CNN ROC-AUC: {auc_cnn:.4f} | Hybrid ROC-AUC: {auc_hybrid:.4f} | Delta: +{auc_hybrid - auc_cnn:.4f}")

    # Save 300 DPI Plot
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    x_idx = np.arange(len(labels))
    ax.plot(x_idx, cnn_aucs, marker='o', linewidth=2.5, markersize=8, color='#d9534f', label='Standard CNN Baseline (MobileNetV2)')
    ax.plot(x_idx, hybrid_aucs, marker='s', linewidth=2.5, markersize=8, color='#1f77b4', label='Hybrid CNN-TDA (Hardware Shield)')

    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, rotation=30, fontsize=10, fontweight='bold')
    ax.set_title("Exp 1: Sensor Noise Robustness under Physical Degradation", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("ROC-AUC Score", fontsize=11, fontweight='bold')
    ax.set_xlabel("Sensor Noise Condition", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10.5, frameon=True, loc='lower left')
    plt.tight_layout()

    out_path = os.path.join(save_dir, "exp1_noise_robustness.jpg")
    plt.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved plot to {out_path}")

    return {"labels": labels, "cnn_aucs": cnn_aucs, "hybrid_aucs": hybrid_aucs}


# =====================================================================
# Experiment 2: JPEG Lossy Compression
# =====================================================================
def run_experiment_2_jpeg(X_val, y_val, cnn_model, hybrid_pipeline, extract_cnn_fn, extract_tda_fn, save_dir="images", dpi=300):
    print("\n[+] Running Experiment 2: JPEG Compression Degradation Suite...")

    qualities = [100, 90, 80, 70, 50]
    cnn_aucs = []
    hybrid_aucs = []

    for q in qualities:
        if q == 100:
            X_deg = X_val.copy()
        else:
            X_deg = apply_jpeg_compression(X_val, quality=q)

        X_cnn_deg = extract_cnn_fn(X_deg)
        X_tda_deg = extract_tda_fn(X_deg)
        X_hybrid_deg = np.hstack([X_tda_deg, X_cnn_deg])

        cnn_prob = cnn_model.predict_proba(X_cnn_deg)[:, 1]
        hybrid_prob = hybrid_pipeline.predict_proba(X_hybrid_deg)[:, 1]

        auc_cnn = roc_auc_score(y_val, cnn_prob)
        auc_hybrid = roc_auc_score(y_val, hybrid_prob)

        cnn_aucs.append(auc_cnn)
        hybrid_aucs.append(auc_hybrid)

        print(f"  JPEG Quality: {q:3d} | CNN ROC-AUC: {auc_cnn:.4f} | Hybrid ROC-AUC: {auc_hybrid:.4f} | Delta: +{auc_hybrid - auc_cnn:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(qualities, cnn_aucs, marker='o', linewidth=2.5, markersize=8, color='#d9534f', label='Standard CNN Baseline')
    ax.plot(qualities, hybrid_aucs, marker='s', linewidth=2.5, markersize=8, color='#1f77b4', label='Hybrid CNN-TDA (Hardware Shield)')

    ax.invert_xaxis()  # 100 on left down to 50 on right
    ax.set_title("Exp 2: Lossy JPEG Compression Robustness", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("JPEG Quality Parameter (Lossy Compression Level)", fontsize=11, fontweight='bold')
    ax.set_ylabel("ROC-AUC Score", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10.5, frameon=True, loc='lower left')
    plt.tight_layout()

    out_path = os.path.join(save_dir, "exp2_jpeg_compression.jpg")
    plt.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved plot to {out_path}")

    return {"qualities": qualities, "cnn_aucs": cnn_aucs, "hybrid_aucs": hybrid_aucs}


# =====================================================================
# Experiment 4: ROI Bounding Box Perturbation Sensitivity
# =====================================================================
def run_experiment_4_roi_perturbation(orig_imgs_val, roi_boxes_val, y_val, cnn_model, hybrid_pipeline, extract_cnn_fn, extract_tda_fn, save_dir="images", dpi=300):
    print("\n[+] Running Experiment 4: ROI Bounding Box Sensitivity Suite...")

    perturbations = [0.0, 0.05, 0.10, 0.20]
    labels = ["0% (Exact)", "±5%", "±10%", "±20%"]
    cnn_aucs = []
    hybrid_aucs = []

    for p, label in zip(perturbations, labels):
        X_deg = apply_roi_perturbation(orig_imgs_val, roi_boxes_val, perturb_pct=p)

        X_cnn_deg = extract_cnn_fn(X_deg)
        X_tda_deg = extract_tda_fn(X_deg)
        X_hybrid_deg = np.hstack([X_tda_deg, X_cnn_deg])

        cnn_prob = cnn_model.predict_proba(X_cnn_deg)[:, 1]
        hybrid_prob = hybrid_pipeline.predict_proba(X_hybrid_deg)[:, 1]

        auc_cnn = roc_auc_score(y_val, cnn_prob)
        auc_hybrid = roc_auc_score(y_val, hybrid_prob)

        cnn_aucs.append(auc_cnn)
        hybrid_aucs.append(auc_hybrid)

        print(f"  ROI Perturbation: {label:10s} | CNN ROC-AUC: {auc_cnn:.4f} | Hybrid ROC-AUC: {auc_hybrid:.4f} | Delta: +{auc_hybrid - auc_cnn:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x_idx = np.arange(len(labels))
    ax.plot(x_idx, cnn_aucs, marker='o', linewidth=2.5, markersize=8, color='#d9534f', label='Standard CNN Baseline')
    ax.plot(x_idx, hybrid_aucs, marker='s', linewidth=2.5, markersize=8, color='#1f77b4', label='Hybrid CNN-TDA (Hardware Shield)')

    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_title("Exp 4: Segmentation Bounding Box Perturbation Sensitivity", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("ROI Bounding Box Shift/Scale Jitter", fontsize=11, fontweight='bold')
    ax.set_ylabel("ROC-AUC Score", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10.5, frameon=True, loc='lower left')
    plt.tight_layout()

    out_path = os.path.join(save_dir, "exp4_roi_sensitivity.jpg")
    plt.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved plot to {out_path}")

    return {"labels": labels, "cnn_aucs": cnn_aucs, "hybrid_aucs": hybrid_aucs}


# =====================================================================
# Experiment 9: Biomechanical Elastic Deformation
# =====================================================================
def run_experiment_9_elastic_deformation(X_val, y_val, cnn_model, hybrid_pipeline, extract_cnn_fn, extract_tda_fn, save_dir="images", dpi=300):
    print("\n[+] Running Experiment 9: Biomechanical Elastic Deformation Suite...")

    alphas = [0, 5, 10, 20]
    labels = ["Alpha 0 (Rigid)", "Alpha 5", "Alpha 10", "Alpha 20"]
    cnn_aucs = []
    hybrid_aucs = []

    for a, label in zip(alphas, labels):
        X_deg = apply_elastic_deformation(X_val, alpha=a, sigma=3.0)

        X_cnn_deg = extract_cnn_fn(X_deg)
        X_tda_deg = extract_tda_fn(X_deg)
        X_hybrid_deg = np.hstack([X_tda_deg, X_cnn_deg])

        cnn_prob = cnn_model.predict_proba(X_cnn_deg)[:, 1]
        hybrid_prob = hybrid_pipeline.predict_proba(X_hybrid_deg)[:, 1]

        auc_cnn = roc_auc_score(y_val, cnn_prob)
        auc_hybrid = roc_auc_score(y_val, hybrid_prob)

        cnn_aucs.append(auc_cnn)
        hybrid_aucs.append(auc_hybrid)

        print(f"  Elastic Deformation: {label:15s} | CNN ROC-AUC: {auc_cnn:.4f} | Hybrid ROC-AUC: {auc_hybrid:.4f} | Delta: +{auc_hybrid - auc_cnn:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x_idx = np.arange(len(labels))
    ax.plot(x_idx, cnn_aucs, marker='o', linewidth=2.5, markersize=8, color='#d9534f', label='Standard CNN Baseline')
    ax.plot(x_idx, hybrid_aucs, marker='s', linewidth=2.5, markersize=8, color='#1f77b4', label='Hybrid CNN-TDA (Hardware Shield)')

    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_title("Exp 9: Biomechanical Tissue Elastic Deformation Robustness", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Tissue Elastic Deformation Intensity (Alpha)", fontsize=11, fontweight='bold')
    ax.set_ylabel("ROC-AUC Score", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10.5, frameon=True, loc='lower left')
    plt.tight_layout()

    out_path = os.path.join(save_dir, "exp9_elastic_deformation.jpg")
    plt.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved plot to {out_path}")

    return {"labels": labels, "cnn_aucs": cnn_aucs, "hybrid_aucs": hybrid_aucs}
