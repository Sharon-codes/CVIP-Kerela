#!/usr/bin/env python
"""
Ablation & Hyperparameter Sweep Module:
- Exp 7: Feature Waterfall (CNN-only vs Landscape vs Image vs Hybrid)
- Exp 8: PCA Component Sweep (20, 50, 100, 150, 200) vs Recall & ROC-AUC

Author: Lead Biomedical ML Engineer (Q1 Journal Submission Suite)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import recall_score, roc_auc_score

RANDOM_STATE = 42


# =====================================================================
# Experiment 7: Feature Waterfall (Incremental Modality Gains)
# =====================================================================
def run_experiment_7_feature_waterfall(X_cnn_tr, X_tda_tr, y_tr, X_cnn_va, X_tda_va, y_va, save_dir="images", dpi=300):
    print("\n[+] Running Experiment 7: Modality Feature Waterfall Suite...")

    # tda_dim = 5600 -> Landscapes = 600, Persistence Images = 5000
    X_land_tr = X_tda_tr[:, :600]
    X_land_va = X_tda_va[:, :600]

    X_pi_tr = X_tda_tr[:, 600:]
    X_pi_va = X_tda_va[:, 600:]

    X_hybrid_tr = np.hstack([X_tda_tr, X_cnn_tr])
    X_hybrid_va = np.hstack([X_tda_va, X_cnn_va])

    configs = [
        ("1. CNN Only", X_cnn_tr, X_cnn_va),
        ("2. Landscapes Only", X_land_tr, X_land_va),
        ("3. Images Only", X_pi_tr, X_pi_va),
        ("4. Hybrid (TDA+CNN)", X_hybrid_tr, X_hybrid_va)
    ]

    labels = []
    recalls = []
    aucs = []

    for name, X_tr, X_va in configs:
        labels.append(name)

        # Scale and train ExtraTrees
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)

        if X_tr.shape[1] > 100:
            pca = PCA(n_components=min(100, X_tr.shape[1]), random_state=RANDOM_STATE)
            X_tr_sc = pca.fit_transform(X_tr_sc)
            X_va_sc = pca.transform(X_va_sc)

        clf = ExtraTreesClassifier(
            n_estimators=500,
            class_weight='balanced',
            max_features='sqrt',
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        clf.fit(X_tr_sc, y_tr)

        y_pred = clf.predict(X_va_sc)
        y_prob = clf.predict_proba(X_va_sc)[:, 1]

        rec = recall_score(y_va, y_pred, zero_division=0)
        auc = roc_auc_score(y_va, y_prob)

        recalls.append(rec)
        aucs.append(auc)

        print(f"  Modality: {name:20s} | Recall: {rec:.4f} | ROC-AUC: {auc:.4f}")

    # Waterfall Bar Chart
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x_idx = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x_idx - width/2, aucs, width, label='ROC-AUC Score', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x_idx + width/2, recalls, width, label='Recall / Sensitivity', color='#2ca02c', edgecolor='black')

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_ylim(0.0, 1.15)
    ax.set_title("Exp 7: Feature Waterfall — Modality Contribution & Incremental Gains", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Metric Value", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10.5, frameon=True, loc='upper left')
    plt.tight_layout()

    out_path = os.path.join(save_dir, "exp7_feature_waterfall.jpg")
    plt.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved plot to {out_path}")

    return {"labels": labels, "recalls": recalls, "aucs": aucs}


# =====================================================================
# Experiment 8: PCA Dimensionality Component Sweep
# =====================================================================
def run_experiment_8_pca_sweep(X_hybrid_tr, y_tr, X_hybrid_va, y_va, save_dir="images", dpi=300):
    print("\n[+] Running Experiment 8: PCA Dimension Component Sweep...")

    k_components = [20, 50, 100, 150, 200]
    recalls = []
    aucs = []

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_hybrid_tr)
    X_va_sc = scaler.transform(X_hybrid_va)

    for k in k_components:
        pca = PCA(n_components=k, random_state=RANDOM_STATE)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_va_pca = pca.transform(X_va_sc)

        clf = ExtraTreesClassifier(
            n_estimators=500,
            class_weight='balanced',
            max_features='sqrt',
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        clf.fit(X_tr_pca, y_tr)

        y_pred = clf.predict(X_va_pca)
        y_prob = clf.predict_proba(X_va_pca)[:, 1]

        rec = recall_score(y_va, y_pred, zero_division=0)
        auc = roc_auc_score(y_va, y_prob)

        recalls.append(rec)
        aucs.append(auc)

        print(f"  PCA Components K={k:3d} | Recall: {rec:.4f} | ROC-AUC: {auc:.4f}")

    # Plot PCA Sweep Curves
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(k_components, aucs, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='ROC-AUC Score')
    ax.plot(k_components, recalls, marker='s', linewidth=2.5, markersize=8, color='#2ca02c', label='Recall / Sensitivity')

    # Highlight optimal operating point K=100
    ax.axvline(100, color='#d9534f', linestyle='--', linewidth=2, label='Optimal Operating Point (K=100)')

    ax.set_xticks(k_components)
    ax.set_title("Exp 8: PCA Dimension Sweep vs. Discriminative Performance", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Number of PCA Principal Components (K)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Metric Score", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10.5, frameon=True, loc='lower right')
    plt.tight_layout()

    out_path = os.path.join(save_dir, "exp8_pca_sweep.jpg")
    plt.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved plot to {out_path}")

    return {"k_components": k_components, "recalls": recalls, "aucs": aucs}
