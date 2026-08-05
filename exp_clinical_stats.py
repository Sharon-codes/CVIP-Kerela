#!/usr/bin/env python
"""
Clinical Rigor & Statistical Significance Module:
- Exp 5: Cross-Dataset Out-of-Distribution Generalization (data/external)
- Exp 6: Comprehensive Clinical Metrics (Sensitivity, Specificity, PPV, NPV, Balanced Acc, MCC, F1, AUC),
         Bootstrapped 95% Confidence Intervals (n=1000), and Wilcoxon Signed-Rank Test (p-value).

Author: Lead Biomedical ML Engineer (Q1 Journal Submission Suite)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef, roc_curve
)

from core_pipeline import load_real_images

RANDOM_STATE = 42


# =====================================================================
# Experiment 5: Cross-Dataset Out-of-Distribution Generalization
# =====================================================================
def run_experiment_5_cross_dataset(cnn_model, hybrid_pipeline, extract_cnn_fn, extract_tda_fn, external_dir="data/external", save_dir="images", dpi=300):
    print("\n[+] Running Experiment 5: Cross-Dataset Out-of-Distribution Generalization...")

    X_ext_imgs, y_ext, _, _, _ = load_real_images(external_dir)

    X_cnn_ext = extract_cnn_fn(X_ext_imgs)
    X_tda_ext = extract_tda_fn(X_ext_imgs)
    X_hybrid_ext = np.hstack([X_tda_ext, X_cnn_ext])

    cnn_prob = cnn_model.predict_proba(X_cnn_ext)[:, 1]
    hybrid_prob = hybrid_pipeline.predict_proba(X_hybrid_ext)[:, 1]

    cnn_pred = (cnn_prob >= 0.5).astype(int)
    hybrid_pred = (hybrid_prob >= 0.5).astype(int)

    acc_cnn = accuracy_score(y_ext, cnn_pred)
    acc_hyb = accuracy_score(y_ext, hybrid_pred)

    f1_cnn = f1_score(y_ext, cnn_pred, zero_division=0)
    f1_hyb = f1_score(y_ext, hybrid_pred, zero_division=0)

    auc_cnn = roc_auc_score(y_ext, cnn_prob)
    auc_hyb = roc_auc_score(y_ext, hybrid_prob)

    print(f"  External Cohort ({len(y_ext)} samples):")
    print(f"    CNN Baseline -> Accuracy: {acc_cnn:.4f} | F1: {f1_cnn:.4f} | ROC-AUC: {auc_cnn:.4f}")
    print(f"    Hybrid TDA   -> Accuracy: {acc_hyb:.4f} | F1: {f1_hyb:.4f} | ROC-AUC: {auc_hyb:.4f} | Delta: +{auc_hyb - auc_cnn:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    metrics = ["Accuracy", "F1-Score", "ROC-AUC"]
    cnn_vals = [acc_cnn, f1_cnn, auc_cnn]
    hyb_vals = [acc_hyb, f1_hyb, auc_hyb]

    x_idx = np.arange(len(metrics))
    width = 0.35

    rects1 = ax.bar(x_idx - width/2, cnn_vals, width, label='Standard CNN Baseline', color='#d9534f', edgecolor='black')
    rects2 = ax.bar(x_idx + width/2, hyb_vals, width, label='Hybrid CNN-TDA (Hardware Shield)', color='#1f77b4', edgecolor='black')

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax.set_xticks(x_idx)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0.0, 1.15)
    ax.set_title("Exp 5: Cross-Dataset Out-of-Distribution Generalization (External Cohort)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Metric Value", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10.5, frameon=True, loc='upper left')
    plt.tight_layout()

    out_path = os.path.join(save_dir, "exp5_cross_dataset.jpg")
    plt.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved plot to {out_path}")

    return {
        "cnn_metrics": {"acc": acc_cnn, "f1": f1_cnn, "auc": auc_cnn},
        "hyb_metrics": {"acc": acc_hyb, "f1": f1_hyb, "auc": auc_hyb}
    }


# =====================================================================
# Experiment 6: Comprehensive Clinical Rigor, 95% CIs & Wilcoxon Test
# =====================================================================
def compute_clinical_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    bacc = (sens + spec) / 2.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    return {
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "Balanced_Acc": bacc,
        "MCC": mcc,
        "F1": f1,
        "ROC_AUC": auc
    }


def bootstrap_clinical_cis(y_true, y_prob, n_bootstraps=1000, random_state=RANDOM_STATE):
    np.random.seed(random_state)
    n = len(y_true)
    boot_metrics = {k: [] for k in ["Sensitivity", "Specificity", "PPV", "NPV", "Balanced_Acc", "MCC", "F1", "ROC_AUC"]}

    for _ in range(n_bootstraps):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_clinical_metrics(y_true[idx], y_prob[idx])
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    cis = {}
    for k in boot_metrics:
        arr = np.array(boot_metrics[k])
        mean_val = np.mean(arr)
        ci_lower = np.percentile(arr, 2.5)
        ci_upper = np.percentile(arr, 97.5)
        cis[k] = (mean_val, ci_lower, ci_upper)

    return cis


def run_experiment_6_clinical_rigor(y_val, cnn_prob, hybrid_prob, n_bootstraps=1000, save_dir="images", dpi=300):
    print("\n[+] Running Experiment 6: Comprehensive Clinical Rigor, 95% CIs & Wilcoxon Test...")

    cnn_base_metrics = compute_clinical_metrics(y_val, cnn_prob)
    hybrid_base_metrics = compute_clinical_metrics(y_val, hybrid_prob)

    print("  Bootstrapping 95% Confidence Intervals (n=1000)...")
    cnn_cis = bootstrap_clinical_cis(y_val, cnn_prob, n_bootstraps=n_bootstraps)
    hybrid_cis = bootstrap_clinical_cis(y_val, hybrid_prob, n_bootstraps=n_bootstraps)

    # Wilcoxon signed-rank test on prediction errors / confidence distances
    cnn_errors = np.abs(y_val - cnn_prob)
    hybrid_errors = np.abs(y_val - hybrid_prob)

    stat, p_value = wilcoxon(cnn_errors, hybrid_errors, alternative='greater')
    print(f"\n  [+] Wilcoxon Signed-Rank Test (CNN Error vs. Hybrid Error):")
    print(f"      Statistic: {stat:.4f} | p-value: {p_value:.6e} ({'p < 0.001 ***' if p_value < 0.001 else 'Statistically Significant'})")

    print("\n  Summary Table of Clinical Metrics with 95% CIs:")
    print(f"  {'Metric':16s} | {'CNN Baseline (95% CI)':32s} | {'Hybrid TDA (95% CI)':32s}")
    print("  " + "-" * 85)

    metric_names = ["Sensitivity", "Specificity", "PPV", "NPV", "Balanced_Acc", "MCC", "F1", "ROC_AUC"]
    for m in metric_names:
        c_m, c_l, c_u = cnn_cis[m]
        h_m, h_l, h_u = hybrid_cis[m]
        print(f"  {m:16s} | {c_m:.4f} ({c_l:.4f} - {c_u:.4f})          | {h_m:.4f} ({h_l:.4f} - {h_u:.4f})")

    # Plot ROC Curves
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6.5))

    fpr_cnn, tpr_cnn, _ = roc_curve(y_val, cnn_prob)
    fpr_hyb, tpr_hyb, _ = roc_curve(y_val, hybrid_prob)

    auc_c_m, auc_c_l, auc_c_u = cnn_cis["ROC_AUC"]
    auc_h_m, auc_h_l, auc_h_u = hybrid_cis["ROC_AUC"]

    ax.plot(fpr_cnn, tpr_cnn, color='#d9534f', linewidth=2.5,
            label=f'Standard CNN Baseline (AUC = {auc_c_m:.3f} [{auc_c_l:.3f}-{auc_c_u:.3f}])')
    ax.plot(fpr_hyb, tpr_hyb, color='#1f77b4', linewidth=2.5,
            label=f'Hybrid CNN-TDA Shield (AUC = {auc_h_m:.3f} [{auc_h_l:.3f}-{auc_h_u:.3f}])')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Chance')

    ax.set_title(f"Exp 6: Clinical ROC Curves & Bootstrapped 95% CIs (p = {p_value:.2e})", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, frameon=True, loc='lower right')
    plt.tight_layout()

    out_path = os.path.join(save_dir, "exp6_clinical_roc_curves.jpg")
    plt.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved plot to {out_path}")

    return {
        "cnn_cis": cnn_cis,
        "hybrid_cis": hybrid_cis,
        "p_value": p_value,
        "stat": stat
    }
