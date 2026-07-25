#!/usr/bin/env python
"""
Topological Attention Gate & Split Conformal Prediction Pipeline
Author: Principal ML Researcher (Algebraic Topology & Computational Diagnostics)

This script implements:
1. Topological Attention Gate (Custom Scikit-Learn Transformer)
2. Split Conformal Prediction Wrapper around ExtraTreesClassifier for marginal coverage
3. Evaluation metrics (Marginal Coverage Rate and Average Set Size)
4. Publication-Ready Visualizations (300 DPI JPEGs in images/ folder)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import dataset loader and feature extraction functions from existing benchmark module
from tda_benchmark import load_dataset, extract_tda_features, PersistenceLandscape, PersistenceImage, extract_cnn_features, train_medical_weights

RANDOM_STATE = 42

# =====================================================================
# 1. Topological Attention Gate (Custom Scikit-Learn Transformer)
# =====================================================================
class TopologicalAttentionGate(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer implementing a Topological Attention Gate.
    Dynamically weights spatial CNN features based on topological persistence representations.
    
    Inputs: Concatenated array [v_tda, v_cnn] where first tda_dim columns are TDA features.
    Outputs: Concatenated array [v_tda, v_cnn_attended] where v_cnn_attended = v_cnn * sigmoid(v_tda @ W + b).
    """
    def __init__(self, tda_dim=5600, random_state=RANDOM_STATE):
        self.tda_dim = tda_dim
        self.random_state = random_state
        self.W_ = None
        self.b_ = None

    def fit(self, X, y=None):
        n_features = X.shape[1]
        cnn_dim = n_features - self.tda_dim
        if cnn_dim <= 0:
            raise ValueError(f"Total features ({n_features}) must exceed tda_dim ({self.tda_dim}).")
            
        rng = np.random.RandomState(self.random_state)
        # Xavier/Glorot initialization for smooth gradient scaling
        limit = np.sqrt(6.0 / (self.tda_dim + cnn_dim))
        self.W_ = rng.uniform(-limit, limit, size=(self.tda_dim, cnn_dim))
        self.b_ = np.zeros(cnn_dim)
        return self

    def transform(self, X):
        if self.W_ is None or self.b_ is None:
            raise ValueError("TopologicalAttentionGate transformer has not been fitted yet.")
            
        v_tda = X[:, :self.tda_dim]
        v_cnn = X[:, self.tda_dim:]
        
        logits = np.dot(v_tda, self.W_) + self.b_
        weights = expit(np.clip(logits, -500, 500))
        
        v_cnn_attended = v_cnn * weights
        return np.hstack([v_tda, v_cnn_attended])


# =====================================================================
# 2. Split Conformal Prediction Helper & Quantile Computation
# =====================================================================
def compute_conformal_quantile(cal_scores, alpha=0.05):
    """
    Computes the (1 - alpha)(1 + 1/n) quantile for split conformal prediction.
    Guarantees finite-sample marginal coverage >= 1 - alpha.
    """
    n_cal = len(cal_scores)
    k = int(np.ceil((n_cal + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n_cal)
    sorted_scores = np.sort(cal_scores)
    return sorted_scores[k - 1]


def run_split_conformal_fold(X_train_full, y_train_full, X_val, y_val, tda_dim, alpha=0.05, cal_size=0.2, random_state=RANDOM_STATE):
    """
    Executes Split Conformal Prediction on a single fold:
    1. Splits fold training data into proper_train and calibration sets (80/20).
    2. Fits StandardScaler, TopologicalAttentionGate, and ExtraTreesClassifier.
    3. Computes non-conformity scores on calibration set and calculates q_hat.
    4. Evaluates Prediction Sets on validation fold.
    """
    # Step A: Split training fold into proper_train and calibration sets
    X_pt, X_cal, y_pt, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=cal_size, stratify=y_train_full, random_state=random_state
    )
    
    # Step B: Build and fit pipeline components on proper_train
    scaler = StandardScaler()
    att_gate = TopologicalAttentionGate(tda_dim=tda_dim, random_state=random_state)
    clf = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=random_state, n_jobs=-1)
    
    X_pt_scaled = scaler.fit_transform(X_pt)
    X_pt_att = att_gate.fit_transform(X_pt_scaled)
    clf.fit(X_pt_att, y_pt)
    
    # Step C: Calibration Phase
    X_cal_scaled = scaler.transform(X_cal)
    X_cal_att = att_gate.transform(X_cal_scaled)
    cal_probs = clf.predict_proba(X_cal_att) # shape (n_cal, n_classes)
    
    # Non-conformity scores: 1.0 - prob[true_class]
    cal_scores = 1.0 - cal_probs[np.arange(len(y_cal)), y_cal]
    q_hat = compute_conformal_quantile(cal_scores, alpha=alpha)
    
    # Step D: Validation Phase
    X_val_scaled = scaler.transform(X_val)
    X_val_att = att_gate.transform(X_val_scaled)
    val_probs = clf.predict_proba(X_val_att)
    val_preds = clf.predict(X_val_att)
    
    n_classes = val_probs.shape[1]
    prediction_sets = []
    for i in range(len(y_val)):
        # Include all classes c where (1 - prob(c)) <= q_hat, i.e. prob(c) >= 1 - q_hat
        pred_set = [c for c in range(n_classes) if (1.0 - val_probs[i, c]) <= q_hat]
        if len(pred_set) == 0:
            pred_set = [np.argmax(val_probs[i])]
        prediction_sets.append(pred_set)
        
    raw_cnn_val = X_val_scaled[:, tda_dim:]
    attended_cnn_val = X_val_att[:, tda_dim:]
    
    return {
        "cal_scores": cal_scores,
        "q_hat": q_hat,
        "val_preds": val_preds,
        "val_probs": val_probs,
        "prediction_sets": prediction_sets,
        "raw_cnn_val": raw_cnn_val,
        "attended_cnn_val": attended_cnn_val
    }


# =====================================================================
# 3. Publication-Ready Visualizations (300 DPI JPEGs)
# =====================================================================
def generate_triage_figures(cal_scores, q_hat, set_sizes, raw_cnn, attended_cnn, save_dir="images", dpi=300):
    """
    Generates and saves 3 publication-ready figures as 300 DPI JPEGs in the save_dir folder:
    - Figure 1: Non-Conformity Score Distribution with q_hat threshold line
    - Figure 2: Clinical Triage Deferral Rate (Set Size = 1 vs Set Size = 2)
    - Figure 3: Topological Attention Impact (KDE comparison of raw vs attended features)
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("\nGenerating Publication-Ready Figures (300 DPI JPEG)...")
    
    # -----------------------------------------------------------------
    # Figure 1: Non-Conformity Score Distribution
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(cal_scores, kde=True, color="#1f77b4", bins=25, stat="density", alpha=0.5, ax=ax)
    ax.axvline(q_hat, color="#d9534f", linestyle="--", linewidth=2.5, 
               label=f"$\hat{{q}}$ Threshold = {q_hat:.4f} (Target $\\alpha=0.05$)")
    ax.set_title("Calibration Set Non-Conformity Score Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Non-Conformity Score $s_i = 1 - \hat{p}(y_i|x_i)$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc="upper left", fontsize=10.5, frameon=True)
    plt.tight_layout()
    fig1_path = os.path.join(save_dir, "non_conformity_distribution.jpg")
    plt.savefig(fig1_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Figure 1 saved: {fig1_path}")

    # -----------------------------------------------------------------
    # Figure 2: Clinical Triage Deferral Rate
    # -----------------------------------------------------------------
    set_size_series = pd.Series(set_sizes)
    size_counts = set_size_series.value_counts().sort_index()
    n_total = len(set_sizes)
    
    n_size_1 = size_counts.get(1, 0)
    n_size_2 = size_counts.get(2, 0)
    pct_1 = (n_size_1 / n_total) * 100.0
    pct_2 = (n_size_2 / n_total) * 100.0
    
    fig, ax = plt.subplots(figsize=(7.5, 5))
    categories = ["Autonomous Decision\n(Set Size = 1)", "Physician Deferral\n(Set Size = 2)"]
    percentages = [pct_1, pct_2]
    colors = ["#2e7d32", "#c62828"]
    
    bars = ax.bar(categories, percentages, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Percentage of Validation Cohort (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Clinical Triage Deferral Rate (Split Conformal Coverage)", fontsize=13, fontweight="bold", pad=12)
    
    for bar, count, pct in zip(bars, [n_size_1, n_size_2], percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + 2.5, 
                f"{pct:.1f}%\n(n={count})", ha='center', va='bottom', fontsize=10.5, fontweight='bold')
                
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, "clinical_triage_deferral.jpg")
    plt.savefig(fig2_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Figure 2 saved: {fig2_path}")

    # -----------------------------------------------------------------
    # Figure 3: Topological Attention Impact
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    raw_flat = raw_cnn.flatten()
    attended_flat = attended_cnn.flatten()
    
    # Subsample if large array to optimize KDE rendering speed
    if len(raw_flat) > 20000:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(raw_flat), 20000, replace=False)
        raw_flat = raw_flat[indices]
        attended_flat = attended_flat[indices]
        
    sns.kdeplot(raw_flat, label="Raw Spatial Features ($v_{cnn}$)", color="#1f77b4", linewidth=2.2, fill=True, alpha=0.25, ax=ax)
    sns.kdeplot(attended_flat, label="Topological Attended Features ($v_{cnn}^{att}$)", color="#ff7f0e", linewidth=2.2, fill=True, alpha=0.35, ax=ax)
    
    ax.set_title("Topological Attention Gating: Feature Variance Compression", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Spatial Feature Intensity Values", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc="upper right", fontsize=10.5, frameon=True)
    plt.tight_layout()
    fig3_path = os.path.join(save_dir, "topological_attention_impact.jpg")
    plt.savefig(fig3_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Figure 3 saved: {fig3_path}")


# =====================================================================
# 4. Cross-Validation Execution Loop
# =====================================================================
def main():
    print("=" * 70)
    print("TOPOLOGICAL ATTENTION & SPLIT CONFORMAL PREDICTION EVALUATION")
    print("=" * 70)
    
    # 1. Load Dataset
    X_imgs, y, groups = load_dataset("data", img_size=(64, 64))
    
    # 2. Extract Topological Diagrams & Vectorize TDA Features
    X_diags = extract_tda_features(X_imgs)
    
    print("\nComputing TDA Representations (Landscapes + Persistence Images)...")
    pl = PersistenceLandscape(n_layers=3, n_values=100, n_jobs=-1)
    X_pl_flat = pl.fit_transform(X_diags).reshape(len(X_diags), -1)
    
    pi = PersistenceImage(sigma=0.1, n_bins=50)
    X_pi_flat = pi.fit_transform(X_diags)
    
    X_tda = np.hstack([X_pl_flat, X_pi_flat])
    tda_dim = X_tda.shape[1]
    print(f"  [+] Total TDA Vector Dimension (tda_dim): {tda_dim}")
    
    # 3. Perform 5-Fold StratifiedGroupKFold Cross-Validation
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    coverages = []
    set_sizes_all = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []
    
    all_cal_scores = []
    last_fold_data = None
    
    print("\nInitiating 5-Fold StratifiedGroupKFold Conformal Benchmark...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_tda, y, groups=groups)):
        fold_t0 = time.time()
        print(f"\nEvaluating Fold {fold+1}/5...", flush=True)
        
        # Split TDA and images for current fold
        X_tda_train, X_tda_val = X_tda[train_idx], X_tda[val_idx]
        X_imgs_train, X_imgs_val = X_imgs[train_idx], X_imgs[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Extract Medical Domain-Adapted Spatial Features (ResNet18 Autoencoder)
        resnet_encoder = train_medical_weights(X_imgs_train, "resnet18", epochs=5)
        X_cnn_train = extract_cnn_features(X_imgs_train, resnet_encoder)
        X_cnn_val = extract_cnn_features(X_imgs_val, resnet_encoder)
        
        # Format hybrid matrix: [v_tda, v_cnn] (First tda_dim columns are TDA)
        X_hybrid_train = np.hstack([X_tda_train, X_cnn_train])
        X_hybrid_val = np.hstack([X_tda_val, X_cnn_val])
        
        # Run Split Conformal Prediction on fold
        fold_res = run_split_conformal_fold(
            X_hybrid_train, y_train, X_hybrid_val, y_val, tda_dim=tda_dim, alpha=0.05, cal_size=0.2, random_state=RANDOM_STATE + fold
        )
        
        # Calculate Conformal Metrics
        pred_sets = fold_res["prediction_sets"]
        cov = np.mean([y_val[i] in pred_sets[i] for i in range(len(y_val))])
        avg_sz = np.mean([len(s) for s in pred_sets])
        
        # Calculate Classification Metrics
        val_preds = fold_res["val_preds"]
        val_probs = fold_res["val_probs"][:, 1]
        
        acc = accuracy_score(y_val, val_preds)
        prec = precision_score(y_val, val_preds, zero_division=0)
        rec = recall_score(y_val, val_preds, zero_division=0)
        f1 = f1_score(y_val, val_preds, zero_division=0)
        auc = roc_auc_score(y_val, val_probs)
        
        coverages.append(cov)
        set_sizes_all.extend([len(s) for s in pred_sets])
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        aucs.append(auc)
        
        all_cal_scores.extend(fold_res["cal_scores"])
        last_fold_data = fold_res
        
        print(f"  [Fold {fold+1}/5] Coverage: {cov*100:.2f}% | Avg Set Size: {avg_sz:.4f} | Accuracy: {acc:.4f} | AUC: {auc:.4f} ({time.time()-fold_t0:.2f}s)")

    # 4. Aggregate & Print Summary Metrics
    print("\n" + "=" * 70)
    print("CONFORMAL PREDICTION & TOPOLOGICAL ATTENTION SUMMARY METRICS")
    print("=" * 70)
    print(f"  Marginal Coverage Rate (Target 95.0%): {np.mean(coverages)*100:.2f}% ± {np.std(coverages)*100:.2f}%")
    print(f"  Average Set Size:                      {np.mean(set_sizes_all):.4f} ± {np.std([np.mean(s) for s in set_sizes_all]):.4f}")
    print(f"  Accuracy:                              {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Precision:                             {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"  Recall (Sensitivity):                  {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"  F1-Score:                              {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  ROC-AUC:                               {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print("=" * 70)

    # 5. Generate Visualizations
    generate_triage_figures(
        cal_scores=np.array(all_cal_scores),
        q_hat=last_fold_data["q_hat"],
        set_sizes=set_sizes_all,
        raw_cnn=last_fold_data["raw_cnn_val"],
        attended_cnn=last_fold_data["attended_cnn_val"],
        save_dir="images",
        dpi=300
    )


if __name__ == "__main__":
    main()
