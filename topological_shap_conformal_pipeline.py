#!/usr/bin/env python
"""
Dynamic Early-Exit, Calibrated Split Conformal Prediction & Topological SHAP Pipeline
Author: Principal ML Researcher (Algebraic Topology & Computational Diagnostics)

This script implements:
1. Dynamic Early-Exit cascading inference logic (Fast Path CNN-only vs Complex Path Hybrid).
2. Split Conformal Prediction with isotonic Calibration using FrozenEstimator.
3. Topological SHAP Integration using TreeExplainer.
4. Publication-Ready 300 DPI Visualizations:
   - Figure 1: Conformal Calibration Score Distribution.
   - Figure 2: Clinical Triage Distribution Pie Chart.
   - Figure 3: SHAP Summary Plot.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import dataset loader and feature extraction functions
from tda_benchmark import load_dataset, extract_tda_features, PersistenceLandscape, PersistenceImage, extract_cnn_features, train_medical_weights

RANDOM_STATE = 42
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# 1. Cascading Early-Exit & Conformal Prediction Loop
# =====================================================================
def run_early_exit_conformal_fold(X_hybrid_train, y_train, X_hybrid_val, y_val, tda_dim=5600, tau=0.95, alpha=0.05, cal_size=0.2, random_state=RANDOM_STATE):
    """
    Evaluates a single fold using Dynamic Early-Exit and Split Conformal Prediction.
    """
    # Split training fold into proper_train (80%) and calibration (20%)
    X_pt, X_cal, y_pt, y_cal = train_test_split(
        X_hybrid_train, y_train, test_size=cal_size, stratify=y_train, random_state=random_state
    )
    
    # StandardScaler fit on proper_train (full features)
    scaler_full = StandardScaler()
    X_pt_scaled = scaler_full.fit_transform(X_pt)
    X_cal_scaled = scaler_full.transform(X_cal)
    X_val_scaled = scaler_full.transform(X_hybrid_val)
    
    # A. Fast Path: Logistic Regression on CNN features only
    scaler_cnn = StandardScaler()
    X_pt_cnn = scaler_cnn.fit_transform(X_pt[:, tda_dim:])
    
    fast_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
    fast_model.fit(X_pt_cnn, y_pt)
    
    # B. Complex Path: Extra Trees on Hybrid features
    base_complex_model = ExtraTreesClassifier(n_estimators=500, class_weight='balanced', random_state=random_state, n_jobs=-1)
    base_complex_model.fit(X_pt_scaled, y_pt)
    
    # Wrap in CalibratedClassifierCV with isotonic calibration pre-fit on calibration set using FrozenEstimator
    calibrated_complex_model = CalibratedClassifierCV(estimator=FrozenEstimator(base_complex_model), method='isotonic')
    calibrated_complex_model.fit(X_cal_scaled, y_cal)
    
    # Compute non-conformity scores on calibration set using the calibrated complex model
    cal_probs = calibrated_complex_model.predict_proba(X_cal_scaled)
    cal_scores = 1.0 - cal_probs[np.arange(len(y_cal)), y_cal]
    
    # Calculate q_hat for conformal prediction (95% coverage)
    n_cal = len(cal_scores)
    k = int(np.ceil((n_cal + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n_cal)
    q_hat = np.sort(cal_scores)[k - 1]
    
    # C. Inference on Validation Set
    X_val_cnn = scaler_cnn.transform(X_hybrid_val[:, tda_dim:])
    
    fast_probs = fast_model.predict_proba(X_val_cnn)
    max_fast_conf = np.max(fast_probs, axis=1)
    
    # Route validation samples
    fast_mask = (max_fast_conf >= tau)
    complex_mask = ~fast_mask
    
    validation_routing = []
    final_probs = np.zeros((len(y_val), 2))
    final_preds = np.zeros(len(y_val), dtype=int)
    prediction_sets = []
    
    # Complex Path Evaluation
    if np.any(complex_mask):
        X_val_complex_scaled = X_val_scaled[complex_mask]
        complex_probs = calibrated_complex_model.predict_proba(X_val_complex_scaled)
        
    complex_idx = 0
    for i in range(len(y_val)):
        if fast_mask[i]:
            validation_routing.append("Fast Path")
            final_probs[i] = fast_probs[i]
            final_preds[i] = np.argmax(fast_probs[i])
            prediction_sets.append([final_preds[i]]) # Singleton prediction set
        else:
            validation_routing.append("Complex Path")
            p_comp = complex_probs[complex_idx]
            final_probs[i] = p_comp
            final_preds[i] = np.argmax(p_comp)
            
            # Include all classes where (1 - prob) <= q_hat
            p_sets = [c for c in range(2) if (1.0 - p_comp[c]) <= q_hat]
            if len(p_sets) == 0:
                p_sets = [final_preds[i]]
            prediction_sets.append(p_sets)
            complex_idx += 1
            
    return {
        "fast_mask": fast_mask,
        "complex_mask": complex_mask,
        "routing": validation_routing,
        "final_probs": final_probs,
        "final_preds": final_preds,
        "prediction_sets": prediction_sets,
        "cal_scores": cal_scores,
        "q_hat": q_hat,
        "base_complex_model": base_complex_model,
        "X_val_scaled": X_val_scaled,
        "y_val": y_val
    }


# =====================================================================
# 2. Publication-Ready Visualizations (300 DPI)
# =====================================================================
def generate_upgraded_figures(cal_scores, q_hat, triage_counts, shap_values, X_sample, tda_dim=5600, save_dir="images", dpi=300):
    """
    Generates and saves 3 publication-ready figures as 300 DPI JPEGs in the save_dir folder.
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("\nGenerating Publication-Ready Figures (300 DPI JPEG)...")
    
    # -----------------------------------------------------------------
    # Figure 1: Conformal Calibration Score Distribution
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(cal_scores, kde=True, color="#1f77b4", bins=25, stat="density", alpha=0.5, ax=ax)
    ax.axvline(q_hat, color="#d9534f", linestyle="--", linewidth=2.5, 
               label=f"q_hat Threshold = {q_hat:.4f} (Target Coverage 95%)")
    ax.set_title("Isotonic-Calibrated Non-Conformity Scores", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Calibration Score s_i = 1 - p(y_i|x_i)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc="upper left", fontsize=10.5, frameon=True)
    plt.tight_layout()
    fig1_path = os.path.join(save_dir, "conformal_calibration.jpg")
    plt.savefig(fig1_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Figure 1 saved: {fig1_path}")

    # -----------------------------------------------------------------
    # Figure 2: Clinical Triage Distribution Pie Chart
    # -----------------------------------------------------------------
    labels = list(triage_counts.keys())
    sizes = list(triage_counts.values())
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
           textprops={'fontsize': 11, 'fontweight': 'bold'}, 
           wedgeprops={'edgecolor': 'black', 'linewidth': 1.2, 'antialiased': True})
    ax.set_title("Clinical Triage & Cascade Inference Distribution", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, "clinical_triage_distribution.jpg")
    plt.savefig(fig2_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Figure 2 saved: {fig2_path}")

    # -----------------------------------------------------------------
    # Figure 3: SHAP Feature Importance Summary Plot
    # -----------------------------------------------------------------
    feature_names = [f"TDA Component {i}" if i < tda_dim else f"CNN Component {i - tda_dim}" for i in range(X_sample.shape[1])]
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", max_display=20, show=False)
    plt.title("Top 20 Feature Importance (Topological vs Spatial Blocks)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    
    fig3_path = os.path.join(save_dir, "shap_summary_plot.jpg")
    plt.savefig(fig3_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Figure 3 saved: {fig3_path}")


# =====================================================================
# 3. Cross-Validation Execution Loop
# =====================================================================
def main():
    print("=" * 80)
    print("DYNAMIC EARLY-EXIT, CALIBRATED CONFORMAL PREDICTION & SHAP STUDY")
    print("=" * 80)
    
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
    
    early_exit_pcts = []
    cpu_time_saved_pcts = []
    
    complex_coverages = []
    complex_set_sizes = []
    
    overall_accuracies = []
    overall_aucs = []
    
    all_cal_scores = []
    last_fold_data = None
    
    triage_counts = {
        "Fast Path (CNN Only)": 0,
        "Complex Path - Confident (Set Size 1)": 0,
        "Complex Path - Deferred to Physician (Set Size 2)": 0
    }
    
    print("\nInitiating 5-Fold Cross-Validation Loop...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_tda, y, groups=groups)):
        fold_t0 = time.time()
        print(f"\nProcessing Fold {fold+1}/5...", flush=True)
        
        # Split TDA and images for current fold
        X_tda_train, X_tda_val = X_tda[train_idx], X_tda[val_idx]
        X_imgs_train, X_imgs_val = X_imgs[train_idx], X_imgs[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Medical Spatial Feature Extraction (ResNet18 Autoencoder)
        resnet_encoder = train_medical_weights(X_imgs_train, "resnet18", epochs=3)
        X_cnn_train = extract_cnn_features(X_imgs_train, resnet_encoder)
        X_cnn_val = extract_cnn_features(X_imgs_val, resnet_encoder)
        
        X_hybrid_train = np.hstack([X_tda_train, X_cnn_train])
        X_hybrid_val = np.hstack([X_tda_val, X_cnn_val])
        
        # Evaluate early-exit + conformal fold
        fold_res = run_early_exit_conformal_fold(
            X_hybrid_train=X_hybrid_train,
            y_train=y_train,
            X_hybrid_val=X_hybrid_val,
            y_val=y_val,
            tda_dim=tda_dim,
            tau=0.95,
            alpha=0.05,
            cal_size=0.2,
            random_state=RANDOM_STATE + fold
        )
        
        # Calculate routing metrics
        n_val = len(y_val)
        n_fast = np.sum(fold_res["fast_mask"])
        n_complex = np.sum(fold_res["complex_mask"])
        pct_fast = (n_fast / n_val) * 100.0
        early_exit_pcts.append(pct_fast)
        
        # Compute CPU time saved (Fast Path = 1ms, Complex Path = 26ms)
        baseline_time = 26.0 * n_val
        cascade_time = (1.0 * n_fast) + (26.0 * n_complex)
        time_saved_pct = ((baseline_time - cascade_time) / baseline_time) * 100.0
        cpu_time_saved_pcts.append(time_saved_pct)
        
        # Conformal metrics on Complex Path
        pred_sets = fold_res["prediction_sets"]
        complex_mask = fold_res["complex_mask"]
        
        complex_sets = [pred_sets[i] for i in range(n_val) if complex_mask[i]]
        y_val_complex = y_val[complex_mask]
        
        if len(complex_sets) > 0:
            cov = np.mean([y_val_complex[i] in complex_sets[i] for i in range(len(complex_sets))])
            avg_sz = np.mean([len(s) for s in complex_sets])
            complex_coverages.append(cov)
            complex_set_sizes.append(avg_sz)
        else:
            cov = 1.0
            avg_sz = 1.0
            
        # Update Triage distribution counts
        for i in range(n_val):
            if fold_res["fast_mask"][i]:
                triage_counts["Fast Path (CNN Only)"] += 1
            else:
                sz = len(pred_sets[i])
                if sz == 1:
                    triage_counts["Complex Path - Confident (Set Size 1)"] += 1
                else:
                    triage_counts["Complex Path - Deferred to Physician (Set Size 2)"] += 1
                    
        # Overall Classification metrics
        acc = accuracy_score(y_val, fold_res["final_preds"])
        auc = roc_auc_score(y_val, fold_res["final_probs"][:, 1])
        
        overall_accuracies.append(acc)
        overall_aucs.append(auc)
        
        all_cal_scores.extend(fold_res["cal_scores"])
        last_fold_data = fold_res
        
        print(f"  [Fold {fold+1}/5] Early-Exit: {pct_fast:.1f}% | CPU Time Saved: {time_saved_pct:.1f}% | "
              f"Complex Coverage: {cov*100:.1f}% | Complex Set Size: {avg_sz:.4f} | "
              f"Overall Accuracy: {acc:.4f} | Overall AUC: {auc:.4f} ({time.time()-fold_t0:.2f}s)", flush=True)

    # Print summary performance metrics
    print("\n" + "=" * 80)
    print("DYNAMIC CASCADE & CALIBRATED CONFORMAL SUMMARY STATS")
    print("=" * 80)
    print(f"  Early-Exit Rate:         {np.mean(early_exit_pcts):.2f}% ± {np.std(early_exit_pcts):.2f}%")
    print(f"  Estimated CPU Time Saved: {np.mean(cpu_time_saved_pcts):.2f}% ± {np.std(cpu_time_saved_pcts):.2f}%")
    print(f"  Complex Path Coverage:   {np.mean(complex_coverages)*100:.2f}% ± {np.std(complex_coverages)*100:.2f}%")
    print(f"  Complex Path Set Size:   {np.mean(complex_set_sizes):.4f} ± {np.std(complex_set_sizes):.4f}")
    print(f"  Overall System Accuracy: {np.mean(overall_accuracies):.4f} ± {np.std(overall_accuracies):.4f}")
    print(f"  Overall System ROC-AUC:  {np.mean(overall_aucs):.4f} ± {np.std(overall_aucs):.4f}")
    print("=" * 80)

    # 4. Topological SHAP Integration on Representative Fold (Last Fold)
    print("\nComputing Topological SHAP Values on validation subset...")
    base_model = last_fold_data["base_complex_model"]
    X_val_scaled = last_fold_data["X_val_scaled"]
    
    # Use validation samples for SHAP values
    shap_sample_idx = np.random.choice(len(X_val_scaled), min(100, len(X_val_scaled)), replace=False)
    X_sample = X_val_scaled[shap_sample_idx]
    
    explainer = shap.TreeExplainer(base_model)
    shap_values_raw = explainer.shap_values(X_sample)
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
    else:
        if len(shap_values_raw.shape) == 3:
            shap_values = shap_values_raw[:, :, 1]
        else:
            shap_values = shap_values_raw
            
    # 5. Generate Visualizations
    generate_upgraded_figures(
        cal_scores=np.array(all_cal_scores),
        q_hat=last_fold_data["q_hat"],
        triage_counts=triage_counts,
        shap_values=shap_values,
        X_sample=X_sample,
        tda_dim=tda_dim,
        save_dir="images",
        dpi=300
    )


if __name__ == "__main__":
    main()
