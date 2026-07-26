#!/usr/bin/env python
"""
Topological SHAP Baseline Pipeline
Author: Principal ML Researcher (Algebraic Topology & Computational Diagnostics)

This script implements:
1. Concatenation of topological features (v_tda) and spatial CNN features (v_cnn).
2. Dimensionality reduction using PCA(n_components=100).
3. Classification via ExtraTreesClassifier(n_estimators=500).
4. StratifiedGroupKFold 5-Fold Cross Validation.
5. SHAP explainability on the best performing fold's PCA features.
6. Generation of Figure 1 (Beeswarm) and Figure 2 (Bar plot) at 300 DPI.
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

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import dataset loader and feature extraction functions
from tda_benchmark import load_dataset, extract_tda_features, PersistenceLandscape, PersistenceImage, extract_cnn_features, train_medical_weights

RANDOM_STATE = 42
warnings.filterwarnings("ignore", category=UserWarning)

def generate_shap_plots(shap_values, X_val_pca, feature_names, save_dir="images", dpi=300):
    """
    Generates and saves two high-resolution 300 DPI JPEGs:
    - Figure 1: SHAP Beeswarm Summary Plot
    - Figure 2: SHAP Feature Importance Bar Plot
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("\nGenerating Publication-Ready Figures (300 DPI JPEG)...")
    
    # Figure 1: SHAP Summary Beeswarm Plot
    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_val_pca, plot_type="dot", feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm Summary Plot (Top PCA Components)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig1_path = os.path.join(save_dir, "shap_beeswarm_summary.jpg")
    plt.savefig(fig1_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Figure 1 saved: {fig1_path}")
    
    # Figure 2: SHAP Feature Importance Bar Plot
    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_val_pca, plot_type="bar", feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance Bar Plot (Top PCA Components)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, "shap_importance_bar.jpg")
    plt.savefig(fig2_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Figure 2 saved: {fig2_path}")

def main():
    print("=" * 80)
    print("TOPOLOGICAL PCA BASELINE PIPELINE & SHAP EXPLAINABILITY")
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
    
    results = {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []}
    fold_data = []
    
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
        
        # Scale hybrid vectors
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_hybrid_train)
        X_val_scaled = scaler.transform(X_hybrid_val)
        
        # PCA Dimensionality Reduction down to 100 components
        pca = PCA(n_components=100, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        
        # Extra Trees Classifier
        clf = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        clf.fit(X_train_pca, y_train)
        
        y_pred = clf.predict(X_val_pca)
        y_prob = clf.predict_proba(X_val_pca)[:, 1]
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob)
        
        results["acc"].append(acc)
        results["prec"].append(prec)
        results["rec"].append(rec)
        results["f1"].append(f1)
        results["auc"].append(auc)
        
        fold_data.append({
            "fold": fold + 1,
            "model": clf,
            "X_val_pca": X_val_pca,
            "y_val": y_val,
            "auc": auc
        })
        
        print(f"  [Fold {fold+1}/5] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} ({time.time()-fold_t0:.2f}s)", flush=True)

    # Print summary performance metrics
    print("\n" + "=" * 80)
    print("PCA BASELINE SYSTEM 5-FOLD SUMMARY STATS")
    print("=" * 80)
    print(f"  Accuracy:  {np.mean(results['acc']):.4f} ± {np.std(results['acc']):.4f}")
    print(f"  Precision: {np.mean(results['prec']):.4f} ± {np.std(results['prec']):.4f}")
    print(f"  Recall:    {np.mean(results['rec']):.4f} ± {np.std(results['rec']):.4f}")
    print(f"  F1-Score:  {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")
    print(f"  ROC-AUC:   {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")
    print("=" * 80)

    # 4. Identify Best Performing Fold based on ROC-AUC
    best_fold_idx = np.argmax([f["auc"] for f in fold_data])
    best_fold = fold_data[best_fold_idx]
    print(f"\n[+] Selected Fold {best_fold['fold']} as the best performing fold (ROC-AUC = {best_fold['auc']:.4f}) for SHAP Explainability.")
    
    best_model = best_fold["model"]
    X_val_pca_full = best_fold["X_val_pca"]
    
    # Subsample validation set for SHAP computation to avoid execution hang (CPU bound TreeExplainer with 500 deep trees)
    shap_sample_idx = np.random.choice(len(X_val_pca_full), min(100, len(X_val_pca_full)), replace=False)
    X_val_pca_best = X_val_pca_full[shap_sample_idx]
    
    # 5. Compute SHAP Values on validation subset
    print("Computing SHAP values using TreeExplainer (on representative 100-sample validation subset)...")
    explainer = shap.TreeExplainer(best_model)
    shap_values_raw = explainer.shap_values(X_val_pca_best)
    
    # Extract SHAP values for class 1 (Malignant)
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
    else:
        if len(shap_values_raw.shape) == 3:
            shap_values = shap_values_raw[:, :, 1]
        else:
            shap_values = shap_values_raw
            
    # Calculate feature importances from mean absolute SHAP values
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_shap)[::-1]
    
    print("\n" + "=" * 50)
    print("TOP 5 MOST IMPORTANT PCA COMPONENTS (SHAP)")
    print("=" * 50)
    for rank, idx in enumerate(top_indices[:5]):
        print(f"  {rank+1}. PCA Component {idx} (Mean |SHAP| = {mean_shap[idx]:.5f})")
    print("=" * 50)
    
    # 6. Generate JPEGs at 300 DPI
    feature_names = [f"PCA Component {i}" for i in range(100)]
    generate_shap_plots(
        shap_values=shap_values,
        X_val_pca=X_val_pca_best,
        feature_names=feature_names,
        save_dir="images",
        dpi=300
    )

if __name__ == "__main__":
    main()
