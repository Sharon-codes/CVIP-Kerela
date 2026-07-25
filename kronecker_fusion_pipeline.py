#!/usr/bin/env python
"""
Kronecker Cross-Covariance Feature Fusion & HistGradientBoosting Pipeline with TTA
Author: Principal ML Researcher (Algebraic Topology & Computational Diagnostics)

This script implements:
1. KroneckerFusion (Scikit-Learn Transformer for outer product interaction)
2. HistGradientBoostingClassifier integration
3. CPU-Friendly Test-Time Augmentation (TTA) inference wrapper
4. Publication-Ready 300 DPI Visualizations:
   - Figure 1: t-SNE Feature Space Comparison (Baseline Concatenation vs Kronecker Fusion)
   - Figure 2: Comparative ROC Curve (Performance Jump)
   - Figure 3: Permutation Feature Importance (Interpretability of Interaction Terms)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance

# Import dataset loader and feature extraction functions from existing benchmark module
from tda_benchmark import load_dataset, extract_tda_features, PersistenceLandscape, PersistenceImage, extract_cnn_features, train_medical_weights

RANDOM_STATE = 42
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# 1. Kronecker Cross-Covariance Feature Fusion Transformer
# =====================================================================
class KroneckerFusion(BaseEstimator, TransformerMixin):
    """
    Kronecker Cross-Covariance Feature Fusion Transformer.
    Compresses TDA and CNN spatial features to n_components using PCA independently,
    then computes the sample-wise outer product (Kronecker product) of the two reduced
    vectors for each sample.
    
    Output dimension: n_components * n_components (50 * 50 = 2,500 dimensions).
    """
    def __init__(self, tda_dim=5600, n_components=50, random_state=RANDOM_STATE):
        self.tda_dim = tda_dim
        self.n_components = n_components
        self.random_state = random_state
        self.pca_tda = None
        self.pca_cnn = None

    def fit(self, X, y=None):
        v_tda = X[:, :self.tda_dim]
        v_cnn = X[:, self.tda_dim:]
        
        self.pca_tda = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_cnn = PCA(n_components=self.n_components, random_state=self.random_state)
        
        self.pca_tda.fit(v_tda)
        self.pca_cnn.fit(v_cnn)
        return self

    def transform(self, X):
        if self.pca_tda is None or self.pca_cnn is None:
            raise ValueError("KroneckerFusion transformer has not been fitted yet.")
            
        v_tda = X[:, :self.tda_dim]
        v_cnn = X[:, self.tda_dim:]
        
        v_tda_pca = self.pca_tda.transform(v_tda)
        v_cnn_pca = self.pca_cnn.transform(v_cnn)
        
        # Compute sample-wise outer product (Kronecker product) using vectorized np.einsum
        v_fused = np.einsum('bi,bj->bij', v_tda_pca, v_cnn_pca).reshape(X.shape[0], -1)
        return v_fused


# =====================================================================
# 2. CPU-Friendly Test-Time Augmentation (TTA) Function
# =====================================================================
def predict_with_tta(X_imgs_val, X_tda_val, resnet_encoder, scaler, kronecker_transformer, model, tda_dim=5600):
    """
    Performs Test-Time Augmentation across 4 spatial variations (Original, H-Flip, V-Flip, Both).
    Averages the predicted probabilities to produce smooth, robust marginal predictions.
    """
    # 4 image variations (ensuring contiguous arrays for PyTorch tensor compatibility)
    img_variations = [
        X_imgs_val,                                    # Original
        np.ascontiguousarray(X_imgs_val[:, :, ::-1]),   # Horizontal flip
        np.ascontiguousarray(X_imgs_val[:, ::-1, :]),   # Vertical flip
        np.ascontiguousarray(X_imgs_val[:, ::-1, ::-1]) # Horizontal + Vertical flip
    ]
    
    probs_list = []
    
    for X_var in img_variations:
        # Extract spatial features for variation
        X_cnn_var = extract_cnn_features(X_var, resnet_encoder)
        X_hybrid_var = np.hstack([X_tda_val, X_cnn_var])
        
        # Pipeline transform
        X_scaled_var = scaler.transform(X_hybrid_var)
        X_fused_var = kronecker_transformer.transform(X_scaled_var)
        
        # Predict probabilities
        p_var = model.predict_proba(X_fused_var)
        probs_list.append(p_var)
        
    # Average predicted probabilities across all 4 variations
    avg_probs = np.mean(probs_list, axis=0)
    return avg_probs


# =====================================================================
# 3. Publication-Ready Visualizations (300 DPI JPEGs)
# =====================================================================
def generate_fusion_figures(X_val_concat, X_val_kronecker, y_val, model_old, model_new, val_probs_old, val_probs_new, save_dir="images", dpi=300):
    """
    Generates and saves 3 publication-ready figures as 300 DPI JPEGs in the save_dir folder:
    - Figure 1: t-SNE Feature Space Comparison (Baseline Concatenation vs Kronecker Fusion)
    - Figure 2: Comparative ROC Curve (Baseline ExtraTrees vs Upgraded HistGB + Kronecker + TTA)
    - Figure 3: Permutation Feature Importance (Top 15 Kronecker Interaction Features)
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("\nGenerating Publication-Ready Figures (300 DPI JPEG)...")
    
    # -----------------------------------------------------------------
    # Figure 1: t-SNE Feature Space Comparison
    # -----------------------------------------------------------------
    print("  [+] Computing t-SNE projections for Baseline vs Kronecker Fusion...")
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=RANDOM_STATE)
    
    # Subsample validation set if large to optimize rendering speed
    max_pts = 1000
    if len(y_val) > max_pts:
        rng = np.random.RandomState(RANDOM_STATE)
        pts_idx = rng.choice(len(y_val), max_pts, replace=False)
        X_concat_sub = X_val_concat[pts_idx]
        X_kron_sub = X_val_kronecker[pts_idx]
        y_sub = y_val[pts_idx]
    else:
        X_concat_sub = X_val_concat
        X_kron_sub = X_val_kronecker
        y_sub = y_val
        
    tsne_concat = tsne.fit_transform(X_concat_sub)
    tsne_kron = tsne.fit_transform(X_kron_sub)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {0: "#1f77b4", 1: "#d62728"}
    labels = {0: "Benign", 1: "Malignant"}
    
    for cls in [0, 1]:
        mask = (y_sub == cls)
        axes[0].scatter(tsne_concat[mask, 0], tsne_concat[mask, 1], c=colors[cls], label=labels[cls], alpha=0.7, edgecolors='w', s=45)
        axes[1].scatter(tsne_kron[mask, 0], tsne_kron[mask, 1], c=colors[cls], label=labels[cls], alpha=0.7, edgecolors='w', s=45)
        
    axes[0].set_title("Baseline Feature Space\n(Simple Concatenation)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=10)
    
    axes[1].set_title("Upgraded Feature Space\n(Kronecker Cross-Covariance Fusion)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=11)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=10)
    
    plt.tight_layout()
    fig1_path = os.path.join(save_dir, "tsne_feature_space_comparison.jpg")
    plt.savefig(fig1_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Figure 1 saved: {fig1_path}")

    # -----------------------------------------------------------------
    # Figure 2: Comparative ROC Curve
    # -----------------------------------------------------------------
    fpr_old, tpr_old, _ = roc_curve(y_val, val_probs_old)
    auc_old = roc_auc_score(y_val, val_probs_old)
    
    fpr_new, tpr_new, _ = roc_curve(y_val, val_probs_new)
    auc_new = roc_auc_score(y_val, val_probs_new)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr_old, tpr_old, color="#7f7f7f", linestyle="--", linewidth=2.0, 
            label=f"Baseline (Extra Trees + Concatenation): AUC = {auc_old:.4f}")
    ax.plot(fpr_new, tpr_new, color="#2ca02c", linewidth=2.8, 
            label=f"Upgraded (Kronecker + HistGB + TTA): AUC = {auc_new:.4f}")
    
    ax.fill_between(fpr_new, tpr_new, alpha=0.15, color="#2ca02c")
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.2, label="Chance Line (AUC = 0.50)")
    
    ax.set_title("Comparative ROC Curve Performance Jump", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR / Sensitivity)", fontsize=11)
    ax.legend(loc="lower right", fontsize=10.5, frameon=True)
    
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, "comparative_roc_curve.jpg")
    plt.savefig(fig2_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Figure 2 saved: {fig2_path}")

    # -----------------------------------------------------------------
    # Figure 3: Permutation Feature Importance
    # -----------------------------------------------------------------
    print("  [+] Computing Permutation Feature Importances on validation set...")
    # Subsample validation set for permutation calculations to be computationally friendly
    if len(y_val) > 150:
        rng = np.random.RandomState(RANDOM_STATE)
        p_idx = rng.choice(len(y_val), 150, replace=False)
        X_perm = X_val_kronecker[p_idx]
        y_perm = y_val[p_idx]
    else:
        X_perm = X_val_kronecker
        y_perm = y_val
        
    perm_imp = permutation_importance(model_new, X_perm, y_perm, n_repeats=2, random_state=RANDOM_STATE, n_jobs=1)
    
    top_indices = np.argsort(perm_imp.importances_mean)[::-1][:15]
    top_means = perm_imp.importances_mean[top_indices]
    top_stds = perm_imp.importances_std[top_indices]
    
    feature_labels = []
    for idx in top_indices:
        tda_comp = (idx // 50) + 1
        cnn_comp = (idx % 50) + 1
        feature_labels.append(f"Interaction (TDA-PC{tda_comp} × CNN-PC{cnn_comp})")
        
    fig, ax = plt.subplots(figsize=(9.5, 6))
    
    y_pos = np.arange(len(top_indices))
    # All features here are derived from the Kronecker interaction (TDA x CNN)
    ax.barh(y_pos, top_means, xerr=top_stds, align='center', color="#ff7f0e", edgecolor="black", alpha=0.85, ecolor="black", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=10)
    ax.invert_yaxis()
    
    ax.set_xlabel("Permutation Importance Mean Decrease in Accuracy", fontsize=11)
    ax.set_title("Top 15 Discriminative Kronecker Interaction Features", fontsize=13, fontweight="bold", pad=12)
    
    plt.tight_layout()
    fig3_path = os.path.join(save_dir, "permutation_feature_importance.jpg")
    plt.savefig(fig3_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Figure 3 saved: {fig3_path}")


# =====================================================================
# 4. Cross-Validation Execution Loop
# =====================================================================
def main():
    print("=" * 70)
    print("KRONECKER FUSION, HISTGRADIENTBOOSTING & TTA BENCHMARK")
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
    
    # Storage for Baseline (ExtraTrees + Concatenation) vs Upgraded (HistGB + Kronecker + TTA)
    results_baseline = {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []}
    results_upgraded = {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []}
    
    last_fold_data = None
    
    print("\nInitiating 5-Fold Cross-Validation Benchmark...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_tda, y, groups=groups)):
        fold_t0 = time.time()
        print(f"\nProcessing Fold {fold+1}/5...", flush=True)
        
        # Split TDA and images for current fold
        X_tda_train, X_tda_val = X_tda[train_idx], X_tda[val_idx]
        X_imgs_train, X_imgs_val = X_imgs[train_idx], X_imgs[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Medical Spatial Feature Extraction (ResNet18 Autoencoder)
        resnet_encoder = train_medical_weights(X_imgs_train, "resnet18", epochs=5)
        X_cnn_train = extract_cnn_features(X_imgs_train, resnet_encoder)
        X_cnn_val = extract_cnn_features(X_imgs_val, resnet_encoder)
        
        X_hybrid_train = np.hstack([X_tda_train, X_cnn_train])
        X_hybrid_val = np.hstack([X_tda_val, X_cnn_val])
        
        # -------------------------------------------------------------
        # A. BASELINE PIPELINE (Extra Trees + Simple Concatenation)
        # -------------------------------------------------------------
        scaler_base = StandardScaler()
        X_train_base_scaled = scaler_base.fit_transform(X_hybrid_train)
        X_val_base_scaled = scaler_base.transform(X_hybrid_val)
        
        clf_base = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        clf_base.fit(X_train_base_scaled, y_train)
        
        y_pred_base = clf_base.predict(X_val_base_scaled)
        y_prob_base = clf_base.predict_proba(X_val_base_scaled)[:, 1]
        
        results_baseline["acc"].append(accuracy_score(y_val, y_pred_base))
        results_baseline["prec"].append(precision_score(y_val, y_pred_base, zero_division=0))
        results_baseline["rec"].append(recall_score(y_val, y_pred_base, zero_division=0))
        results_baseline["f1"].append(f1_score(y_val, y_pred_base, zero_division=0))
        results_baseline["auc"].append(roc_auc_score(y_val, y_prob_base))
        
        # -------------------------------------------------------------
        # B. UPGRADED PIPELINE (Kronecker Fusion + HistGradientBoosting + TTA)
        # -------------------------------------------------------------
        scaler_up = StandardScaler()
        X_train_up_scaled = scaler_up.fit_transform(X_hybrid_train)
        
        kronecker_trans = KroneckerFusion(tda_dim=tda_dim, n_components=50, random_state=RANDOM_STATE)
        X_train_fused = kronecker_trans.fit_transform(X_train_up_scaled)
        
        clf_upgraded = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=7,
            l2_regularization=0.1,
            random_state=RANDOM_STATE
        )
        clf_upgraded.fit(X_train_fused, y_train)
        
        # Execute Test-Time Augmentation (TTA) on validation fold
        probs_val_tta = predict_with_tta(
            X_imgs_val=X_imgs_val,
            X_tda_val=X_tda_val,
            resnet_encoder=resnet_encoder,
            scaler=scaler_up,
            kronecker_transformer=kronecker_trans,
            model=clf_upgraded,
            tda_dim=tda_dim
        )
        
        y_prob_up = probs_val_tta[:, 1]
        y_pred_up = (y_prob_up >= 0.5).astype(int)
        
        acc_up = accuracy_score(y_val, y_pred_up)
        prec_up = precision_score(y_val, y_pred_up, zero_division=0)
        rec_up = recall_score(y_val, y_pred_up, zero_division=0)
        f1_up = f1_score(y_val, y_pred_up, zero_division=0)
        auc_up = roc_auc_score(y_val, y_prob_up)
        
        results_upgraded["acc"].append(acc_up)
        results_upgraded["prec"].append(prec_up)
        results_upgraded["rec"].append(rec_up)
        results_upgraded["f1"].append(f1_up)
        results_upgraded["auc"].append(auc_up)
        
        # Save last fold validation data for plotting
        X_val_up_scaled = scaler_up.transform(X_hybrid_val)
        X_val_fused = kronecker_trans.transform(X_val_up_scaled)
        
        last_fold_data = {
            "X_val_concat": X_val_base_scaled,
            "X_val_kronecker": X_val_fused,
            "y_val": y_val,
            "model_old": clf_base,
            "model_new": clf_upgraded,
            "val_probs_old": y_prob_base,
            "val_probs_new": y_prob_up
        }
        
        print(f"  [Fold {fold+1}/5] Baseline Accuracy: {results_baseline['acc'][-1]:.4f}, AUC: {results_baseline['auc'][-1]:.4f} | "
              f"Upgraded Accuracy: {acc_up:.4f}, AUC: {auc_up:.4f} ({time.time()-fold_t0:.2f}s)", flush=True)

    # 4. Print Comparative Performance Table
    print("\n" + "=" * 80)
    print("PUBLICATION-GRADE BENCHMARK PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    
    summary_rows = []
    for name, res in [("Baseline (Extra Trees + Concatenation)", results_baseline), 
                      ("Upgraded (Kronecker + HistGB + TTA)", results_upgraded)]:
        summary_rows.append({
            "Architecture Pipeline": name,
            "ACCURACY": f"{np.mean(res['acc']):.4f} ± {np.std(res['acc']):.4f}",
            "PRECISION": f"{np.mean(res['prec']):.4f} ± {np.std(res['prec']):.4f}",
            "RECALL": f"{np.mean(res['rec']):.4f} ± {np.std(res['rec']):.4f}",
            "F1-SCORE": f"{np.mean(res['f1']):.4f} ± {np.std(res['f1']):.4f}",
            "ROC-AUC": f"{np.mean(res['auc']):.4f} ± {np.std(res['auc']):.4f}"
        })
        
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_markdown(index=False))
    print("=" * 80)

    # 5. Generate Figures
    generate_fusion_figures(
        X_val_concat=last_fold_data["X_val_concat"],
        X_val_kronecker=last_fold_data["X_val_kronecker"],
        y_val=last_fold_data["y_val"],
        model_old=last_fold_data["model_old"],
        model_new=last_fold_data["model_new"],
        val_probs_old=last_fold_data["val_probs_old"],
        val_probs_new=last_fold_data["val_probs_new"],
        save_dir="images",
        dpi=300
    )


if __name__ == "__main__":
    main()
