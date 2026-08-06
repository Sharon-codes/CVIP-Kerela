#!/usr/bin/env python
"""
Algorithmic Rigor & Multi-Classifier Benchmarking Module:
Benchmarks ExtraTrees, RandomForest, LightGBM, and SVM pipelines.
Includes in-loop GridSearchCV tuning of L1 feature selector (C), PCA components (k),
and classifier regularizations inside StratifiedGroupKFold splits.

Evaluates Internal Validation & Unseen External Cohort Generalization,
profiles CPU Inference Latency (ms/img) and Peak RAM (MB), and outputs
a formatted Markdown comparative matrix.

Author: Lead Biomedical ML Engineer (Q1 Journal Submission Suite)
"""

import os
import sys
import time
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, recall_score, confusion_matrix, accuracy_score, f1_score
)

from core_pipeline import setup_data_split, load_real_images, extract_cnn, extract_tda

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore")


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    return {
        "auc": auc,
        "sens": sens,
        "spec": spec,
        "acc": acc,
        "f1": f1
    }


def benchmark_classifiers(primary_dir="data/primary", external_dir="data/external", save_dir="images", dpi=300):
    print("=" * 90)
    print("ALGORITHMIC RIGOR & MULTI-CLASSIFIER BENCHMARKING (EXTRA TREES, RF, LIGHTGBM, SVM)")
    print("Target Journal: Biomedical Physics & Engineering Express")
    print("=" * 90)

    # 1. Setup & Load Data
    setup_data_split()

    print("\n[+] Loading Primary Dataset...")
    X_imgs_prim, y_prim, groups_prim, _, _ = load_real_images(primary_dir)

    print("\n[+] Loading External Cohort Dataset...")
    X_imgs_ext, y_ext, _, _, _ = load_real_images(external_dir)

    # 2. Extract Features
    print("\n[+] Extracting Spatial Features (MobileNetV2 512-D) for Primary & External Cohorts...")
    X_cnn_prim = extract_cnn(X_imgs_prim)
    X_cnn_ext = extract_cnn(X_imgs_ext)

    print("\n[+] Extracting Topological Features (Cubical Homology 5,600-D) for Primary & External Cohorts...")
    X_tda_prim = extract_tda(X_imgs_prim)
    X_tda_ext = extract_tda(X_imgs_ext)

    X_hybrid_prim = np.hstack([X_tda_prim, X_cnn_prim])
    X_hybrid_ext = np.hstack([X_tda_ext, X_cnn_ext])

    # 3. Define 4 Classifier Models & Search Grids
    classifier_configs = {
        "Model A (ExtraTrees)": {
            "base_clf": ExtraTreesClassifier(
                n_estimators=500, class_weight='balanced_subsample',
                max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1
            ),
            "param_grid": {
                'feature_selection__estimator__C': [0.1, 0.5, 1.0],
                'pca__n_components': [60, 90, 120],
                'clf__min_samples_leaf': [2, 4, 8]
            }
        },
        "Model B (RandomForest)": {
            "base_clf": RandomForestClassifier(
                n_estimators=500, class_weight='balanced_subsample',
                max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1
            ),
            "param_grid": {
                'feature_selection__estimator__C': [0.1, 0.5, 1.0],
                'pca__n_components': [60, 90, 120],
                'clf__min_samples_leaf': [2, 4, 8]
            }
        },
        "Model C (LightGBM)": {
            "base_clf": LGBMClassifier(
                n_estimators=300, class_weight='balanced', max_depth=5,
                num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            ),
            "param_grid": {
                'feature_selection__estimator__C': [0.1, 0.5, 1.0],
                'pca__n_components': [60, 90, 120],
                'clf__reg_alpha': [0.1, 1.0],
                'clf__reg_lambda': [0.1, 1.0]
            }
        },
        "Model D (SVM)": {
            "base_clf": SVC(
                kernel='rbf', class_weight='balanced', probability=True,
                random_state=RANDOM_STATE
            ),
            "param_grid": {
                'feature_selection__estimator__C': [0.1, 0.5, 1.0],
                'pca__n_components': [60, 90, 120],
                'clf__C': [0.1, 1.0, 10.0],
                'clf__gamma': ['scale', 'auto']
            }
        }
    }

    # 4. Stratified Group K-Fold Split (5 Folds)
    sgkf = StratifiedGroupKFold(n_splits=5)
    splits = list(sgkf.split(X_hybrid_prim, y_prim, groups=groups_prim))
    train_idx, val_idx = splits[0]  # Representative Fold 1

    X_train, y_train = X_hybrid_prim[train_idx], y_prim[train_idx]
    X_val, y_val = X_hybrid_prim[val_idx], y_prim[val_idx]

    results_table = []

    for model_name, cfg in classifier_configs.items():
        print(f"\n" + "-" * 70)
        print(f"[+] Benchmarking & Tuning {model_name}...")
        print("-" * 70)

        # Build 4-Step Pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(
                LogisticRegression(
                    penalty='l1', solver='liblinear',
                    class_weight='balanced', random_state=RANDOM_STATE
                )
            )),
            ('pca', PCA(random_state=RANDOM_STATE)),
            ('clf', cfg["base_clf"])
        ])

        # GridSearchCV (3-fold inner CV)
        grid_search = GridSearchCV(
            pipe, param_grid=cfg["param_grid"], scoring='roc_auc',
            cv=3, n_jobs=-1, verbose=0
        )

        t0_fit = time.perf_counter()
        grid_search.fit(X_train, y_train)
        t_fit_elapsed = time.perf_counter() - t0_fit

        best_pipe = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"  [+] GridSearchCV Complete in {t_fit_elapsed:.2f}s.")
        print(f"  [+] Best Hyperparameters: {best_params}")

        # Evaluate on Internal Validation Fold
        y_val_prob = best_pipe.predict_proba(X_val)[:, 1]
        val_m = compute_metrics(y_val, y_val_prob)

        # Evaluate on Unseen External Cohort
        y_ext_prob = best_pipe.predict_proba(X_hybrid_ext)[:, 1]
        ext_m = compute_metrics(y_ext, y_ext_prob)

        # Profile CPU Latency and Peak RAM per image
        sample_indices = np.random.choice(len(X_val), min(100, len(X_val)), replace=False)
        X_sample = X_val[sample_indices]

        tracemalloc.start()
        t0_lat = time.perf_counter()
        _ = best_pipe.predict_proba(X_sample)
        t1_lat = time.perf_counter()
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        latency_ms = ((t1_lat - t0_lat) / len(X_sample)) * 1000.0
        peak_ram_mb = peak_mem_bytes / (1024.0 * 1024.0)

        print(f"  Internal Validation -> AUC: {val_m['auc']:.4f} | Recall: {val_m['sens']:.4f} | Specificity: {val_m['spec']:.4f}")
        print(f"  External Cohort     -> AUC: {ext_m['auc']:.4f} | Recall: {ext_m['sens']:.4f} | Specificity: {ext_m['spec']:.4f}")
        print(f"  Hardware Profile    -> Latency: {latency_ms:.2f} ms/img | Peak RAM: {peak_ram_mb:.2f} MB")

        results_table.append({
            "Model": model_name,
            "Internal_AUC": val_m['auc'],
            "Internal_Recall": val_m['sens'],
            "Internal_Specificity": val_m['spec'],
            "External_AUC": ext_m['auc'],
            "External_Recall": ext_m['sens'],
            "External_Specificity": ext_m['spec'],
            "Latency_ms": latency_ms,
            "Peak_RAM_MB": peak_ram_mb,
            "Best_Params": str(best_params)
        })

    df_res = pd.DataFrame(results_table)

    # 5. Output Markdown Table Matrix
    print("\n" + "=" * 90)
    print("FINAL MULTI-CLASSIFIER ALGORITHMIC BENCHMARK MATRIX")
    print("=" * 90)

    md_table = "| Model | Internal AUC | External AUC | Internal Spec | Internal Recall | External Spec | External Recall | Latency (ms/img) | Peak RAM (MB) |\n"
    md_table += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for r in results_table:
        md_table += f"| **{r['Model']}** | {r['Internal_AUC']:.4f} | {r['External_AUC']:.4f} | {r['Internal_Specificity']:.4f} | {r['Internal_Recall']:.4f} | {r['External_Specificity']:.4f} | {r['External_Recall']:.4f} | {r['Latency_ms']:.2f} ms | {r['Peak_RAM_MB']:.2f} MB |\n"

    print(md_table)

    # Save to Markdown Report File
    os.makedirs(save_dir, exist_ok=True)
    report_file = "exp_classifiers_results.md"
    with open(report_file, "w") as f:
        f.write("# Multi-Classifier Algorithmic Rigor Benchmark\n\n")
        f.write("GridSearchCV Hyperparameter Optimization across ExtraTrees, RandomForest, LightGBM, and SVM.\n\n")
        f.write(md_table)
        f.write("\n\n### Best Hyperparameter Configurations\n\n")
        for r in results_table:
            f.write(f"- **{r['Model']}**: `{r['Best_Params']}`\n")

    print(f"[+] Written comparative markdown report to {report_file}")

    # 6. Plot Publication-Grade 300 DPI Bar Chart
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    models_list = df_res["Model"].values
    int_aucs = df_res["Internal_AUC"].values
    ext_aucs = df_res["External_AUC"].values

    x_idx = np.arange(len(models_list))
    width = 0.35

    rects1 = ax.bar(x_idx - width/2, int_aucs, width, label='Internal Validation AUC', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x_idx + width/2, ext_aucs, width, label='External Cohort AUC', color='#ff7f0e', edgecolor='black')

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax.set_xticks(x_idx)
    ax.set_xticklabels(models_list, fontsize=10.5, fontweight='bold')
    ax.set_ylim(0.0, 1.15)
    ax.set_title("Multi-Classifier Algorithmic Benchmarking (GridSearchCV Optimized)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("ROC-AUC Score", fontsize=11, fontweight='bold')
    ax.legend(fontsize=10.5, frameon=True, loc='upper right')
    plt.tight_layout()

    out_path = os.path.join(save_dir, "exp_classifier_comparison.jpg")
    plt.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved comparative publication plot to {out_path}")

    return df_res, md_table


if __name__ == "__main__":
    benchmark_classifiers()
