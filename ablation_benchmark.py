import os
import time
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from gtda.diagrams import PersistenceLandscape
import torch
import torch.nn as nn

# Import custom functions from our benchmark module
from tda_benchmark import load_dataset, extract_tda_features, PersistenceImage

RANDOM_STATE = 42

def run_task_a():
    print("\n" + "="*50)
    print("RUNNING TASK A: Landscape vs. Image Ablation")
    print("="*50)
    
    # 1. Load dataset (64x64)
    X_imgs, y, groups = load_dataset("data", img_size=(64, 64))
    
    # 2. Extract diagrams
    X_diags = extract_tda_features(X_imgs)
    
    # 3. Setup single-fold split (Fold 1 of 5)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sgkf.split(X_diags, y, groups=groups))
    
    X_diags_train, X_diags_val = X_diags[train_idx], X_diags[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Setup Vectorizers
    pl = PersistenceLandscape(n_layers=3, n_values=100, n_jobs=-1)
    pi = PersistenceImage(sigma=0.1, n_bins=50)
    
    # Track 1: Landscape Only
    print("\nEvaluating Track 1: Persistence Landscapes Only...")
    X_train_pl = pl.fit_transform(X_diags_train).reshape(len(train_idx), -1)
    X_val_pl = pl.transform(X_diags_val).reshape(len(val_idx), -1)
    
    scaler_pl = StandardScaler()
    X_train_pl_scaled = scaler_pl.fit_transform(X_train_pl)
    X_val_pl_scaled = scaler_pl.transform(X_val_pl)
    
    clf_pl = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    clf_pl.fit(X_train_pl_scaled, y_train)
    y_pred_pl = clf_pl.predict(X_val_pl_scaled)
    y_prob_pl = clf_pl.predict_proba(X_val_pl_scaled)[:, 1]
    acc_pl = accuracy_score(y_val, y_pred_pl)
    auc_pl = roc_auc_score(y_val, y_prob_pl)
    print(f"  Landscape Only - Accuracy: {acc_pl:.4f}, AUC: {auc_pl:.4f}")
    
    # Track 2: Image Only
    print("\nEvaluating Track 2: Persistence Images Only...")
    X_train_pi = pi.fit_transform(X_diags_train)
    X_val_pi = pi.transform(X_diags_val)
    
    scaler_pi = StandardScaler()
    X_train_pi_scaled = scaler_pi.fit_transform(X_train_pi)
    X_val_pi_scaled = scaler_pi.transform(X_val_pi)
    
    clf_pi = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    clf_pi.fit(X_train_pi_scaled, y_train)
    y_pred_pi = clf_pi.predict(X_val_pi_scaled)
    y_prob_pi = clf_pi.predict_proba(X_val_pi_scaled)[:, 1]
    acc_pi = accuracy_score(y_val, y_pred_pi)
    auc_pi = roc_auc_score(y_val, y_prob_pi)
    print(f"  Persistence Image Only - Accuracy: {acc_pi:.4f}, AUC: {auc_pi:.4f}")
    
    # Track 3: Concatenated
    print("\nEvaluating Track 3: Concatenated Landscapes + Images...")
    X_train_concat = np.hstack([X_train_pl, X_train_pi])
    X_val_concat = np.hstack([X_val_pl, X_val_pi])
    
    scaler_concat = StandardScaler()
    X_train_concat_scaled = scaler_concat.fit_transform(X_train_concat)
    X_val_concat_scaled = scaler_concat.transform(X_val_concat)
    
    clf_concat = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    clf_concat.fit(X_train_concat_scaled, y_train)
    y_pred_concat = clf_concat.predict(X_val_concat_scaled)
    y_prob_concat = clf_concat.predict_proba(X_val_concat_scaled)[:, 1]
    acc_concat = accuracy_score(y_val, y_pred_concat)
    auc_concat = roc_auc_score(y_val, y_prob_concat)
    print(f"  Concatenated - Accuracy: {acc_concat:.4f}, AUC: {auc_concat:.4f}")
    
    return {
        "Landscape Only": {"Accuracy": acc_pl, "AUC": auc_pl},
        "Persistence Image Only": {"Accuracy": acc_pi, "AUC": auc_pi},
        "Concatenated": {"Accuracy": acc_concat, "AUC": auc_concat}
    }

def run_task_b():
    print("\n" + "="*50)
    print("RUNNING TASK B: Inference Runtime Benchmarking")
    print("="*50)
    
    # 1. Load dataset (64x64)
    X_imgs, y, groups = load_dataset("data", img_size=(64, 64))
    
    # Take first 100 images
    X_100 = X_imgs[:100]
    
    # Time Homology Extraction on CPU
    print("Benchmarking Homology extraction for 100 images...")
    t0 = time.time()
    X_diags_100 = extract_tda_features(X_100, batch_size=100)
    total_homology_time_ms = (time.time() - t0) * 1000
    avg_homology_time_ms = total_homology_time_ms / 100.0
    print(f"  Total Homology Time: {total_homology_time_ms:.2f} ms")
    print(f"  Average Homology Time per image: {avg_homology_time_ms:.2f} ms")
    
    # Setup vectors and train a quick model to benchmark classification
    pl = PersistenceLandscape(n_layers=3, n_values=100, n_jobs=-1)
    pi = PersistenceImage(sigma=0.1, n_bins=50)
    
    X_pl_100 = pl.fit_transform(X_diags_100).reshape(100, -1)
    X_pi_100 = pi.fit_transform(X_diags_100)
    X_concat_100 = np.hstack([X_pl_100, X_pi_100])
    
    scaler = StandardScaler()
    X_scaled_100 = scaler.fit_transform(X_concat_100)
    
    clf = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_scaled_100, y[:100])
    
    # Benchmark classification time (predict)
    print("Benchmarking classification prediction time...")
    t1 = time.time()
    _ = clf.predict(X_scaled_100)
    total_clf_time_ms = (time.time() - t1) * 1000
    avg_clf_time_ms = total_clf_time_ms / 100.0
    print(f"  Total Classification Time: {total_clf_time_ms:.2f} ms")
    print(f"  Average Classification Time per image: {avg_clf_time_ms:.2f} ms")
    
    return {
        "avg_homology_ms": avg_homology_time_ms,
        "avg_classification_ms": avg_clf_time_ms
    }

def run_task_c():
    print("\n" + "="*50)
    print("RUNNING TASK C: Resolution Ablation (64x64 vs. 128x128)")
    print("="*50)
    
    # Load a small subset of 500 images at both resolutions
    subset_size = 500
    
    # 1. Evaluate 64x64
    print("Loading 500 images at 64x64...")
    X_imgs_64, y_64, groups_64 = load_dataset("data", img_size=(64, 64))
    # Select first 500
    X_imgs_64 = X_imgs_64[:subset_size]
    y_64 = y_64[:subset_size]
    groups_64 = groups_64[:subset_size]
    
    print("Extracting Homology at 64x64...")
    t0 = time.time()
    X_diags_64 = extract_tda_features(X_imgs_64)
    time_64 = time.time() - t0
    
    # Single-fold accuracy
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sgkf.split(X_diags_64, y_64, groups=groups_64))
    
    pl = PersistenceLandscape(n_layers=3, n_values=100, n_jobs=-1)
    pi = PersistenceImage(sigma=0.1, n_bins=50)
    
    X_train_pl = pl.fit_transform(X_diags_64[train_idx]).reshape(len(train_idx), -1)
    X_val_pl = pl.transform(X_diags_64[val_idx]).reshape(len(val_idx), -1)
    X_train_pi = pi.fit_transform(X_diags_64[train_idx])
    X_val_pi = pi.transform(X_diags_64[val_idx])
    X_train_tda = np.hstack([X_train_pl, X_train_pi])
    X_val_tda = np.hstack([X_val_pl, X_val_pi])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_tda)
    X_val_scaled = scaler.transform(X_val_tda)
    
    clf = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_scaled, y_64[train_idx])
    acc_64 = accuracy_score(y_64[val_idx], clf.predict(X_val_scaled))
    
    print(f"  64x64: CPU Extraction Time = {time_64:.2f}s, Validation Accuracy = {acc_64:.4f}")
    
    # 2. Evaluate 128x128
    print("\nLoading 500 images at 128x128...")
    X_imgs_128, y_128, groups_128 = load_dataset("data", img_size=(128, 128))
    X_imgs_128 = X_imgs_128[:subset_size]
    y_128 = y_128[:subset_size]
    groups_128 = groups_128[:subset_size]
    
    print("Extracting Homology at 128x128...")
    t1 = time.time()
    X_diags_128 = extract_tda_features(X_imgs_128)
    time_128 = time.time() - t1
    
    # Single-fold accuracy
    train_idx_128, val_idx_128 = next(sgkf.split(X_diags_128, y_128, groups=groups_128))
    
    X_train_pl_128 = pl.fit_transform(X_diags_128[train_idx_128]).reshape(len(train_idx_128), -1)
    X_val_pl_128 = pl.transform(X_diags_128[val_idx_128]).reshape(len(val_idx_128), -1)
    X_train_pi_128 = pi.fit_transform(X_diags_128[train_idx_128])
    X_val_pi_128 = pi.transform(X_diags_128[val_idx_128])
    X_train_tda_128 = np.hstack([X_train_pl_128, X_train_pi_128])
    X_val_tda_128 = np.hstack([X_val_pl_128, X_val_pi_128])
    
    scaler_128 = StandardScaler()
    X_train_scaled_128 = scaler_128.fit_transform(X_train_tda_128)
    X_val_scaled_128 = scaler_128.transform(X_val_tda_128)
    
    clf_128 = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    clf_128.fit(X_train_scaled_128, y_128[train_idx_128])
    acc_128 = accuracy_score(y_128[val_idx_128], clf_128.predict(X_val_scaled_128))
    
    print(f"  128x128: CPU Extraction Time = {time_128:.2f}s, Validation Accuracy = {acc_128:.4f}")
    
    return {
        "64x64": {"Time": time_64, "Accuracy": acc_64},
        "128x128": {"Time": time_128, "Accuracy": acc_128}
    }

def main():
    res_a = run_task_a()
    res_b = run_task_b()
    res_c = run_task_c()
    
    print("\n" + "="*50)
    print("ALL ABLATIONS AND RUNTIME BENCHMARKS COMPLETED")
    print("="*50)
    
    print("\n[Task A: Landscape vs. Image Ablation Results]")
    print(pd.DataFrame(res_a).T.to_markdown())
    
    print("\n[Task B: Inference Runtime Benchmark Results]")
    print(f"- Average Homology Extraction: {res_b['avg_homology_ms']:.4f} ms/image")
    print(f"- Average Classification: {res_b['avg_classification_ms']:.4f} ms/image")
    
    print("\n[Task C: Resolution Ablation (500 samples)]")
    print(pd.DataFrame(res_c).T.to_markdown())

if __name__ == '__main__':
    main()
