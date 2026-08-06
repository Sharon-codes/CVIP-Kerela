#!/usr/bin/env python
"""
Experiment 1: Lightweight CNN Comparison & Edge-AI Resource Profiling
Compares Hybrid CNN-TDA-SVM against standalone lightweight architectures:
- MobileNetV2 (CNN only)
- ResNet18 (CNN only)
- EfficientNet-B0 (CNN only)

Outputs: Internal AUC, External AUC, External Accuracy, CPU Latency (ms/img), Peak RAM (MB).

Author: Lead Biomedical ML Engineer (IEEE Q1 Submission Suite)
"""

import os
import sys
import time
import tracemalloc
import warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix, f1_score

from core_pipeline import setup_data_split, load_real_images, extract_cnn, extract_tda

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Feature Extractors for ResNet18 and EfficientNet-B0
# ---------------------------------------------------------------------
class ResNet18FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = tv_models.ResNet18_Weights.DEFAULT
        model = tv_models.resnet18(weights=weights)
        # Remove classification head to output 512-D spatial features
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def extract(self, images_np):
        feats = []
        batch_size = 64
        n_samples = len(images_np)
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_imgs = images_np[i:i+batch_size]
                tensors = torch.from_numpy(batch_imgs).float().unsqueeze(1).repeat(1, 3, 1, 1).to(self.device)
                tensors = (tensors - self.mean) / self.std
                out = self.model(tensors)
                feats.append(out.cpu().numpy())
        return np.vstack(feats)


class EfficientNetB0FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT
        model = tv_models.efficientnet_b0(weights=weights)
        # Remove classifier head to output 1280-D spatial features
        model.classifier = nn.Identity()
        model.eval()
        self.model = model.to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def extract(self, images_np):
        feats = []
        batch_size = 64
        n_samples = len(images_np)
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_imgs = images_np[i:i+batch_size]
                tensors = torch.from_numpy(batch_imgs).float().unsqueeze(1).repeat(1, 3, 1, 1).to(self.device)
                tensors = (tensors - self.mean) / self.std
                out = self.model(tensors)
                feats.append(out.cpu().numpy())
        return np.vstack(feats)


def run_baselines_experiment():
    print("=" * 80)
    print("EXPERIMENT 1: LIGHTWEIGHT CNN COMPARISON & EDGE-AI PROFILING")
    print("=" * 80)

    setup_data_split()

    print("\n[+] Loading Real Images...")
    X_imgs_prim, y_prim, groups_prim, _, _ = load_real_images("data/primary")
    X_imgs_ext, y_ext, _, _, _ = load_real_images("data/external")

    # 1. Feature Extraction across Architectures
    print("\n[+] Extracting MobileNetV2 Features (512-D)...")
    X_mobilenet_prim = extract_cnn(X_imgs_prim)
    X_mobilenet_ext = extract_cnn(X_imgs_ext)

    print("\n[+] Extracting ResNet18 Features (512-D)...")
    resnet_extractor = ResNet18FeatureExtractor()
    X_resnet_prim = resnet_extractor.extract(X_imgs_prim)
    X_resnet_ext = resnet_extractor.extract(X_imgs_ext)

    print("\n[+] Extracting EfficientNet-B0 Features (1280-D)...")
    effnet_extractor = EfficientNetB0FeatureExtractor()
    X_effnet_prim = effnet_extractor.extract(X_imgs_prim)
    X_effnet_ext = effnet_extractor.extract(X_imgs_ext)

    print("\n[+] Extracting Topological Features (Cubical Homology 5600-D)...")
    X_tda_prim = extract_tda(X_imgs_prim)
    X_tda_ext = extract_tda(X_imgs_ext)

    X_hybrid_prim = np.hstack([X_tda_prim, X_mobilenet_prim])
    X_hybrid_ext = np.hstack([X_tda_ext, X_mobilenet_ext])

    # 2. Define Pipelines
    pipelines = {
        "MobileNetV2 (CNN Only)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE))
        ]),
        "ResNet18 (CNN Only)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE))
        ]),
        "EfficientNet-B0 (CNN Only)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE))
        ]),
        "Hybrid CNN-TDA (Final Pipeline)": Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(
                LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE, C=0.1)
            )),
            ('pca', PCA(n_components=120, random_state=RANDOM_STATE)),
            ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, C=10.0, gamma='scale', random_state=RANDOM_STATE))
        ])
    }

    feature_datasets = {
        "MobileNetV2 (CNN Only)": (X_mobilenet_prim, X_mobilenet_ext),
        "ResNet18 (CNN Only)": (X_resnet_prim, X_resnet_ext),
        "EfficientNet-B0 (CNN Only)": (X_effnet_prim, X_effnet_ext),
        "Hybrid CNN-TDA (Final Pipeline)": (X_hybrid_prim, X_hybrid_ext)
    }

    # 5-fold StratifiedGroupKFold Split
    sgkf = StratifiedGroupKFold(n_splits=5)
    splits = list(sgkf.split(X_imgs_prim, y_prim, groups=groups_prim))
    train_idx, val_idx = splits[0]

    baseline_results = []
    saved_probs = {}

    for name, pipe in pipelines.items():
        X_tr, X_te = feature_datasets[name]

        X_train, y_train = X_tr[train_idx], y_prim[train_idx]
        X_val, y_val = X_tr[val_idx], y_prim[val_idx]

        pipe.fit(X_train, y_train)

        # Internal Val Evaluation
        y_val_prob = pipe.predict_proba(X_val)[:, 1]
        int_auc = roc_auc_score(y_val, y_val_prob)

        # External Cohort Evaluation
        y_ext_prob = pipe.predict_proba(X_te)[:, 1]
        ext_auc = roc_auc_score(y_ext, y_ext_prob)
        ext_pred = (y_ext_prob >= 0.5).astype(int)
        ext_acc = accuracy_score(y_ext, ext_pred)
        ext_sens = recall_score(y_ext, ext_pred)
        tn, fp, fn, tp = confusion_matrix(y_ext, ext_pred).ravel()
        ext_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        saved_probs[name] = {
            "y_val": y_val,
            "y_val_prob": y_val_prob,
            "y_ext": y_ext,
            "y_ext_prob": y_ext_prob,
            "pipe": pipe
        }

        # Profile CPU Latency & Peak RAM
        sample_indices = np.random.choice(len(X_val), min(100, len(X_val)), replace=False)
        X_sample = X_val[sample_indices]

        tracemalloc.start()
        t0 = time.perf_counter()
        _ = pipe.predict_proba(X_sample)
        t1 = time.perf_counter()
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        latency_ms = ((t1 - t0) / len(X_sample)) * 1000.0
        peak_ram_mb = peak_mem_bytes / (1024.0 * 1024.0)

        print(f"\n[+] {name}:")
        print(f"    Internal ROC-AUC : {int_auc:.4f}")
        print(f"    External ROC-AUC : {ext_auc:.4f}")
        print(f"    External Accuracy: {ext_acc:.4f} (Sens: {ext_sens:.4f}, Spec: {ext_spec:.4f})")
        print(f"    CPU Latency      : {latency_ms:.2f} ms/img")
        print(f"    Peak RAM         : {peak_ram_mb:.2f} MB")

        baseline_results.append({
            "Architecture": name,
            "Internal_AUC": int_auc,
            "External_AUC": ext_auc,
            "External_Acc": ext_acc,
            "External_Sens": ext_sens,
            "External_Spec": ext_spec,
            "Latency_ms": latency_ms,
            "Peak_RAM_MB": peak_ram_mb
        })

    return baseline_results, saved_probs


if __name__ == "__main__":
    run_baselines_experiment()
