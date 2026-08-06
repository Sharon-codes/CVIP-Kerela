#!/usr/bin/env python
"""
Core Pipeline Module: Real Data Ingestion, Patient-Isolated Grouping,
MobileNetV2 Spatial Vector Extraction, Cubical Homology TDA Vectorization,
and Model Building.

Author: Lead Biomedical ML Engineer (Q1 Journal Submission Suite)
"""

import os
import sys
import re
import gc
import shutil
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

from gtda.externals.python import CubicalComplex
from gtda.diagrams import PersistenceLandscape

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=UserWarning)


# =====================================================================
# 1. Dataset Partitioning & Real Data Ingestion
# =====================================================================
def setup_data_split(base_dir="data"):
    """
    Ensures data/primary and data/external exist and are populated with real medical images.
    If not present, partitions the raw dataset into Primary (80% patients) and External (20% patients)
    strictly by Patient ID to ensure zero data leakage.
    """
    primary_dir = os.path.join(base_dir, "primary")
    external_dir = os.path.join(base_dir, "external")

    if os.path.exists(primary_dir) and os.path.exists(external_dir):
        print("[+] Primary and External dataset splits already exist.")
        return primary_dir, external_dir

    benign_src = os.path.join(base_dir, "BreastCancer_Benign")
    malignant_src = os.path.join(base_dir, "BreastCancer_Maglinant")

    if not os.path.exists(benign_src) or not os.path.exists(malignant_src):
        # Fallback search
        for d in os.listdir(base_dir):
            full = os.path.join(base_dir, d)
            if os.path.isdir(full):
                if "benign" in d.lower():
                    benign_src = full
                elif "maglin" in d.lower() or "malign" in d.lower():
                    malignant_src = full

    print("[+] Partitioning real medical imagery into primary (80%) and external (20%) patient splits...")

    for parent in [primary_dir, external_dir]:
        os.makedirs(os.path.join(parent, "benign"), exist_ok=True)
        os.makedirs(os.path.join(parent, "malignant"), exist_ok=True)

    for label_str, src_path in [("benign", benign_src), ("malignant", malignant_src)]:
        files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # Group files by Patient ID
        patient_map = {}
        for f in files:
            match = re.match(r'^(\d+)', f)
            pid = match.group(1) if match else os.path.splitext(f)[0]
            if pid not in patient_map:
                patient_map[pid] = []
            patient_map[pid].append(f)

        patient_ids = sorted(list(patient_map.keys()))
        np.random.seed(RANDOM_STATE)
        np.random.shuffle(patient_ids)

        n_ext = max(1, int(0.20 * len(patient_ids)))
        ext_pids = set(patient_ids[:n_ext])
        prim_pids = set(patient_ids[n_ext:])

        for pid in prim_pids:
            for f in patient_map[pid]:
                shutil.copy2(os.path.join(src_path, f), os.path.join(primary_dir, label_str, f))

        for pid in ext_pids:
            for f in patient_map[pid]:
                shutil.copy2(os.path.join(src_path, f), os.path.join(external_dir, label_str, f))

    print(f"  [+] Created {primary_dir} and {external_dir} successfully.")
    return primary_dir, external_dir


def load_real_images(data_dir, img_size=(64, 64)):
    """
    Loads real radiological images from disk recursively via os.walk, applies Otsu thresholding ROI crop,
    parses Patient IDs for leakage-free CV grouping.
    Returns: X_imgs (N, 64, 64), y (N,), groups (N,), roi_boxes, orig_imgs
    """
    images = []
    labels = []
    groups = []
    roi_boxes = []
    orig_imgs = []

    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                full = os.path.join(root, f)
                label = 1 if ('malignant' in root.lower() or 'maglinant' in root.lower()) else 0
                paths.append((full, label, f))

    paths = sorted(paths, key=lambda x: x[2])
    print(f"  [+] Loading real medical imagery from {data_dir} ({len(paths)} images)...")

    for file_path, label, filename in tqdm(paths, desc="Ingesting ROI Imagery", leave=False):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            continue

        orig_imgs.append(img.copy())

        # Otsu Adaptive Thresholding to extract ROI
        img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            if w > 5 and h > 5:
                cropped = img[y:y+h, x:x+w]
                roi_boxes.append((x, y, w, h))
            else:
                cropped = img
                roi_boxes.append((0, 0, img.shape[1], img.shape[0]))
        else:
            cropped = img
            roi_boxes.append((0, 0, img.shape[1], img.shape[0]))

        # Resize and normalize
        img_resized = cv2.resize(cropped, img_size, interpolation=cv2.INTER_AREA)
        img_norm = img_resized.astype(np.float32) / 255.0

        # Parse Patient ID
        match = re.match(r'^(\d+)', filename)
        patient_id = match.group(1) if match else os.path.splitext(filename)[0]

        images.append(img_norm)
        labels.append(label)
        groups.append(patient_id)

    X_imgs = np.array(images)
    y = np.array(labels)
    groups = np.array(groups)

    print(f"  [+] Loaded {len(y)} real images | Benign: {np.sum(y==0)} | Malignant: {np.sum(y==1)} | Patients: {len(np.unique(groups))}")
    return X_imgs, y, groups, roi_boxes, orig_imgs


# =====================================================================
# 2. Persistence Image Transformer Class
# =====================================================================
class PersistenceImage(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=0.1, n_bins=50):
        self.sigma = sigma
        self.n_bins = n_bins

    def fit(self, X, y=None):
        valid_mask = ~((X[:, :, 0] == 0) & (X[:, :, 1] == 0))
        if np.any(valid_mask):
            births = X[:, :, 0][valid_mask]
            persistences = (X[:, :, 1] - X[:, :, 0])[valid_mask]
            self.min_birth_ = np.min(births)
            self.max_birth_ = np.max(births)
            self.min_pers_ = np.min(persistences)
            self.max_pers_ = np.max(persistences)
        else:
            self.min_birth_, self.max_birth_ = 0.0, 1.0
            self.min_pers_, self.max_pers_ = 0.0, 1.0

        if self.max_birth_ == self.min_birth_:
            self.max_birth_ += 1e-5
        if self.max_pers_ == self.min_pers_:
            self.max_pers_ += 1e-5
        return self

    def transform(self, X):
        n_samples = len(X)
        out = np.zeros((n_samples, 2, self.n_bins, self.n_bins), dtype=np.float32)
        dx = (self.max_birth_ - self.min_birth_) / self.n_bins
        sigma_pixels = max(0.5, self.sigma / dx if dx > 0 else 1.0)
        ksize = max(3, int(6 * sigma_pixels) | 1)

        for i, diag in enumerate(X):
            valid_mask = ~((diag[:, 0] == 0) & (diag[:, 1] == 0))
            valid_diag = diag[valid_mask]
            for dim_idx in [0, 1]:
                pts = valid_diag[valid_diag[:, 2] == dim_idx]
                if len(pts) > 0:
                    b = pts[:, 0]
                    p = pts[:, 1] - pts[:, 0]
                    img, _, _ = np.histogram2d(
                        b, p, bins=self.n_bins,
                        range=[[self.min_birth_, self.max_birth_], [self.min_pers_, self.max_pers_]],
                        weights=p
                    )
                    out[i, dim_idx] = cv2.GaussianBlur(img.astype(np.float32), (ksize, ksize), sigmaX=sigma_pixels)

        return out.reshape(n_samples, -1)


# =====================================================================
# 3. MobileNetV2 Spatial Extraction (512-D)
# =====================================================================
class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        try:
            mobilenet = models.mobilenet_v2(weights=models.MobileNetV2_Weights.DEFAULT)
        except Exception:
            mobilenet = models.mobilenet_v2(pretrained=True)
            
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def extract_cnn(X_imgs, batch_size=128, out_dim=512):
    """
    Passes ROI images through pre-trained MobileNetV2 to extract a 512-D spatial vector per sample.
    Converts 1-channel grayscale to 3 channels and applies standard ImageNet normalization.
    """
    device = torch.device("cpu")
    model = MobileNetV2FeatureExtractor(out_dim=out_dim).to(device)
    model.eval()

    n_samples = len(X_imgs)
    features = []

    # ImageNet mean and std tensors
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(device)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = X_imgs[i:i+batch_size]
            # Convert 1-channel grayscale (B, H, W) to 3-channel tensor (B, 3, H, W)
            t_batch = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
            # Apply standard ImageNet normalization
            t_batch = (t_batch - mean) / std
            feat = model(t_batch)
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


# =====================================================================
# 4. Cubical Homology TDA Feature Extraction (5,600-D)
# =====================================================================
def extract_tda(X_imgs, batch_size=500):
    """
    Computes persistent homology (H0, H1) via gtda's CubicalComplex.
    Extracts and flattens Persistence Landscapes (3 layers, 100 values) and Persistence Images (2x50x50).
    Returns concatenated 5,600-D topological vector.
    """
    n_samples = len(X_imgs)
    all_pts_0 = []
    all_pts_1 = []

    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_imgs = X_imgs[i:batch_end]

        for img in batch_imgs:
            cc = CubicalComplex(dimensions=list(img.shape), top_dimensional_cells=img.flatten())
            persistence_pts = cc.persistence()

            pts_0 = [[b, d, 0] for dim, (b, d) in persistence_pts if dim == 0 and not np.isinf(b) and not np.isinf(d)]
            pts_1 = [[b, d, 1] for dim, (b, d) in persistence_pts if dim == 1 and not np.isinf(b) and not np.isinf(d)]

            all_pts_0.append(pts_0)
            all_pts_1.append(pts_1)

        gc.collect()

    max_0 = max(len(pts) for pts in all_pts_0) if all_pts_0 else 1
    max_1 = max(len(pts) for pts in all_pts_1) if all_pts_1 else 1

    max_0 = max(max_0, 1)
    max_1 = max(max_1, 1)

    padded_diags = []
    for pts_0, pts_1 in zip(all_pts_0, all_pts_1):
        pad_0 = pts_0 + [[0.0, 0.0, 0]] * (max_0 - len(pts_0))
        pad_1 = pts_1 + [[0.0, 0.0, 1]] * (max_1 - len(pts_1))
        padded_diags.append(pad_0 + pad_1)

    diagrams = np.array(padded_diags, dtype=np.float32)

    # Persistence Landscapes (3 layers, 100 values)
    pl = PersistenceLandscape(n_layers=3, n_values=100, n_jobs=-1)
    X_pl_flat = pl.fit_transform(diagrams).reshape(n_samples, -1)

    # Persistence Images (2x50x50 = 5000)
    pi = PersistenceImage(sigma=0.1, n_bins=50)
    X_pi_flat = pi.fit_transform(diagrams)

    X_tda = np.hstack([X_pl_flat, X_pi_flat])
    return X_tda


# =====================================================================
# 5. Classifier Builders
# =====================================================================
def build_models(n_components=50, random_state=RANDOM_STATE):
    """
    Initializes CNN-only classifier and Hybrid pipeline wrapped with StandardScaler, L1 Lasso selection, PCA, and balanced ExtraTrees.
    """
    cnn_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', ExtraTreesClassifier(
            n_estimators=500,
            class_weight='balanced',
            max_features='sqrt',
            min_samples_leaf=8,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    hybrid_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('l1_select', SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', C=0.01, random_state=random_state, class_weight='balanced')
        )),
        ('pca', PCA(n_components=n_components, random_state=random_state)),
        ('clf', ExtraTreesClassifier(
            n_estimators=500,
            class_weight='balanced',
            max_features='sqrt',
            min_samples_leaf=8,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    return cnn_model, hybrid_pipeline
