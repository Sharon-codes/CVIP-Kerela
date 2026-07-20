#!/usr/bin/env python
"""
Exhaustive Benchmark Study comparing ML algorithms trained on TDA features.
Author: Principal ML Researcher (Algebraic Topology & Computational Diagnostics)
"""

import sys
import os
import gc
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# 1. Environment Setup & Dependency Check
# =====================================================================
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "tabulate": "tabulate",
    "gtda": "giotto-tda",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "torch": "torch",
    "torchvision": "torchvision",
    "medmnist": "medmnist"
}

print("=" * 70)
print("ALGEBRAIC TOPOLOGY & COMPUTATIONAL DIAGNOSTICS: BENCHMARK STUDY SETUP")
print("=" * 70)
print("Verifying runtime dependencies...")

for module_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        importlib = __import__('importlib')
        importlib.import_module(module_name)
        print(f"  [+] {pip_name} is satisfied.")
    except ImportError:
        print(f"  [-] {pip_name} is missing. Installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"  [+] Successfully installed {pip_name}.")
        except Exception as e:
            print(f"  [!] CRITICAL: Failed to install dependency {pip_name}. Error: {e}")
            sys.exit(1)

# Now import the libraries
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision

from gtda.externals.python import CubicalComplex
from gtda.diagrams import PersistenceLandscape

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_sample_weight

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import medmnist

def load_resnet18():
    import ssl
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass
        
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        return resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        try:
            from torchvision.models import resnet18
            return resnet18(pretrained=True)
        except Exception as e:
            print(f"  [!] Warning: Could not download pre-trained ResNet18 ({e}). Loading with random weights.")
            try:
                from torchvision.models import resnet18
                return resnet18(pretrained=False)
            except Exception:
                from torchvision.models import resnet18
                return resnet18()

def load_mobilenet_v2():
    import ssl
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass
        
    try:
        from torchvision.models import mobilenet_v2, MobileNetV2_Weights
        return mobilenet_v2(weights=MobileNetV2_Weights.DEFAULT)
    except Exception:
        try:
            from torchvision.models import mobilenet_v2
            return mobilenet_v2(pretrained=True)
        except Exception as e:
            print(f"  [!] Warning: Could not download pre-trained MobileNetV2 ({e}). Loading with random weights.")
            try:
                from torchvision.models import mobilenet_v2
                return mobilenet_v2(pretrained=False)
            except Exception:
                from torchvision.models import mobilenet_v2
                return mobilenet_v2()

class PyTorchFusionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

class ImageAutoencoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class MobileNetAutoencoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        if len(z.shape) == 2:
            z = z.unsqueeze(2).unsqueeze(3)
        return self.decoder(z)

def train_medical_weights(X_imgs_train, model_name, epochs=5, batch_size=128):
    """
    Trains a self-supervised Autoencoder on a small subset of the medical images
    to adapt the weights for the grayscale breast cancer modality.
    """
    print(f"  [+] Self-supervised training of {model_name} encoder on radiological slides...", flush=True)
    device = torch.device('cpu')
    
    # 1. Prepare encoder
    if model_name == "resnet18":
        resnet = load_resnet18()
        # Modify conv1 to accept 1 channel grayscale
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder = nn.Sequential(*list(resnet.children())[:-1]) # outputs (B, 512, 1, 1)
        model = ImageAutoencoder(encoder).to(device)
    else: # mobilenet_v2
        mobilenet = load_mobilenet_v2()
        # Modify conv1 to accept 1 channel grayscale
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        class MobileNetEncoder(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.features = m.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
            def forward(self, x):
                x = self.features(x)
                x = self.pool(x)
                return x # shape (B, 1280, 1, 1)
                
        encoder = MobileNetEncoder(mobilenet)
        model = MobileNetAutoencoder(encoder).to(device)
        
    # 2. Select a subset of 500 images to train fast on CPU
    subset_size = min(500, len(X_imgs_train))
    np.random.seed(42)
    indices = np.random.choice(len(X_imgs_train), subset_size, replace=False)
    X_subset = X_imgs_train[indices]
    
    # 3. Prepare data loader
    t_imgs = torch.tensor(X_subset, dtype=torch.float32).unsqueeze(1).to(device) # (B, 1, 64, 64)
    t_imgs = (t_imgs - t_imgs.min()) / (t_imgs.max() - t_imgs.min() + 1e-8)
    
    dataset = torch.utils.data.TensorDataset(t_imgs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Train loop
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    model.train()
    for epoch in range(epochs):
        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            
    # 5. Return the trained encoder
    model.encoder.eval()
    return model.encoder

def extract_cnn_features(X_imgs, model_feat, batch_size=128):
    model_feat.eval()
    n_samples = len(X_imgs)
    features = []
    
    device = torch.device('cpu')
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch = X_imgs[i:batch_end]
            
            # shape (batch_size, 1, 64, 64)
            t_batch = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)
            # Min-Max normalize each image slice to [0, 1] for medical domain consistency
            t_batch = (t_batch - t_batch.min()) / (t_batch.max() - t_batch.min() + 1e-8)
            
            feat = model_feat(t_batch)
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu().numpy())
            
    return np.concatenate(features, axis=0)

def train_predict_fusion_net(X_train, y_train, X_val, epochs=20, batch_size=128, lr=0.001):
    device = torch.device('cpu')
    X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    
    model = PyTorchFusionNet(X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        logits = model(X_val_t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(np.int32)
        
    return preds, probs

def record_metrics(results, roc_curves_data, model_name, y_val, y_pred, y_prob):
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_prob)
    
    results[model_name]["accuracy"].append(acc)
    results[model_name]["precision"].append(prec)
    results[model_name]["recall"].append(rec)
    results[model_name]["f1"].append(f1)
    results[model_name]["roc_auc"].append(auc)
    
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_curves_data[model_name].append((fpr, tpr, auc))

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
        
        # Calculate sigma in pixel units
        dx = (self.max_birth_ - self.min_birth_) / self.n_bins
        sigma_pixels = self.sigma / dx if dx > 0 else 1.0
        sigma_pixels = max(0.5, sigma_pixels)
        ksize = max(3, int(6 * sigma_pixels) | 1)
        
        for i, diag in enumerate(X):
            valid_mask = ~((diag[:, 0] == 0) & (diag[:, 1] == 0))
            valid_diag = diag[valid_mask]
            for dim_idx in [0, 1]:
                pts = valid_diag[valid_diag[:, 2] == dim_idx]
                if len(pts) > 0:
                    b = pts[:, 0]
                    p = pts[:, 1] - pts[:, 0]
                    
                    # 2D Histogram binning weighted by persistence
                    img, _, _ = np.histogram2d(
                        b, p, bins=self.n_bins,
                        range=[[self.min_birth_, self.max_birth_], [self.min_pers_, self.max_pers_]],
                        weights=p
                    )
                    
                    # Apply Gaussian Blur (cv2.GaussianBlur is highly optimized C++ code)
                    out[i, dim_idx] = cv2.GaussianBlur(img.astype(np.float32), (ksize, ksize), sigmaX=sigma_pixels)
                    
        return out.reshape(n_samples, -1)

# Set seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =====================================================================
# 2. Data Ingestion & Preprocessing (Foulproof Test 1)
# =====================================================================
def find_data_dirs(base_path="data"):
    """
    Dynamically searches the data directory to find benign and malignant subfolders.
    Handles spelling variations and typos gracefully.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base data directory '{base_path}' does not exist.")
    
    benign_dir = None
    malignant_dir = None
    
    for d in os.listdir(base_path):
        full_path = os.path.join(base_path, d)
        if os.path.isdir(full_path):
            name_lower = d.lower()
            if "benign" in name_lower:
                benign_dir = full_path
            elif "malign" in name_lower or "maglin" in name_lower:
                malignant_dir = full_path
                
    if not benign_dir or not malignant_dir:
        # Fallback search
        fallback_benign = os.path.join(base_path, "benign")
        fallback_malignant = os.path.join(base_path, "malignant")
        if os.path.isdir(fallback_benign):
            benign_dir = fallback_benign
        if os.path.isdir(fallback_malignant):
            malignant_dir = fallback_malignant
            
    if not benign_dir or not malignant_dir:
        raise ValueError(
            f"Data Ingestion Error: Could not locate both benign and malignant subfolders in '{base_path}'. "
            f"Found folders: {os.listdir(base_path)}"
        )
        
    return benign_dir, malignant_dir


def load_dataset(base_path="data", img_size=(64, 64)):
    """
    Loads images from benign and malignant folders, applies preprocessing:
    1. Grayscale conversion.
    2. Gaussian blur (eliminates high-frequency noise which corrupts topological persistence).
    3. Resize to target dimension.
    
    Returns lists of preprocessed images, labels, and patient ID groups.
    """
    benign_dir, malignant_dir = find_data_dirs(base_path)
    print(f"\nIngesting data from:\n  Benign:    {benign_dir}\n  Malignant: {malignant_dir}")
    
    paths = []
    
    # Class 0: Benign, Class 1: Malignant
    for label, folder in [(0, benign_dir), (1, malignant_dir)]:
        for f in os.listdir(folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                paths.append((os.path.join(folder, f), label, f))
                
    if not paths:
        raise ValueError("Data Integrity Error: No image files found in directories.")
        
    images = []
    valid_labels = []
    groups = []
    corrupted_count = 0
    
    import re
    
    print("\nPreprocessing imagery (Grayscale -> Gaussian Blur -> Otsu Bounding Box Crop -> Resize)...")
    for file_path, label, filename in tqdm(paths, desc="Loading dataset"):
        try:
            # Load in grayscale
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                raise ValueError("Corrupted or unreadable image file.")
                
            # Apply Gaussian Blur (5x5 kernel standard for Otsu's thresholding)
            img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Otsu's Adaptive Thresholding to isolate the dense mass
            _, thresh = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Extract the largest bounding box contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                if w > 5 and h > 5:
                    cropped = img[y:y+h, x:x+w]
                else:
                    cropped = img
            else:
                cropped = img
                
            # Resize cropped region to uniform target dimensions (e.g. 64x64)
            img_resized = cv2.resize(cropped, img_size, interpolation=cv2.INTER_AREA)
            
            # Normalise pixels to [0, 1] for stable cubical filtration
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Parse patient ID from the filename
            match = re.match(r'^(\d+)', filename)
            if match:
                patient_id = match.group(1)
            else:
                patient_id = os.path.splitext(filename)[0]
                
            images.append(img_normalized)
            valid_labels.append(label)
            groups.append(patient_id)
        except Exception as e:
            corrupted_count += 1
            # Warning log for corrupted file
            warnings.warn(f"Skipping corrupted file {file_path}. Reason: {e}")
            
    if corrupted_count > 0:
        print(f"  [!] Warning: Safely skipped {corrupted_count} corrupted/unreadable image files.")
        
    X_imgs = np.array(images)
    y = np.array(valid_labels)
    groups = np.array(groups)
    
    # Assert classes are present
    classes = np.unique(y)
    assert len(classes) == 2, f"Assertion Error: Expected exactly 2 classes (benign, malignant), found {len(classes)}."
    
    benign_count = np.sum(y == 0)
    malignant_count = np.sum(y == 1)
    unique_patients = len(np.unique(groups))
    print(f"Data Integrity Verification:\n  Total loaded: {len(y)}\n  Benign:       {benign_count}\n  Malignant:    {malignant_count}\n  Unique Patients: {unique_patients}")
    
    # Calculate imbalance ratio
    imbalance_ratio = malignant_count / benign_count if benign_count > 0 else 0
    print(f"  Class Imbalance Ratio (Malignant/Benign): {imbalance_ratio:.2f}")
    
    return X_imgs, y, groups

# =====================================================================
# 3. Topological Feature Extraction (Foulproof Test 2)
# =====================================================================
def extract_tda_features(X_imgs, batch_size=500):
    """
    Computes persistent homology (H0 and H1) via gtda's CubicalComplex.
    Vectorizes diagrams using PersistenceLandscape (3 layers, 100 values).
    Processes data in batches and calls gc.collect() to prevent RAM bloat on CPU.
    """
    print("\nExtracting Topological features via Cubical Homology...")
    n_samples = len(X_imgs)
    all_pts_0 = []
    all_pts_1 = []
    
    t0 = time.time()
    
    # Step 1: Compute persistence diagrams in batched chunks
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_imgs = X_imgs[i:batch_end]
        
        for img in batch_imgs:
            # gtda CubicalComplex expects 1D flattened cells and dimensions
            cc = CubicalComplex(dimensions=list(img.shape), top_dimensional_cells=img.flatten())
            persistence_pts = cc.persistence()
            
            # Separate points by homology dimension (0 and 1)
            # Filter out infinite points (e.g. H0 main component which never dies) to prevent NaNs/Infs
            pts_0 = [[b, d, 0] for dim, (b, d) in persistence_pts if dim == 0 and not np.isinf(b) and not np.isinf(d)]
            pts_1 = [[b, d, 1] for dim, (b, d) in persistence_pts if dim == 1 and not np.isinf(b) and not np.isinf(d)]
            
            all_pts_0.append(pts_0)
            all_pts_1.append(pts_1)
            
        # Memory Management: Explicit garbage collection after each batch
        gc.collect()
        
    # Step 2: Pad diagrams per dimension to create uniform shapes for vectorization
    max_0 = max(len(pts) for pts in all_pts_0)
    max_1 = max(len(pts) for pts in all_pts_1)
    
    # Handle edge case where a dimension has zero points across all diagrams
    max_0 = max(max_0, 1)
    max_1 = max(max_1, 1)
    
    padded_diags = []
    for pts_0, pts_1 in zip(all_pts_0, all_pts_1):
        # Pad dimension 0 points to max_0 with zero-persistence points [0.0, 0.0, 0]
        pad_0 = pts_0 + [[0.0, 0.0, 0]] * (max_0 - len(pts_0))
        # Pad dimension 1 points to max_1 with zero-persistence points [0.0, 0.0, 1]
        pad_1 = pts_1 + [[0.0, 0.0, 1]] * (max_1 - len(pts_1))
        padded_diags.append(pad_0 + pad_1)
        
    X_diags = np.array(padded_diags)
    
    # Assertions for safety (ZERO NaNs, Infinite values, or empty arrays)
    assert X_diags.size > 0, "Assertion Error: Generated diagram array is empty."
    assert not np.isnan(X_diags).any(), "Assertion Error: Diagram array contains NaNs."
    assert not np.isinf(X_diags).any(), "Assertion Error: Diagram array contains Infinite values."
    
    t_elapsed = time.time() - t0
    print(f"  [+] Homology extraction complete in {t_elapsed:.2f} seconds.", flush=True)
    print(f"  [+] Diagrams array shape: {X_diags.shape} (H0 max={max_0}, H1 max={max_1})", flush=True)
    
    return X_diags

# =====================================================================
# 4. Multi-Model Benchmarking (Foulproof Test 3)
# =====================================================================
def run_benchmark(X_imgs, X_diags, y, groups):
    """
    Compares:
    - Extra Trees (TDA-Only)
    - ResNet18 + TDA (Extra Trees) Baseline
    - ResNet18 + TDA (Extra Trees) + PCA
    - ResNet18 + TDA (Extra Trees) + L1
    - MobileNetV2 + TDA (Extra Trees) Baseline
    - MobileNetV2 + TDA (Extra Trees) + PCA
    - MobileNetV2 + TDA (Extra Trees) + L1
    Using StratifiedGroupKFold cross-validation split strictly by patient ID.
    Feature reduction (PCA and L1 SVC selection) is fitted strictly inside each fold.
    """
    print("\nConfiguring models and initiating StratifiedGroupKFold Multi-Model Benchmark...", flush=True)
    
    all_model_names = [
        "Extra Trees (TDA-Only)",
        "ResNet18 + TDA (Extra Trees)",
        "ResNet18 + TDA (Extra Trees) + PCA",
        "ResNet18 + TDA (Extra Trees) + L1",
        "MobileNetV2 + TDA (Extra Trees)",
        "MobileNetV2 + TDA (Extra Trees) + PCA",
        "MobileNetV2 + TDA (Extra Trees) + L1"
    ]
    
    results = {name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []} for name in all_model_names}
    roc_curves_data = {name: [] for name in all_model_names}
    
    # 3. Setup CV split
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import LinearSVC
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    total_folds = 5
    t_start = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_diags, y, groups=groups)):
        fold_t0 = time.time()
        print(f"\nProcessing Fold {fold+1}/5...", flush=True)
        
        # Split inputs
        X_diags_train, X_diags_val = X_diags[train_idx], X_diags[val_idx]
        X_imgs_train, X_imgs_val = X_imgs[train_idx], X_imgs[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # A. Fit landscapes and persistence images (TDA vectorization)
        print("  Evaluating topological persistence (Landscapes + Persistence Images)...", flush=True)
        pl = PersistenceLandscape(n_layers=3, n_values=100, n_jobs=-1)
        X_train_pl_flat = pl.fit_transform(X_diags_train).reshape(len(train_idx), -1)
        X_val_pl_flat = pl.transform(X_diags_val).reshape(len(val_idx), -1)
        
        pi = PersistenceImage(sigma=0.1, n_bins=50)
        X_train_pi_flat = pi.fit_transform(X_diags_train)
        X_val_pi_flat = pi.transform(X_diags_val)
        
        X_train_tda = np.hstack([X_train_pl_flat, X_train_pi_flat])
        X_val_tda = np.hstack([X_val_pl_flat, X_val_pi_flat])
        
        # B. Extract Medical domain weights inside fold via Autoencoders
        resnet_feat = train_medical_weights(X_imgs_train, "resnet18", epochs=5)
        X_train_resnet = extract_cnn_features(X_imgs_train, resnet_feat)
        X_val_resnet = extract_cnn_features(X_imgs_val, resnet_feat)
        
        mobilenet_feat = train_medical_weights(X_imgs_train, "mobilenet_v2", epochs=5)
        X_train_mobilenet = extract_cnn_features(X_imgs_train, mobilenet_feat)
        X_val_mobilenet = extract_cnn_features(X_imgs_val, mobilenet_feat)
        
        # C. Concatenate features for hybrids
        X_train_resnet_tda = np.hstack([X_train_resnet, X_train_tda])
        X_val_resnet_tda = np.hstack([X_val_resnet, X_val_tda])
        
        X_train_mobilenet_tda = np.hstack([X_train_mobilenet, X_train_tda])
        X_val_mobilenet_tda = np.hstack([X_val_mobilenet, X_val_tda])
        
        # D. Standard Scaling (dynamic fit strictly on train fold)
        scaler_tda = StandardScaler()
        X_train_tda_scaled = scaler_tda.fit_transform(X_train_tda)
        X_val_tda_scaled = scaler_tda.transform(X_val_tda)
        
        scaler_res = StandardScaler()
        X_train_res_scaled = scaler_res.fit_transform(X_train_resnet_tda)
        X_val_res_scaled = scaler_res.transform(X_val_resnet_tda)
        
        scaler_mob = StandardScaler()
        X_train_mob_scaled = scaler_mob.fit_transform(X_train_mobilenet_tda)
        X_val_mob_scaled = scaler_mob.transform(X_val_mobilenet_tda)
        
        # E. Fit Classifiers and Record Metrics
        # 1. Extra Trees (TDA-Only)
        print("  Fitting Extra Trees (TDA-Only)...", flush=True)
        clf = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train_tda_scaled, y_train)
        y_pred = clf.predict(X_val_tda_scaled)
        y_prob = clf.predict_proba(X_val_tda_scaled)[:, 1]
        record_metrics(results, roc_curves_data, "Extra Trees (TDA-Only)", y_val, y_pred, y_prob)
        
        # 2. ResNet18 + TDA Hybrid Track
        print("  Fitting ResNet18 + TDA models (Baseline, PCA, L1)...", flush=True)
        # 2.1 Baseline
        clf = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train_res_scaled, y_train)
        y_pred = clf.predict(X_val_res_scaled)
        y_prob = clf.predict_proba(X_val_res_scaled)[:, 1]
        record_metrics(results, roc_curves_data, "ResNet18 + TDA (Extra Trees)", y_val, y_pred, y_prob)
        
        # 2.2 PCA Track (100 components)
        pca_res = PCA(n_components=100, random_state=RANDOM_STATE)
        X_train_res_pca = pca_res.fit_transform(X_train_res_scaled)
        X_val_res_pca = pca_res.transform(X_val_res_scaled)
        
        clf = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train_res_pca, y_train)
        y_pred = clf.predict(X_val_res_pca)
        y_prob = clf.predict_proba(X_val_res_pca)[:, 1]
        record_metrics(results, roc_curves_data, "ResNet18 + TDA (Extra Trees) + PCA", y_val, y_pred, y_prob)
        
        # 2.3 L1 Selection Track
        selector_res = SelectFromModel(
            LinearSVC(penalty='l1', dual=False, C=0.01, random_state=RANDOM_STATE),
            prefit=False
        )
        X_train_res_l1 = selector_res.fit_transform(X_train_res_scaled, y_train)
        X_val_res_l1 = selector_res.transform(X_val_res_scaled)
        
        clf = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train_res_l1, y_train)
        y_pred = clf.predict(X_val_res_l1)
        y_prob = clf.predict_proba(X_val_res_l1)[:, 1]
        record_metrics(results, roc_curves_data, "ResNet18 + TDA (Extra Trees) + L1", y_val, y_pred, y_prob)
        
        # 3. MobileNetV2 + TDA Hybrid Track
        print("  Fitting MobileNetV2 + TDA models (Baseline, PCA, L1)...", flush=True)
        # 3.1 Baseline
        clf = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train_mob_scaled, y_train)
        y_pred = clf.predict(X_val_mob_scaled)
        y_prob = clf.predict_proba(X_val_mob_scaled)[:, 1]
        record_metrics(results, roc_curves_data, "MobileNetV2 + TDA (Extra Trees)", y_val, y_pred, y_prob)
        
        # 3.2 PCA Track (100 components)
        pca_mob = PCA(n_components=100, random_state=RANDOM_STATE)
        X_train_mob_pca = pca_mob.fit_transform(X_train_mob_scaled)
        X_val_mob_pca = pca_mob.transform(X_val_mob_scaled)
        
        clf = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train_mob_pca, y_train)
        y_pred = clf.predict(X_val_mob_pca)
        y_prob = clf.predict_proba(X_val_mob_pca)[:, 1]
        record_metrics(results, roc_curves_data, "MobileNetV2 + TDA (Extra Trees) + PCA", y_val, y_pred, y_prob)
        
        # 3.3 L1 Selection Track
        selector_mob = SelectFromModel(
            LinearSVC(penalty='l1', dual=False, C=0.01, random_state=RANDOM_STATE),
            prefit=False
        )
        X_train_mob_l1 = selector_mob.fit_transform(X_train_mob_scaled, y_train)
        X_val_mob_l1 = selector_mob.transform(X_val_mob_scaled)
        
        clf = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train_mob_l1, y_train)
        y_pred = clf.predict(X_val_mob_l1)
        y_prob = clf.predict_proba(X_val_mob_l1)[:, 1]
        record_metrics(results, roc_curves_data, "MobileNetV2 + TDA (Extra Trees) + L1", y_val, y_pred, y_prob)
        
        # Fold complete eta report
        t_elapsed = time.time() - t_start
        avg_fold_time = t_elapsed / (fold + 1)
        remaining_folds = total_folds - (fold + 1)
        eta = avg_fold_time * remaining_folds
        eta_min, eta_sec = divmod(int(eta), 60)
        print(f"  [Fold {fold+1}/5] Completed. Time: {time.time() - fold_t0:.2f}s. ETA: {eta_min:02d}:{eta_sec:02d}", flush=True)
        
    # Convergence and Overfitting Check
    summary_data = []
    print("\n" + "=" * 70)
    print("CONVERGENCE & STATISTICAL OVERFITTING CHECK")
    print("=" * 70)
    
    for model_name, metrics in results.items():
        model_summary = {"Model": model_name}
        for metric_name, values in metrics.items():
            if not values:
                model_summary[f"{metric_name}_mean"] = np.nan
                model_summary[f"{metric_name}_std"] = np.nan
                continue
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            model_summary[f"{metric_name}_mean"] = mean_val
            model_summary[f"{metric_name}_std"] = std_val
            
            # Warning if standard deviation is higher than 0.05
            if std_val > 0.05:
                print(
                    f"  [!] WARNING: High variance detected in {model_name} for '{metric_name}' metric! "
                    f"Mean: {mean_val:.4f}, Std: {std_val:.4f}. Model may be unstable or overfitting."
                )
                
        summary_data.append(model_summary)
        
    summary_df = pd.DataFrame(summary_data)
    return summary_df, roc_curves_data

# =====================================================================
# 5. Visualizations & Deliverables (Foulproof Test 4)
# =====================================================================
def generate_visualizations(summary_df, roc_curves_data, X_imgs, y, X_diags, landscapes):
    """
    Generates and saves four high-resolution graphics at 300 DPI in the images/ directory:
    Plot 1: Side-by-side Comparative Persistence Diagram (Benign vs Malignant).
    Plot 2: Average Persistence Landscape Waveforms (Layer 0, H0 and H1).
    Plot 3: Multi-Model ROC Curve comparison.
    Plot 4: Summary Performance Heatmap.
    """
    os.makedirs("images", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # -----------------------------------------------------------------
    # Plot 1: Side-by-side comparative Persistence Diagram
    # -----------------------------------------------------------------
    print("\nGenerating Plot 1: Comparative Persistence Diagrams...")
    # Find first sample of each class
    idx_benign = np.where(y == 0)[0][0]
    idx_malignant = np.where(y == 1)[0][0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    
    for ax, idx, title in zip(axes, [idx_benign, idx_malignant], ["Benign Tumor (Sample)", "Malignant Tumor (Sample)"]):
        # Extract diagram points (birth, death, dimension)
        diag = X_diags[idx]
        # Filter padding (birth == death == 0)
        valid_mask = ~((diag[:, 0] == 0) & (diag[:, 1] == 0))
        diag = diag[valid_mask]
        
        # Split by homology dimension
        diag_h0 = diag[diag[:, 2] == 0]
        diag_h1 = diag[diag[:, 2] == 1]
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7, label="Diagonal (Birth=Death)")
        
        # Scatter H0 and H1 points
        ax.scatter(diag_h0[:, 0], diag_h0[:, 1], color="darkorange", marker="o", s=40, alpha=0.8, label="$H_0$ (Connected Components)")
        ax.scatter(diag_h1[:, 0], diag_h1[:, 1], color="royalblue", marker="^", s=45, alpha=0.8, label="$H_1$ (Loops/Tunnels)")
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Birth Filtration Value", fontsize=12)
        ax.set_ylabel("Death Filtration Value", fontsize=12)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
    plt.tight_layout()
    plot1_path = "images/comparative_persistence_diagram.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # -----------------------------------------------------------------
    # Plot 2: Average Persistence Landscape Waveforms
    # -----------------------------------------------------------------
    print("Generating Plot 2: Average Persistence Landscapes...")
    # landscapes shape: (n_samples, n_homology_dims, n_layers, n_values) = (n_samples, 2, 3, 100)
    # H0 is index 0, H1 is index 1. Layer 0 is index 0.
    # Compute mean and standard deviation waveforms
    mean_h0_b = np.mean(landscapes[y == 0, 0, 0, :], axis=0)
    std_h0_b = np.std(landscapes[y == 0, 0, 0, :], axis=0)
    mean_h0_m = np.mean(landscapes[y == 1, 0, 0, :], axis=0)
    std_h0_m = np.std(landscapes[y == 1, 0, 0, :], axis=0)
    
    mean_h1_b = np.mean(landscapes[y == 0, 1, 0, :], axis=0)
    std_h1_b = np.std(landscapes[y == 0, 1, 0, :], axis=0)
    mean_h1_m = np.mean(landscapes[y == 1, 1, 0, :], axis=0)
    std_h1_m = np.std(landscapes[y == 1, 1, 0, :], axis=0)
    
    x_grid = np.linspace(0, 1, 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # H0 Waveform
    axes[0].plot(x_grid, mean_h0_b, color="royalblue", linewidth=2.5, label="Benign (Mean)")
    axes[0].fill_between(x_grid, mean_h0_b - std_h0_b, mean_h0_b + std_h0_b, color="royalblue", alpha=0.15)
    axes[0].plot(x_grid, mean_h0_m, color="crimson", linewidth=2.5, label="Malignant (Mean)")
    axes[0].fill_between(x_grid, mean_h0_m - std_h0_m, mean_h0_m + std_h0_m, color="crimson", alpha=0.15)
    axes[0].set_title("Average $H_0$ Landscape (Layer 0)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Filtration Parameter $t$", fontsize=11)
    axes[0].set_ylabel("Landscape Value $\Lambda_1(t)$", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=10)
    
    # H1 Waveform
    axes[1].plot(x_grid, mean_h1_b, color="royalblue", linewidth=2.5, label="Benign (Mean)")
    axes[1].fill_between(x_grid, mean_h1_b - std_h1_b, mean_h1_b + std_h1_b, color="royalblue", alpha=0.15)
    axes[1].plot(x_grid, mean_h1_m, color="crimson", linewidth=2.5, label="Malignant (Mean)")
    axes[1].fill_between(x_grid, mean_h1_m - std_h1_m, mean_h1_m + std_h1_m, color="crimson", alpha=0.15)
    axes[1].set_title("Average $H_1$ Landscape (Layer 0)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Filtration Parameter $t$", fontsize=11)
    axes[1].set_ylabel("Landscape Value $\Lambda_1(t)$", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=10)
    
    plt.tight_layout()
    plot2_path = "images/average_persistence_landscape.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # -----------------------------------------------------------------
    # Plot 3: Multi-Model ROC Curve Comparison
    # -----------------------------------------------------------------
    print("Generating Plot 3: Multi-Model ROC Curves...")
    plt.figure(figsize=(10, 8))
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    
    for idx, (model_name, curves) in enumerate(roc_curves_data.items()):
        if not curves:
            continue
        
        # Interpolate curves to compute a smooth average ROC line
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        
        for fpr, tpr, auc in curves:
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(auc)
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        plt.plot(
            mean_fpr, mean_tpr,
            color=colors[idx % len(colors)],
            linewidth=2.5,
            label=f"{model_name} (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})"
        )
        
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, label="Chance Line (AUC = 0.50)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title("Multi-Model ROC Curve Comparison (Topological Features)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10.5)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plot3_path = "images/multimodel_roc_curve.png"
    plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # -----------------------------------------------------------------
    # Plot 4: Summary Performance Heatmap
    # -----------------------------------------------------------------
    print("Generating Plot 4: Summary Performance Matrix...")
    # Extract only the means for the heatmap
    heatmap_df = summary_df.set_index("Model")
    mean_cols = [c for c in heatmap_df.columns if c.endswith("_mean")]
    heatmap_mean = heatmap_df[mean_cols]
    
    # Rename columns for presentation
    clean_cols = [c.replace("_mean", "").upper() for c in mean_cols]
    heatmap_mean.columns = clean_cols
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_mean,
        annot=True,
        cmap="viridis",
        fmt=".4f",
        linewidths=0.5,
        cbar_kws={'label': 'Performance Score'}
    )
    plt.title("Model Performance Comparison (Mean across 5 Folds)", fontsize=14, fontweight="bold")
    plt.ylabel("Classifier Model", fontsize=12)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plot4_path = "images/performance_comparison_heatmap.png"
    plt.savefig(plot4_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Assert all four files are created successfully
    assert os.path.exists(plot1_path), "Visualisation Error: comparative_persistence_diagram.png was not written."
    assert os.path.exists(plot2_path), "Visualisation Error: average_persistence_landscape.png was not written."
    assert os.path.exists(plot3_path), "Visualisation Error: multimodel_roc_curve.png was not written."
    assert os.path.exists(plot4_path), "Visualisation Error: performance_comparison_heatmap.png was not written."
    
    print("  [+] All four high-resolution graphics successfully written to images/ directory at 300 DPI.\n")

# =====================================================================
# Main Execution Block
# =====================================================================
def main():
    start_time = time.time()
    
    # 1. Load data
    try:
        X_imgs, y, groups = load_dataset("data", img_size=(64, 64))
    except Exception as e:
        print(f"\n[!] Fatal Error loading dataset: {e}")
        sys.exit(1)
        
    # 2. Extract topological features (Diagrams only)
    try:
        X_diags = extract_tda_features(X_imgs)
    except Exception as e:
        print(f"\n[!] Fatal Error during topological feature extraction: {e}")
        sys.exit(1)
        
    # 3. Benchmarking (Scalers, CNN features and Landscapes calculated per fold, split by patient groups)
    try:
        summary_df, roc_curves_data = run_benchmark(X_imgs, X_diags, y, groups)
    except Exception as e:
        print(f"\n[!] Fatal Error during multi-model benchmarking: {e}")
        sys.exit(1)
        
    # 4. Generate landscapes globally ONLY for visualization of waveforms
    try:
        print("\nComputing global persistence landscapes for visualization only...", flush=True)
        pl_global = PersistenceLandscape(n_layers=3, n_values=100, n_jobs=-1)
        landscapes_global = pl_global.fit_transform(X_diags)
    except Exception as e:
        print(f"\n[!] Fatal Error during global landscape calculation for visualization: {e}")
        sys.exit(1)
        
    # 5. Save visuals
    try:
        generate_visualizations(summary_df, roc_curves_data, X_imgs, y, X_diags, landscapes_global)
    except Exception as e:
        print(f"\n[!] Fatal Error during visualization generation: {e}")
        sys.exit(1)
        
    # 5. Output Markdown Results Table
    print("=" * 70)
    print("PUBLICATION-GRADE BENCHMARK PERFORMANCE TABLE")
    print("=" * 70)
    
    # Formatting columns
    markdown_df = pd.DataFrame()
    markdown_df["Model"] = summary_df["Model"]
    
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for m in metrics:
        markdown_df[m.upper()] = summary_df.apply(
            lambda r: f"{r[m+'_mean']:.4f} \u00b1 {r[m+'_std']:.4f}", axis=1
        )
        
    print(markdown_df.to_markdown(index=False))
    print("\n" + "=" * 70)
    print(f"Pipeline executed successfully. Total execution time: {time.time() - start_time:.2f} seconds.")
    print("=" * 70)


if __name__ == '__main__':
    main()
