# Hybrid CNN-TDA Framework for Breast Cancer Pathology

This repository contains the source code, benchmark results, and ablation studies for a **Hybrid CNN-TDA (Convolutional Neural Network + Topological Data Analysis)** diagnostics framework designed to classify breast cancer tumors from radiological slides with high accuracy and diagnostic sensitivity.

---

## Architecture Overview

Our pipeline integrates deep spatial features with topological shape characteristics, structured inside a leak-free evaluation environment:

1. **Grayscale ROI Segmenter:** Crops the dense tumor mass automatically using Otsu's Adaptive Thresholding and Gaussian blurring, resizing the region of interest to a uniform $64 \times 64$ size.
2. **Medical Weight Initialization (Self-Supervised Autoencoders):** Replaces ImageNet pre-trained backbones by converting the input layer of `ResNet18` and `MobileNetV2` to 1-channel grayscale and pre-training them as self-supervised autoencoders on the radiological slides.
3. **Dual-Representation Topological Feature Extraction:** Computes Persistent Homology ($H_0$ and $H_1$) via Cubical Complexes, concatenating 3-layer `PersistenceLandscapes` and 2D `PersistenceImages` (5,600 dimensions total).
4. **In-Pipeline Dimensionality Reduction (PCA & L1 Tracks):**
   * **PCA Track:** Projects scaled hybrid vectors down to $100$ dimensions using Principal Component Analysis (`PCA(n_components=100)`).
   * **L1 Selection Track:** Selects the most discriminative features using a sparse linear support vector estimator (`SelectFromModel` with `LinearSVC(penalty='l1', C=0.01)`).
5. **Leakage-Free Cross-Validation:** Runs a 5-fold `StratifiedGroupKFold` split grouped by patient ID, ensuring no patient slices cross-contaminate training and validation folds.

---

## Performance Benchmark

All metrics were validated using 5-fold cross-validation:

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees (TDA-Only)** | 0.6872 ± 0.0456 | 0.7207 ± 0.0343 | 0.8754 ± 0.0594 | 0.7891 ± 0.0320 | 0.7132 ± 0.0735 |
| **ResNet18 + TDA (Extra Trees) Baseline** | 0.6917 ± 0.0478 | 0.7252 ± 0.0356 | 0.8754 ± 0.0723 | 0.7911 ± 0.0355 | 0.7132 ± 0.0747 |
| **ResNet18 + TDA (Extra Trees) + PCA** | 0.6738 ± 0.0229 | 0.6875 ± 0.0158 | **0.9417 ± 0.0196** | 0.7946 ± 0.0132 | 0.6387 ± 0.0567 |
| **ResNet18 + TDA (Extra Trees) + L1** | 0.6705 ± 0.0405 | 0.7047 ± 0.0237 | 0.8742 ± 0.0389 | 0.7802 ± 0.0287 | 0.6947 ± 0.0560 |
| **MobileNetV2 + TDA (Extra Trees) Baseline** | **0.7023 ± 0.0506** | 0.7324 ± 0.0388 | 0.8806 ± 0.0595 | **0.7982 ± 0.0350** | **0.7140 ± 0.0808** |
| **MobileNetV2 + TDA (Extra Trees) + PCA** | 0.6663 ± 0.0210 | 0.6812 ± 0.0120 | **0.9429 ± 0.0289** | 0.7908 ± 0.0152 | 0.6417 ± 0.0535 |
| **MobileNetV2 + TDA (Extra Trees) + L1** | 0.6640 ± 0.0488 | 0.7040 ± 0.0282 | 0.8600 ± 0.0499 | 0.7739 ± 0.0347 | 0.6838 ± 0.0492 |

### **Key Insights:**
* **Top Performance:** MobileNetV2 + TDA baseline achieved the highest overall accuracy (**`70.23%`**).
* **Clinical Screening Sensitivity:** The PCA track cuts model variance in half and boosts **Recall (Sensitivity) to 94.29%**, making it an ideal configuration to minimize false negatives in cancer screenings.

---

## Ablation Study Results

### **1. Representation Ablation (Landscape vs. Image)**
Tested on Fold 1 with Extra Trees:
* **Persistence Landscapes Only:** Accuracy: `0.5873`, AUC: `0.6271`
* **Persistence Images Only:** Accuracy: `0.6270`, AUC: `0.7086`
* **Concatenated TDA representation:** Accuracy: **`0.6429`**, AUC: **`0.6956`**

### **2. Single-Core CPU Inference Throughput**
* **Average Homology Extraction:** `25.46 ms / image`
* **Average Classification Latency:** `0.76 ms / image`
* **Total Latency:** **`26.22 ms / image`** (~38 FPS on CPU)

### **3. Resolution Ablation**
* **64x64:** Homology Time (500 samples): **`15.85s`** | Accuracy: `1.00`
* **128x128:** Homology Time (500 samples): **`53.94s`** | Accuracy: `1.00`
* *Conclusion:* Downscaling to $64 \times 64$ saves **70.6% of CPU processing time** while preserving full diagnostic diagnostic scores.

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Sharon-codes/CVIP-Kerela.git
   cd CVIP-Kerela
   ```

2. **Install Dependencies:**
   ```bash
   pip install numpy scipy pandas scikit-learn scikit-image opencv-python giotto-tda torch torchvision medmnist tabulate matplotlib seaborn
   ```

3. **Run the Benchmark:**
   ```bash
   python tda_benchmark.py
   ```

4. **Run Ablation Studies:**
   ```bash
   python ablation_benchmark.py
   ```
