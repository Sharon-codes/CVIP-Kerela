# The Limitations of Persistent Homology for Edge-Based Oncology: An Empirical Analysis of Covariate Shift and Sensor Degradation
**Target Conference / Journal:** IEEE INDICON / IEEE Transactions on Medical Imaging  

---

## IEEE INDICON MANUSCRIPT DRAFT SECTIONS

### ABSTRACT

Persistent homology and Topological Data Analysis (TDA) have gained traction in medical image analysis due to their theoretical coordinate invariance and mathematical guarantees of shape persistence. However, their practical viability on resource-constrained edge-AI hardware remains largely unscrutinized under real-world clinical covariate shift and physical sensor degradation. In this empirical critique, we benchmark a Hybrid MobileNetV2–TDA pipeline against lightweight convolutional neural network (CNN) baselines (MobileNetV2, ResNet18, EfficientNet-B0) across 7,632 real patient radiological scans ($N_{primary}=6,120$, $N_{external}=1,512$). Our empirical evaluations reveal two critical vulnerabilities: (1) **Failure of Topological Shielding:** Under 13 distinct physical sensor degradation conditions (Gaussian noise $\sigma \in [0.01, 0.10]$, camera rotation $\theta \in [15^\circ, 90^\circ]$, brightness shifts $\pm 20\%$, and JPEG compression loss $Q \in [30, 90]$), topological feature concatenation consistently fails to shield the model, producing negative ROC-AUC performance deltas relative to standalone EfficientNet-B0 (e.g., $\Delta = -0.0635$ under Gaussian noise $\sigma=0.01$ and $\Delta = -0.1338$ under JPEG compression $Q=50$). McNemar's test confirms no statistically significant difference in misclassification patterns between the Hybrid pipeline and MobileNetV2 ($p = 0.3041$). (2) **The Resolution-Memory Catch-22:** Scaling spatial resolution from $32 \times 32$ to $256 \times 256$ increases average Betti-1 loop detection from $30.99$ to $1,035.30$, but exponentially inflates TDA extraction latency from $14.54\text{ ms/img}$ to $278.48\text{ ms/img}$ ($19.2\times$ increase) and peak memory usage from $10.27\text{ MB}$ to $331.82\text{ MB}$ ($32.3\times$ increase). Consequently, while lightweight CNNs achieve superior external generalization (MobileNetV2 AUC = 0.6179, EfficientNet-B0 AUC = 0.6107) with minimal memory footprints ($0.59\text{ MB}$–$1.47\text{ MB}$), persistent homology creates a severe computational bottleneck that renders it fundamentally unsuited for sub-5MB edge-AI oncology devices.

---

### RESULTS & DISCUSSION

#### A. The Failure of Topological Shielding under Physical Degradation

A central hypothesis in applied TDA literature asserts that persistent homology features—by virtue of tracking global topological invariants ($H_0$ connected components and $H_1$ 1-dimensional loops)—provide an inherent hardware shield against physical sensor perturbations and high-frequency sensor noise. To empirically test this hypothesis, we subjected the out-of-distribution external cohort ($N=1,512$ scans from 21 isolated patients) to 13 controlled physical sensor degradation regimes and evaluated the delta in ROC-AUC ($\Delta = \text{AUC}_{\text{Hybrid}} - \text{AUC}_{\text{EfficientNet}}$).

As detailed in Table I, the Hybrid CNN-TDA pipeline systematically underperformed standalone EfficientNet-B0 across all degradation parameters. Under additive Gaussian noise ($\sigma = 0.01$), the standalone EfficientNet-B0 retained an ROC-AUC of 0.5989, whereas the Hybrid model collapsed to 0.5354 ($\Delta = -0.0635$). At higher noise levels ($\sigma \in [0.05, 0.10]$), the topological features saturated, forcing the Hybrid AUC down to the random baseline of 0.5000. Under rotational transformations ($\theta = 90^\circ$), EfficientNet-B0 maintained an AUC of 0.6079 compared to the Hybrid model's 0.5520 ($\Delta = -0.0559$). Similarly, under lossy JPEG compression ($Q=50$), the performance gap widened to $\Delta = -0.1338$ (EfficientNet-B0 AUC = 0.6265 vs. Hybrid AUC = 0.4927).

```
TABLE I: PHYSICAL SENSOR DEGRADATION BENCHMARK (EXTERNAL COHORT N=1,512)
-----------------------------------------------------------------------------------------
Perturbation Type         Parameter      Hybrid CNN-TDA AUC  EffNet-B0 AUC   Delta (Δ)
-----------------------------------------------------------------------------------------
Gaussian Noise            σ = 0.01             0.5354           0.5989        -0.0635
Gaussian Noise            σ = 0.05             0.5000           0.5151        -0.0151
Gaussian Noise            σ = 0.010            0.5000           0.5330        -0.0330
Camera Rotation           θ = 15°              0.5860           0.5890        -0.0030
Camera Rotation           θ = 45°              0.5696           0.5983        -0.0287
Camera Rotation           θ = 90°              0.5520           0.6079        -0.0559
Gain / Brightness Shift   -20%                 0.5279           0.6255        -0.0976
Gain / Brightness Shift   -10%                 0.5388           0.6215        -0.0828
Gain / Brightness Shift   +10%                 0.5314           0.6126        -0.0812
Gain / Brightness Shift   +20%                 0.5329           0.6177        -0.0848
JPEG Lossy Compression    Q = 90               0.5328           0.5927        -0.0599
JPEG Lossy Compression    Q = 70               0.5292           0.6124        -0.0832
JPEG Lossy Compression    Q = 50               0.4927           0.6265        -0.1338
JPEG Lossy Compression    Q = 30               0.4967           0.5634        -0.0667
-----------------------------------------------------------------------------------------
```

Furthermore, non-parametric 1,000-iteration bootstrapping yielded an external 95% Confidence Interval for the Hybrid model's Specificity of `0.1155 - 0.1770` (mean Specificity = 0.1447), proving that the addition of topological vectors fails to prevent decision-boundary collapse. While the Wilcoxon signed-rank test confirmed that probability outputs differed significantly between models ($W = 516,985.0, p = 1.217 \times 10^{-3}$), McNemar's test for misclassification table equivalence yielded $\chi^2 = 1.0562, p = 0.3041$, establishing that TDA features introduce no statistically significant structural correction over pure spatial CNN representations.

#### B. The Resolution-Memory Catch-22

To elucidate the computational mechanics behind TDA's edge failure, we conducted a resolution scaling experiment across 500 real radiological scans. We measured Betti-1 loop extraction count, CPU inference latency, and peak memory consumption across four spatial resolutions: $32 \times 32$, $64 \times 64$, $128 \times 128$, and $256 \times 256$.

```
TABLE II: TDA RESOLUTION SCALABILITY & RESOURCE BOTTLENECK (N=500 REAL SCANS)
-----------------------------------------------------------------------------------------
Tensor Resolution  Avg Betti-1 Loops Detected   Latency (ms/img)   Peak Memory RAM (MB)
-----------------------------------------------------------------------------------------
32 x 32                   31.00                     14.54                10.27
64 x 64 (Baseline)       122.04                     16.91                35.16
128 x 128                448.52                     69.71               128.09
256 x 256               1035.30                    278.48               331.82
-----------------------------------------------------------------------------------------
```

The empirical scaling curves reveal a fundamental **Resolution-Memory Catch-22**:
1. **Low-Resolution Information Loss ($32 \times 32$ / $64 \times 64$):** At lower resolutions, Cubical Complex filtration detects only macro-scale topological features (31.00 Betti-1 loops at $32 \times 32$; 122.04 loops at $64 \times 64$). However, at these coarse resolutions, fine micro-calcification boundaries and speculative tumor margins are blurred out, causing topological signatures to encode image noise rather than pathological invariants.
2. **High-Resolution Resource Explosion ($128 \times 128$ / $256 \times 256$):** Increasing the spatial resolution to $256 \times 256$ captures rich topological structures ($1,035.30$ Betti-1 loops per scan). However, the algorithmic complexity of cubical persistence filtration scales non-linearly. Latency spikes from $16.91\text{ ms/img}$ to $278.48\text{ ms/img}$ ($16.5\times$ increase), and peak RAM requirements explode to **331.82 MB** ($9.4\times$ increase over $64 \times 64$).

In edge-AI oncology deployments—where microcontrollers and embedded mobile processors are strictly constrained to $\le 5\text{ MB}$ of RAM and require sub-10ms inference—standalone lightweight CNNs (MobileNetV2: $0.59\text{ MB}$ RAM, $1.25\text{ ms/img}$; EfficientNet-B0: $1.47\text{ MB}$ RAM, $2.33\text{ ms/img}$) dominate TDA in both computational efficiency and out-of-distribution generalization. Persistent homology thus presents an unsustainable resource bottleneck that outweighs its theoretical guarantees in clinical edge diagnostics.

---

## COMPLETE NUMERICAL BENCHMARKING DATA

### 1. Model Latency & Memory Footprint Comparison
- **MobileNetV2 (CNN Only):** Internal AUC = 0.7350, External AUC = 0.6179, Accuracy = 0.6667, Latency = 1.25 ms/img, Peak RAM = 0.59 MB.
- **ResNet18 (CNN Only):** Internal AUC = 0.7060, External AUC = 0.5472, Accuracy = 0.6442, Latency = 1.47 ms/img, Peak RAM = 0.59 MB.
- **EfficientNet-B0 (CNN Only):** Internal AUC = 0.7588, External AUC = 0.6107, Accuracy = 0.6376, Latency = 2.33 ms/img, Peak RAM = 1.47 MB.
- **Hybrid CNN-TDA (SVM):** Internal AUC = 0.5886, External AUC = 0.5816, Accuracy = 0.6574, Latency = 0.12 ms/img (after extraction), Peak RAM = 2.45 MB.

### 2. Statistical Metrics & Confidence Intervals
- **1,000-Bootstrap 95% CIs (External Cohort):**
  - External AUC: 0.5820 (95% CI: 0.5500 - 0.6117)
  - Sensitivity: 0.9137 (95% CI: 0.8960 - 0.9302)
  - Specificity: 0.1447 (95% CI: 0.1155 - 0.1770)
- **Wilcoxon Test:** Statistic = 516,985.00, $p$-value = $1.217 \times 10^{-3}$ ($p < 0.05$).
- **McNemar's Test:** Contingency matrix = `[[921, 73], [87, 431]]`, Statistic = 1.0562, $p$-value = 0.3041.

---

## 300 DPI IEEE Publication Figures
All 3 figures are saved in the `IEEE Manuscript/` directory:
1. `IEEE Manuscript/fig_umap_separation.png` (300 DPI UMAP Manifold Visualization)
2. `IEEE Manuscript/fig_robustness_curves.png` (300 DPI Sensor Degradation Curves)
3. `IEEE Manuscript/fig_ablation_waterfall.png` (300 DPI Component Ablation Waterfall)
