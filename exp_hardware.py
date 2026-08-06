#!/usr/bin/env python
"""
Hardware & System Profiling Module:
- Exp 3: Resolution Scaling (32x32, 48x48, 64x64, 96x96, 128x128, 256x256) vs Latency & RAM
- Exp 10: Granular Pipeline Breakdown (ROI -> CNN -> TDA -> PCA -> Classifier)

Uses time.perf_counter() and tracemalloc to profile physical edge-device latency and RAM overhead.
Author: Lead Biomedical ML Engineer (Q1 Journal Submission Suite)
"""

import os
import time
import tracemalloc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from core_pipeline import extract_cnn, extract_tda


# =====================================================================
# Experiment 3: Tensor Resolution Scaling Analysis
# =====================================================================
def run_experiment_3_resolution_scaling(raw_imgs, y_sample, save_dir="images", dpi=300):
    print("\n[+] Running Experiment 3: Tensor Resolution Scaling & Memory Profiling...")

    resolutions = [32, 48, 64, 96, 128, 256]
    latencies_ms = []
    peak_rams_mb = []

    # Take a representative sample of 50 images for benchmarking latency
    n_sample = min(50, len(raw_imgs))
    sample_imgs = raw_imgs[:n_sample]

    for res in resolutions:
        # Resize raw images to current resolution
        resized_batch = np.array([cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0 for img in sample_imgs])

        tracemalloc.start()
        t0 = time.perf_counter()

        # Run CNN and TDA extraction
        _ = extract_cnn(resized_batch)
        _ = extract_tda(resized_batch)

        t1 = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        avg_latency_ms = ((t1 - t0) / n_sample) * 1000.0
        peak_ram_mb = peak_mem / (1024.0 * 1024.0)

        latencies_ms.append(avg_latency_ms)
        peak_rams_mb.append(peak_ram_mb)

        print(f"  Resolution: {res}x{res:3d} | Latency: {avg_latency_ms:6.2f} ms/img | Peak RAM: {peak_ram_mb:6.2f} MB")

    # Plot Dual Y-Axis Figure
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    color_lat = '#d9534f'
    color_ram = '#1f77b4'

    ax1.set_xlabel('Tensor Resolution (Pixels)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Inference Latency (ms / image)', color=color_lat, fontsize=11, fontweight='bold')
    line1 = ax1.plot(resolutions, latencies_ms, color=color_lat, marker='o', linewidth=2.5, markersize=8, label='Inference Latency (ms)')
    ax1.tick_params(axis='y', labelcolor=color_lat)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Peak Memory Overhead (MB RAM)', color=color_ram, fontsize=11, fontweight='bold')
    line2 = ax2.plot(resolutions, peak_rams_mb, color=color_ram, marker='s', linestyle='--', linewidth=2.5, markersize=8, label='Peak Memory (MB)')
    ax2.tick_params(axis='y', labelcolor=color_ram)

    # Added legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10.5, frameon=True)

    plt.title("Exp 3: Hardware Scaling — Tensor Resolution vs. Latency & Memory", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()

    out_path = os.path.join(save_dir, "exp3_resolution_scaling.jpg")
    plt.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved plot to {out_path}")

    return {"resolutions": resolutions, "latencies_ms": latencies_ms, "peak_rams_mb": peak_rams_mb}


# =====================================================================
# Experiment 10: Granular Pipeline Latency & Memory Breakdown
# =====================================================================
def run_experiment_10_pipeline_breakdown(sample_raw_imgs, roi_boxes_sample, hybrid_pipeline, save_dir="images", dpi=300):
    print("\n[+] Running Experiment 10: Granular Pipeline Breakdown & CPU Profiling...")

    n_sample = min(50, len(sample_raw_imgs))
    raw_sub = sample_raw_imgs[:n_sample]
    boxes_sub = roi_boxes_sample[:n_sample]

    stages = [
        "1. ROI & Preproc",
        "2. MobileNetV2 (CNN)",
        "3. Cubical Homology (TDA)",
        "4. PCA Compression",
        "5. ExtraTrees Classifier"
    ]

    stage_latencies = []
    stage_mems = []

    # 1. ROI & Preproc
    tracemalloc.start()
    t0 = time.perf_counter()
    preproc_imgs = []
    for img, (x, y, w, h) in zip(raw_sub, boxes_sub):
        crop = img[y:y+h, x:x+w] if w > 5 and h > 5 else img
        resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        preproc_imgs.append(resized.astype(np.float32) / 255.0)
    preproc_imgs = np.array(preproc_imgs)
    t1 = time.perf_counter()
    _, peak1 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stage_latencies.append(((t1 - t0) / n_sample) * 1000.0)
    stage_mems.append(peak1 / (1024.0 * 1024.0))

    # 2. MobileNetV2 CNN Extraction
    tracemalloc.start()
    t0 = time.perf_counter()
    X_cnn_sample = extract_cnn(preproc_imgs)
    t1 = time.perf_counter()
    _, peak2 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stage_latencies.append(((t1 - t0) / n_sample) * 1000.0)
    stage_mems.append(peak2 / (1024.0 * 1024.0))

    # 3. Cubical Homology TDA Extraction
    tracemalloc.start()
    t0 = time.perf_counter()
    X_tda_sample = extract_tda(preproc_imgs)
    t1 = time.perf_counter()
    _, peak3 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stage_latencies.append(((t1 - t0) / n_sample) * 1000.0)
    stage_mems.append(peak3 / (1024.0 * 1024.0))

    X_hybrid_sample = np.hstack([X_tda_sample, X_cnn_sample])

    # 4. PCA Compression
    scaler = hybrid_pipeline.named_steps['scaler']
    pca = hybrid_pipeline.named_steps['pca']
    clf = hybrid_pipeline.named_steps['clf']

    tracemalloc.start()
    t0 = time.perf_counter()
    X_scaled = scaler.transform(X_hybrid_sample)
    if 'l1_select' in hybrid_pipeline.named_steps:
        X_scaled = hybrid_pipeline.named_steps['l1_select'].transform(X_scaled)
    X_pca = pca.transform(X_scaled)
    t1 = time.perf_counter()
    _, peak4 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stage_latencies.append(((t1 - t0) / n_sample) * 1000.0)
    stage_mems.append(peak4 / (1024.0 * 1024.0))

    # 5. ExtraTrees Classifier
    tracemalloc.start()
    t0 = time.perf_counter()
    _ = clf.predict_proba(X_pca)
    t1 = time.perf_counter()
    _, peak5 = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stage_latencies.append(((t1 - t0) / n_sample) * 1000.0)
    stage_mems.append(peak5 / (1024.0 * 1024.0))

    total_lat = sum(stage_latencies)
    print("\n  Granular System Stage Breakdown:")
    for s, l, m in zip(stages, stage_latencies, stage_mems):
        print(f"    {s:28s} : {l:6.2f} ms ({l/total_lat*100:5.1f}%) | Peak RAM: {m:5.2f} MB")
    print(f"    {'TOTAL SYSTEM INFERENCE TIME':28s} : {total_lat:6.2f} ms / image (~{1000.0/total_lat:.1f} FPS)")

    # Stacked / Horizontal Bar Plot
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd', '#d62728']
    bars = ax.barh(stages, stage_latencies, color=colors, edgecolor='black', linewidth=1.2)

    for bar, lat in zip(bars, stage_latencies):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.0,
                f"{lat:.2f} ms ({lat/total_lat*100:.1f}%)",
                va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, max(stage_latencies) * 1.35)
    ax.set_title(f"Exp 10: Granular Latency Profile Across Pipeline Stages (Total = {total_lat:.2f} ms)",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Latency per Image Slice (ms)", fontsize=11, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(save_dir, "exp10_pipeline_breakdown.jpg")
    plt.savefig(out_path, dpi=dpi, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved plot to {out_path}")

    return {
        "stages": stages,
        "stage_latencies": stage_latencies,
        "stage_mems": stage_mems,
        "total_latency_ms": total_lat
    }
