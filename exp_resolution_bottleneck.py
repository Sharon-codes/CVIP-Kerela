#!/usr/bin/env python
"""
Experiment: Resolution Bottleneck & TDA Scalability Analysis
Tests extraction of Betti-1 Persistence Landscapes across different tensor resolutions (32x32, 64x64, 128x128, 256x256)
for 500 real patient medical images.

Tracks:
  1. Average Number of Betti-1 loops detected per image.
  2. TDA Extraction Latency (ms per image).
  3. Peak RAM required for TDA extraction (MB).

Author: Lead Biomedical ML Engineer (IEEE Manuscript Critique Suite)
"""

import os
import time
import gc
import tracemalloc
import numpy as np
import cv2
from tqdm import tqdm
import warnings
import pandas as pd

from gtda.externals.python import CubicalComplex
from gtda.diagrams import PersistenceLandscape

from core_pipeline import load_real_images, RANDOM_STATE

warnings.filterwarnings("ignore")


def run_resolution_bottleneck_experiment(n_samples=500):
    print("=" * 80)
    print("EXPERIMENT: RESOLUTION BOTTLENECK & TDA SCALABILITY (BETTI-1 ANALYSIS)")
    print(f"Dataset Subset: {n_samples} Real Patient Scans")
    print("=" * 80)

    # Load 500 real patient images at raw resolution
    X_imgs_prim, y_prim, _, _, orig_imgs = load_real_images("data/primary")
    if len(orig_imgs) < n_samples:
        n_samples = len(orig_imgs)
    
    # Use first n_samples original unresized gray images
    sample_imgs = orig_imgs[:n_samples]

    resolutions = [(32, 32), (64, 64), (128, 128), (256, 256)]
    results = []

    for res in resolutions:
        res_str = f"{res[0]}x{res[1]}"
        print(f"\n[+] Processing Resolution: {res_str}...")

        # Resize images to target resolution
        resized_imgs = [cv2.resize(img, res, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 for img in sample_imgs]

        # Profile RAM & Latency
        gc.collect()
        tracemalloc.start()
        start_time = time.perf_counter()

        betti1_counts = []
        all_pts_0 = []
        all_pts_1 = []

        for img in tqdm(resized_imgs, desc=f"Cubical Homology {res_str}", leave=False):
            cc = CubicalComplex(dimensions=list(img.shape), top_dimensional_cells=img.flatten())
            persistence_pts = cc.persistence()

            pts_0 = [[b, d, 0] for dim, (b, d) in persistence_pts if dim == 0 and not np.isinf(b) and not np.isinf(d)]
            pts_1 = [[b, d, 1] for dim, (b, d) in persistence_pts if dim == 1 and not np.isinf(b) and not np.isinf(d)]

            betti1_counts.append(len(pts_1))
            all_pts_0.append(pts_0)
            all_pts_1.append(pts_1)

        # Pad diagrams and extract Betti-1 Persistence Landscapes
        max_0 = max(max(len(pts) for pts in all_pts_0), 1)
        max_1 = max(max(len(pts) for pts in all_pts_1), 1)

        padded_diags = []
        for pts_0, pts_1 in zip(all_pts_0, all_pts_1):
            pad_0 = pts_0 + [[0.0, 0.0, 0]] * (max_0 - len(pts_0))
            pad_1 = pts_1 + [[0.0, 0.0, 1]] * (max_1 - len(pts_1))
            padded_diags.append(pad_0 + pad_1)

        diagrams = np.array(padded_diags, dtype=np.float32)
        pl = PersistenceLandscape(n_layers=3, n_values=100, n_jobs=-1)
        _ = pl.fit_transform(diagrams)

        end_time = time.perf_counter()
        current_ram, peak_ram_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tot_latency_ms = (end_time - start_time) * 1000.0
        avg_latency_ms = tot_latency_ms / n_samples
        peak_ram_mb = peak_ram_bytes / (1024.0 * 1024.0)
        avg_betti1_loops = float(np.mean(betti1_counts))

        print(f"  Resolution: {res_str:8s} | Avg Betti-1 Loops: {avg_betti1_loops:8.2f} | Latency: {avg_latency_ms:6.2f} ms/img | Peak RAM: {peak_ram_mb:6.2f} MB")

        results.append({
            "Resolution": res_str,
            "Avg_Betti1_Loops": avg_betti1_loops,
            "Latency_ms_per_img": avg_latency_ms,
            "Peak_RAM_MB": peak_ram_mb
        })

    df_res = pd.DataFrame(results)
    return df_res


if __name__ == "__main__":
    df_res = run_resolution_bottleneck_experiment(n_samples=500)
    print("\n" + "=" * 80)
    print("EMPIRICAL RESOLUTION BOTTLENECK SUMMARY TABLE:")
    print("=" * 80)
    print(df_res.to_string(index=False))
