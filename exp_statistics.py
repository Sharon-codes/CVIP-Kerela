#!/usr/bin/env python
"""
Experiment 2: IEEE Statistical Rigor Suite
Calculates:
- 1,000-iteration Bootstrapped 95% Confidence Intervals for Sensitivity, Specificity, and AUC on the external cohort.
- Wilcoxon Signed-Rank Test comparing probability distributions of Hybrid model vs. EfficientNet-B0.
- McNemar's Test on paired contingency tables of Hybrid model vs. CNN-only MobileNetV2.

Author: Lead Biomedical ML Engineer (IEEE Q1 Submission Suite)
"""

import numpy as np
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def bootstrap_ci(y_true, y_prob, n_bootstraps=1000, alpha=0.95):
    boot_aucs, boot_sens, boot_specs = [], [], []
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        y_t, y_p = y_true[idx], y_prob[idx]

        # Ensure both classes present in bootstrap sample
        if len(np.unique(y_t)) < 2:
            continue

        pred = (y_p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, pred, labels=[0, 1]).ravel()

        boot_aucs.append(roc_auc_score(y_t, y_p))
        boot_sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        boot_specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    lower_p = (1.0 - alpha) / 2.0 * 100
    upper_p = (1.0 + alpha) / 2.0 * 100

    ci_auc = (np.percentile(boot_aucs, lower_p), np.percentile(boot_aucs, upper_p))
    ci_sens = (np.percentile(boot_sens, lower_p), np.percentile(boot_sens, upper_p))
    ci_spec = (np.percentile(boot_specs, lower_p), np.percentile(boot_specs, upper_p))

    return {
        "mean_auc": np.mean(boot_aucs), "ci_auc": ci_auc,
        "mean_sens": np.mean(boot_sens), "ci_sens": ci_sens,
        "mean_spec": np.mean(boot_specs), "ci_spec": ci_spec
    }


def run_statistical_tests(saved_probs):
    print("=" * 80)
    print("EXPERIMENT 2: IEEE STATISTICAL RIGOR (BOOTSTRAP 95% CI, WILCOXON, MCNEMAR)")
    print("=" * 80)

    hybrid_data = saved_probs["Hybrid CNN-TDA (Final Pipeline)"]
    mobilenet_data = saved_probs["MobileNetV2 (CNN Only)"]
    effnet_data = saved_probs["EfficientNet-B0 (CNN Only)"]

    y_ext = hybrid_data["y_ext"]
    y_prob_hybrid = hybrid_data["y_ext_prob"]
    y_prob_mobilenet = mobilenet_data["y_ext_prob"]
    y_prob_effnet = effnet_data["y_ext_prob"]

    # 1. 1,000-iteration Bootstrapped 95% Confidence Intervals
    print("\n[+] 1,000-Iteration Bootstrapped 95% Confidence Intervals (External Cohort)...")
    ci_hybrid = bootstrap_ci(y_ext, y_prob_hybrid)
    ci_mobilenet = bootstrap_ci(y_ext, y_prob_mobilenet)
    ci_effnet = bootstrap_ci(y_ext, y_prob_effnet)

    print(f"  Hybrid CNN-TDA  -> AUC: {ci_hybrid['mean_auc']:.4f} (95% CI: {ci_hybrid['ci_auc'][0]:.4f} - {ci_hybrid['ci_auc'][1]:.4f})")
    print(f"                     Sens: {ci_hybrid['mean_sens']:.4f} (95% CI: {ci_hybrid['ci_sens'][0]:.4f} - {ci_hybrid['ci_sens'][1]:.4f})")
    print(f"                     Spec: {ci_hybrid['mean_spec']:.4f} (95% CI: {ci_hybrid['ci_spec'][0]:.4f} - {ci_hybrid['ci_spec'][1]:.4f})")

    # 2. Wilcoxon Signed-Rank Test (Hybrid vs. EfficientNet-B0)
    print("\n[+] Wilcoxon Signed-Rank Test (Hybrid vs. EfficientNet-B0 Probability Distributions)...")
    diffs = np.abs(y_prob_hybrid - y_ext) - np.abs(y_prob_effnet - y_ext)
    wilc_stat, wilc_p = stats.wilcoxon(diffs)
    print(f"  Wilcoxon Statistic: {wilc_stat:.4f} | p-value: {wilc_p:.6e} (p < 0.05: {wilc_p < 0.05})")

    # 3. McNemar's Test (Hybrid vs. MobileNetV2)
    print("\n[+] McNemar's Test (Contingency Matrix of Misclassifications: Hybrid vs. MobileNetV2)...")
    pred_hybrid = (y_prob_hybrid >= 0.5).astype(int)
    pred_mobilenet = (y_prob_mobilenet >= 0.5).astype(int)

    correct_hybrid = (pred_hybrid == y_ext)
    correct_mobilenet = (pred_mobilenet == y_ext)

    # Contingency Table:
    # [ [both_correct, hybrid_correct_mobile_wrong], [mobile_correct_hybrid_wrong, both_wrong] ]
    a = np.sum(correct_hybrid & correct_mobilenet)
    b = np.sum(correct_hybrid & ~correct_mobilenet)
    c = np.sum(~correct_hybrid & correct_mobilenet)
    d = np.sum(~correct_hybrid & ~correct_mobilenet)

    contingency_table = [[a, b], [c, d]]
    mcnemar_result = mcnemar(contingency_table, exact=False, correction=True)
    mcn_stat = mcnemar_result.statistic
    mcn_p = mcnemar_result.pvalue

    print(f"  Contingency Table: [[a={a}, b={b}], [c={c}, d={d}]]")
    print(f"  McNemar Statistic: {mcn_stat:.4f} | p-value: {mcn_p:.6e} (Significant Difference: {mcn_p < 0.05})")

    stats_summary = {
        "ci_hybrid": ci_hybrid,
        "ci_mobilenet": ci_mobilenet,
        "ci_effnet": ci_effnet,
        "wilcoxon_stat": wilc_stat,
        "wilcoxon_p": wilc_p,
        "mcnemar_table": contingency_table,
        "mcnemar_stat": mcn_stat,
        "mcnemar_p": mcn_p
    }

    return stats_summary


if __name__ == "__main__":
    from exp_baselines import run_baselines_experiment
    _, saved_probs = run_baselines_experiment()
    run_statistical_tests(saved_probs)
