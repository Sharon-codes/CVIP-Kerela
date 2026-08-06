#!/usr/bin/env python
"""
Experiment 4: Strict Pipeline Component Ablation & SHAP Feature Quantification
Evaluates the step-by-step contribution of each pipeline component:
1. CNN Only (MobileNetV2 features -> SVM)
2. TDA Only (Persistence Landscapes -> SVM)
3. CNN + TDA (Concatenated -> SVM)
4. CNN + TDA + L1 + PCA + SVM (Finalized Pipeline)

Calculates SHAP values to quantify feature importance of Betti-1 TDA features vs. spatial features.

Author: Lead Biomedical ML Engineer (IEEE Q1 Submission Suite)
"""

import numpy as np
import shap
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix

from core_pipeline import load_real_images, extract_cnn, extract_tda

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore")


def run_ablation_experiment(X_cnn_prim=None, X_cnn_ext=None, X_tda_prim=None, X_tda_ext=None, y_prim=None, y_ext=None):
    print("=" * 80)
    print("EXPERIMENT 4: STRICT PIPELINE COMPONENT ABLATION & SHAP ANALYSIS")
    print("=" * 80)

    if X_cnn_prim is None or X_tda_prim is None:
        X_imgs_prim, y_prim, groups_prim, _, _ = load_real_images("data/primary")
        X_imgs_ext, y_ext, _, _, _ = load_real_images("data/external")

        print("\n[+] Extracting Features for Ablation Analysis...")
        X_cnn_prim = extract_cnn(X_imgs_prim)
        X_cnn_ext = extract_cnn(X_imgs_ext)

        X_tda_prim = extract_tda(X_imgs_prim)
        X_tda_ext = extract_tda(X_imgs_ext)

    X_hybrid_prim = np.hstack([X_tda_prim, X_cnn_prim])
    X_hybrid_ext = np.hstack([X_tda_ext, X_cnn_ext])

    # 4 Pipeline Stages
    stages = {
        "1. CNN Only": (
            X_cnn_prim, X_cnn_ext,
            Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE))])
        ),
        "2. TDA Only": (
            X_tda_prim, X_tda_ext,
            Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE))])
        ),
        "3. CNN + TDA (Raw)": (
            X_hybrid_prim, X_hybrid_ext,
            Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE))])
        ),
        "4. Finalized Pipeline (CNN+TDA+L1+PCA+SVM)": (
            X_hybrid_prim, X_hybrid_ext,
            Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectFromModel(
                    LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE, C=0.1)
                )),
                ('pca', PCA(n_components=120, random_state=RANDOM_STATE)),
                ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, C=10.0, gamma='scale', random_state=RANDOM_STATE))
            ])
        )
    }

    ablation_results = []

    for name, (X_tr, X_te, pipe) in stages.items():
        pipe.fit(X_tr, y_prim)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        ext_auc = roc_auc_score(y_ext, y_prob)
        ext_pred = (y_prob >= 0.5).astype(int)
        ext_acc = accuracy_score(y_ext, ext_pred)
        ext_sens = recall_score(y_ext, ext_pred)
        tn, fp, fn, tp = confusion_matrix(y_ext, ext_pred).ravel()
        ext_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"\n[+] {name}:")
        print(f"    External ROC-AUC : {ext_auc:.4f}")
        print(f"    External Accuracy: {ext_acc:.4f}")
        print(f"    Sensitivity      : {ext_sens:.4f} | Specificity: {ext_spec:.4f}")

        ablation_results.append({
            "Stage": name,
            "External_AUC": ext_auc,
            "External_Acc": ext_acc,
            "Sensitivity": ext_sens,
            "Specificity": ext_spec
        })

    # SHAP Quantification for Finalized Pipeline
    print("\n[+] Computing SHAP Feature Quantification for Finalized Pipeline...")
    final_pipe = stages["4. Finalized Pipeline (CNN+TDA+L1+PCA+SVM)"][2]

    # Transform features through pipeline steps before classifier
    scaler = final_pipe.named_steps['scaler']
    feat_sel = final_pipe.named_steps['feature_selection']
    pca = final_pipe.named_steps['pca']
    clf = final_pipe.named_steps['clf']

    X_scaled_ext = scaler.transform(X_hybrid_ext)
    X_sel_ext = feat_sel.transform(X_scaled_ext)
    X_pca_ext = pca.transform(X_sel_ext)

    # Use SHAP KernelExplainer with sampled background and test sets for fast execution
    bg_samples = shap.sample(X_pca_ext, 10, random_state=RANDOM_STATE)
    test_samples = X_pca_ext[:10]

    explainer = shap.KernelExplainer(clf.predict_proba, bg_samples)
    shap_values = explainer.shap_values(test_samples)

    if isinstance(shap_values, list):
        shap_vals_class1 = np.abs(shap_values[1]).mean(axis=0)
    else:
        shap_vals_class1 = np.abs(shap_values).mean(axis=0)

    top10_components = np.argsort(shap_vals_class1)[::-1][:10]
    print(f"  Top 10 PCA Component SHAP Importances: {shap_vals_class1[top10_components]}")

    return ablation_results, shap_vals_class1


if __name__ == "__main__":
    run_ablation_experiment()
