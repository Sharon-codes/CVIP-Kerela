# Multi-Classifier Algorithmic Rigor Benchmark

GridSearchCV Hyperparameter Optimization across ExtraTrees, RandomForest, LightGBM, and SVM.

| Model | Internal AUC | External AUC | Internal Spec | Internal Recall | External Spec | External Recall | Latency (ms/img) | Peak RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (ExtraTrees)** | 0.6007 | 0.4731 | 0.1167 | 0.9769 | 0.0774 | 0.8859 | 9.68 ms | 2.45 MB |
| **Model B (RandomForest)** | 0.6074 | 0.4582 | 0.1806 | 0.9201 | 0.0635 | 0.8710 | 7.67 ms | 2.45 MB |
| **Model C (LightGBM)** | 0.6256 | 0.5037 | 0.3528 | 0.8252 | 0.3413 | 0.6250 | 0.32 ms | 2.45 MB |
| **Model D (SVM)** | 0.5886 | 0.5816 | 0.3194 | 0.7789 | 0.1448 | 0.9137 | 0.18 ms | 2.45 MB |


### Best Hyperparameter Configurations

- **Model A (ExtraTrees)**: `{'clf__min_samples_leaf': 2, 'feature_selection__estimator__C': 0.1, 'pca__n_components': 90}`
- **Model B (RandomForest)**: `{'clf__min_samples_leaf': 4, 'feature_selection__estimator__C': 0.1, 'pca__n_components': 120}`
- **Model C (LightGBM)**: `{'clf__reg_alpha': 1.0, 'clf__reg_lambda': 0.1, 'feature_selection__estimator__C': 0.1, 'pca__n_components': 90}`
- **Model D (SVM)**: `{'clf__C': 10.0, 'clf__gamma': 'scale', 'feature_selection__estimator__C': 0.1, 'pca__n_components': 120}`
