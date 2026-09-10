# Lab 01 Report: Skin Disease Image Classification

**Submitted By:** Talal Ahmed Tarar  
**Institution:** COMSATS University Islamabad  
**Course:** Introduction to Computer Vision  
**Date:** September 10, 2026  

---

## Table 1. Comparison of Transfer Learning Models
*All deep learning models were fine-tuned for 15 epochs on the ISIC Skin Cancer dataset.*

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AlexNet | 48.31 | 45.74 | 48.61 | 43.72 | 89.32 |
| VGG16 | 55.93 | 58.05 | 54.86 | 50.37 | 89.52 |
| VGG19 | 53.39 | 50.33 | 52.78 | 47.64 | 93.22 |
| ResNet18 | 44.92 | 44.62 | 45.83 | 41.66 | 85.76 |
| ResNet50 | 56.78 | 51.83 | 55.56 | 50.49 | 89.44 |
| ResNet101 | 52.54 | 51.57 | 52.08 | 46.82 | 83.04 |
| DenseNet121 | 56.78 | 52.23 | 55.56 | 51.42 | 89.85 |
| EfficientNet-B0 | 57.63 | 51.39 | 56.25 | 50.58 | 91.58 |

---

## Table 2. Comparison of Different Classifiers
*Feature Extractor Used: Pre-trained ResNet50*

| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Deep Features | Logistic Regression | 46.61 | 48.30 | 47.22 | 44.21 | 87.63 |
| Deep Features | Decision Tree | 24.58 | 22.47 | 29.17 | 23.88 | 59.62 |
| Deep Features | Random Forest | 35.59 | 44.45 | 38.19 | 30.85 | 77.51 |
| Deep Features | K-Nearest Neighbors (KNN) | 37.29 | 37.65 | 36.57 | 33.59 | 75.93 |
| Deep Features | Linear SVM | 44.07 | 41.68 | 45.14 | 40.09 | 89.11 |
| Deep Features | RBF-SVM | 45.76 | 51.74 | 46.53 | 43.23 | 89.03 |
| Deep Features | XGBoost | 47.46 | 48.10 | 44.91 | 42.73 | 79.77 |

---

## Table 3. Computational Efficiency Comparison

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AlexNet | 61.10 | 233.09 | 1.43 | 1.57 | 48.31 |
| VGG16 | 138.36 | 527.80 | 30.94 | 9.14 | 55.93 |
| VGG19 | 143.67 | 548.06 | 39.26 | 11.34 | 53.39 |
| ResNet18 | 11.69 | 44.66 | 3.65 | 2.24 | 44.92 |
| ResNet50 | 25.56 | 97.78 | 8.27 | 5.95 | 56.78 |
| ResNet101 | 44.55 | 170.51 | 15.73 | 12.78 | 52.54 |
| DenseNet121 | 7.98 | 30.97 | 5.79 | 20.92 | 56.78 |
| EfficientNet-B0 | 5.29 | 20.44 | 0.83 | 9.69 | 57.63 |
