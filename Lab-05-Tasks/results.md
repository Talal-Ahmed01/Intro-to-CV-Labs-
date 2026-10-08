# LAB 05: HOG-Based Industrial Defect Detection and Classification

**Name:** Talal Ahmed Tarar  
**Registration Number:** FA23-BAI-031  

---

## 1. Introduction
Manual inspection in manufacturing pipelines is historically prone to human error and throughput bottlenecks. To mitigate this, computer vision systems utilize feature descriptors to automate anomaly detection. This report details the development of an automated visual inspection prototype utilizing the Histogram of Oriented Gradients (HOG) algorithm. By capturing shape, edge, and gradient information from material surfaces, the HOG descriptor translates raw pixel data into structured feature vectors, which are then evaluated by machine learning classifiers to identify specific product flaws.

## 2. Industrial Application
In steel manufacturing, surface defects directly compromise the structural integrity and aesthetic quality of the final product. An automated computer vision pipeline deployed on the factory floor enables real-time, non-destructive quality control. By instantly categorizing steel sheets as either acceptable or defective, manufacturers can reduce manual labor costs, prevent defective products from reaching clients, and maintain a high-velocity production line without sacrificing accuracy.

## 3. Dataset Description
The system was trained and evaluated on the NEU Steel Surface Defect Database (NEU-DET). To achieve advanced multi-class defect classification, the dataset was segmented into six distinct anomaly categories:
*   crazing
*   inclusion
*   patches
*   pitted_surface
*   rolled-in_scale
*   scratches

The data was structured into a training set of 1,700 images and a validation set of 100 images. Notably, the dataset consists exclusively of defective surfaces, meaning all samples fundamentally trigger a rejection in a binary pass/fail system.

## 4. Image Preprocessing
To optimize processing time within a constrained window, all images were resized to a uniform 64x64 pixel resolution. Since HOG relies on gradient intensity rather than color channels, the images were immediately converted to grayscale upon loading. This significantly reduced the computational overhead while preserving the essential structural geometry required for accurate defect classification.

## 5. HOG Feature Extraction
HOG features were extracted using the `skimage.feature.hog` library. The baseline extraction parameters were configured as follows:
*   **Orientations:** 9
*   **Pixels per cell:** 8x8
*   **Cells per block:** 2x2
*   **Block Normalization:** L2-Hys

This configuration yielded a highly dimensional feature vector that successfully captured the edge directions of the various surface defects. Visualizations comparing the original 64x64 grayscale images to their HOG representations confirmed that distinct patterns (such as parallel scratches versus clustered crazing) were mathematically highlighted.

## 6. Classification Methodology
Two distinct machine learning algorithms were trained on the extracted HOG feature vectors:
1.  **Support Vector Machine (SVM):** Configured with an RBF (Radial Basis Function) kernel and probability estimation enabled.
2.  **XGBoost Classifier:** A gradient boosting framework configured with 100 estimators, a maximum depth of 3, and a learning rate of 0.1.
Target labels (defect strings) were numerically encoded using `LabelEncoder` prior to training to ensure compatibility with the XGBoost architecture.

## 7. Experimental Setup
The pipeline was developed and executed in a Google Colab environment utilizing a T4 GPU. The experimental workflow isolated distinct variables:
*   **Classifier Comparison:** HOG + SVM vs. HOG + XGBoost.
*   **Parameter Tuning:** Iterating through different HOG cell sizes (4x4, 8x8, 16x16) and orientations (6, 9, 12) using a high-speed LinearSVC to benchmark the optimal extraction geometry.
*   **Robustness Evaluation:** Applying synthetic corruptions (Brightness variation, Gaussian noise, Rotation, and Blur) to the validation set to measure performance degradation against the baseline XGBoost model.

## 8. Results
The SVM classifier outfitted with an RBF kernel outperformed the XGBoost model across all primary metrics. 

**Classifier Performance Comparison (Baseline HOG: 8x8 cell, 9 ori)**

| Metric | SVM (HOG) | XGBoost (HOG) |
| :--- | :--- | :--- |
| **Accuracy** | 0.9000 | 0.8600 |
| **Precision (Weighted)** | 0.9056 | 0.8726 |
| **Recall (Weighted)** | 0.9000 | 0.8600 |
| **F1-Score (Weighted)** | 0.8965 | 0.8626 |

<br>

**HOG Parameter Experiment Results (Evaluated with LinearSVC)**

| Cell Size | Orientations | Accuracy |
| :--- | :--- | :--- |
| 4x4 | 6 | 0.52 |
| 4x4 | 9 | 0.49 |
| 4x4 | 12 | 0.56 |
| 8x8 | 6 | 0.69 |
| 8x8 | 9 | 0.62 |
| 8x8 | 12 | 0.74 |
| 16x16 | 6 | 0.71 |
| 16x16 | 9 | 0.74 |
| 16x16 | 12 | 0.74 |

The parameter experiments demonstrated that an 8x8 cell size with 12 orientations, or a 16x16 cell size with 9 or 12 orientations, yielded the highest accuracy (0.74) for the linear classifier, proving that overly granular cells (4x4) capture too much noise and degrade the model's predictive capability.

## 9. Robustness Analysis
To simulate real-world factory floor conditions, the validation images were synthetically altered. The performance was measured against the baseline XGBoost model (Accuracy: 0.86).

| Condition | Accuracy | Acc Drop | F1-Score | F1 Drop |
| :--- | :--- | :--- | :--- | :--- |
| **Brightness Variation** (x1.5) | 0.63 | 0.23 | 0.6298 | 0.2327 |
| **Rotation** (15 degrees) | 0.38 | 0.48 | 0.3236 | 0.5389 |
| **Blur** (Gaussian 5x5) | 0.38 | 0.48 | 0.2983 | 0.5642 |
| **Gaussian Noise** | 0.24 | 0.62 | 0.1680 | 0.6944 |

## 10. Industrial Deployment Discussion
The final quality-control module was designed to process a standard image, extract the HOG vector, cross-reference it against the trained model, and output an actionable command for the manufacturing line. Because the system utilizes an aggressively down-sampled 64x64 resolution, it is highly lightweight and suitable for real-time edge computing deployment (e.g., Raspberry Pi or industrial PLC). 
A standard output from the terminal execution successfully maps the defect to a rigid action state:
`Prediction: DEFECTIVE (crazing) | Confidence: 65.80% | Action: REJECT PRODUCT`

## 11. Limitations
The primary limitation of this pipeline is its high sensitivity to environmental disruption. As evidenced by the Robustness Analysis, introducing a minor 15-degree rotation or a 5x5 blur drops accuracy by nearly 50%, while Gaussian noise cripples the model entirely (0.24 accuracy). HOG relies on rigid gradient orientations; any shift in the camera angle or heavy sensor noise distorts the mathematical structure of the edges. Additionally, the dataset lacked a "Class 0: Normal" category. Consequently, the model was forced to map any input to one of the six defect classes, requiring threshold-based logic to simulate an "ACCEPT PRODUCT" state.

## 12. Conclusion
The implementation of a HOG-based feature extraction pipeline paired with Support Vector Machines provides a highly effective, albeit environmentally sensitive, solution for industrial defect classification. Achieving 90% accuracy on multi-class defect categorization proves the viability of gradient-based computer vision in metallurgy. While deep learning approaches (like Convolutional Neural Networks) might offer better rotational invariance, this HOG/SVM architecture remains a computationally efficient, deployable prototype that fulfills the core objectives of automated quality control.