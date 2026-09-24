# Lab 03 Report: Edge Detection Techniques and Their Impact on Classification

**Submitted By:** Talal Ahmed Tarar  
**Institution:** COMSATS University Islamabad  
**Course:** Introduction to Computer Vision  

---

## 1. Introduction
Edge detection is a fundamental concept in traditional computer vision, aimed at identifying boundaries within an image by detecting discontinuities in pixel intensity. This laboratory investigates various spatial-domain edge detection techniques (Sobel, Prewitt, Laplacian, LoG, and Canny) and their susceptibility to image noise (Gaussian and Salt-and-Pepper). Furthermore, the lab evaluates whether converting images into edge maps benefits deep learning models (Convolutional Neural Networks) and classical machine learning classifiers when performing complex medical image classification on the ISIC skin cancer dataset.

## 2. Methodology
The experiment was conducted in multiple stages using OpenCV, Scikit-learn, and PyTorch. Comparative edge detection was performed using first-order (Sobel, Prewitt), second-order (Laplacian, LoG), and multi-stage (Canny) detectors. Artificial Gaussian and Salt-and-Pepper noise were introduced to evaluate detector robustness, followed by applying Gaussian and Median filters to restore edge quality. A parameter analysis of the Canny edge detector was conducted by varying hysteresis thresholds and kernel sizes. Finally, CNN models (EfficientNet-B0, ResNet18) and Classical ML models (SVM, Random Forest, KNN) were trained on three dataset variants: raw images, filtered images, and Canny edge maps, to compare downstream classification performance.

## 3. Experimental Setup
*   **Environment:** Google Colab (T4 GPU).
*   **Libraries:** PyTorch, Torchvision, OpenCV, Scikit-learn, Matplotlib.
*   **Dataset:** ISIC Skin Cancer Dataset (9 disease classes including melanoma, basal cell carcinoma, nevus).
*   **Models:** EfficientNet-B0 (CNN 1), ResNet18 (CNN 2), Linear SVM, Random Forest, K-Nearest Neighbors (KNN). Deep features for ML models were extracted using a frozen ResNet18 backbone.
*   **Evaluation Metrics:** Accuracy, Precision (Macro), Recall (Macro), F1-Score (Macro), Training Time, and Inference Time.

---

## 4. Results

### Table 1. Effect of Noise and Preprocessing on Edge Detection

| Edge Detector | Input Image | Noise Type | Preprocessing | Edge Quality | Noise Sensitivity | Observations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sobel** | Original | None | None | Thick, continuous | Low | Detects strong lesion boundaries effectively but produces thick edges rather than single-pixel lines. |
| **Sobel** | Noisy | Gaussian | None | Degraded, messy | High | Background noise gradients are amplified, obscuring the actual lesion boundary with false edges. |
| **Sobel** | Noisy | Salt & Pepper | None | Severely degraded | Very High | Every salt and pepper pixel is detected as a sharp, high-contrast edge, rendering the map useless. |
| **Sobel** | Noisy | Gaussian | Gaussian Filter | Blurred but recovered | Moderate | The filter suppresses the noise, but the resulting edges are significantly thicker and blurred. |
| **Sobel** | Noisy | Salt & Pepper | Median Filter | Restored, continuous | Low (post-filter) | Median filtering perfectly removes the S&P impulse noise without blurring the lesion edges, recovering the map. |
| **Prewitt** | Original | None | None | Thick, continuous | Low | Practically identical to Sobel, but slightly more sensitive to horizontal and vertical edges over diagonals. |
| **Laplacian** | Original | None | None | Fine, disjointed | Very High | Amplifies fine skin texture and hair as false edges because it calculates the second derivative without smoothing. |
| **LoG** | Noisy | Gaussian | Built-in Gaussian | Clean, closed loops | Low | The built-in Gaussian filter suppresses the noise before the Laplacian isolates the zero-crossings, yielding continuous boundaries. |
| **Canny** | Original | None | Built-in smoothing | Sharp, single-pixel | Low | Produces the best structural representation of the lesion with thin, continuous, and well-defined borders. |
| **Canny** | Noisy | Gaussian | Gaussian Filter | Fragmented | Moderate | The extra external smoothing causes Canny to lose some of the weaker, genuine lesion boundaries. |
| **Canny** | Noisy | Salt & Pepper | Median Filter | Sharp, single-pixel | Low (post-filter) | The median filter removes the impulse noise completely, allowing Canny to trace the true lesion border. |

---

### Table 2. Canny Parameter Analysis

| Configuration | Low Threshold | High Threshold | Kernel Size | Edge Quality | Number of Detected Edges | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Canny-1** | 30 | 100 | 3x3 | Over-detailed | 2353 | Captures too much irrelevant background skin texture and hair due to overly permissive thresholds. |
| **Canny-2** | 50 | 150 | 3x3 | Clean & Continuous | 499 | Optimal balance; successfully isolates the main structural boundaries of the lesion while ignoring minor pores. |
| **Canny-3** | 100 | 200 | 3x3 | Fragmented | 219 | Too strict; misses weaker genuine edges, resulting in broken boundaries. |
| **Canny-4** | 50 | 150 | 5x5 | Severely Noisy | 14289 | The larger kernel amplifies broader gradients across the skin, flooding the image with thickened false edges. |

---

### Table 3. Cross-Lab Classification Performance Comparison
*Note: Evaluated after 1 Epoch for CNNs and frozen feature extraction for Classical ML on Edge Maps (Set C).*

| Model/Classifier | Accuracy Raw (Lab 1) | Accuracy Filtered (Lab 2) | Accuracy Edge (Lab 3) | Precision (Edge) | Recall (Edge) | F1-Score (Edge) | Training Time (s) | Inference Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN Model 1 (EfficientNet-B0)** | 42.37% | 43.22% | 16.95% | 12.04% | 13.89% | 8.49% | ~35.2 | 9.7 |
| **CNN Model 2 (ResNet18)** | 44.92% | 56.78% | 15.20% | 11.50% | 12.10% | 7.80% | ~32.1 | 8.5 |
| **SVM (Linear)** | 44.07% | - | 12.50% | 10.10% | 11.20% | 6.50% | ~15.0 | 2.1 |
| **Random Forest** | 35.59% | - | 14.10% | 11.00% | 12.50% | 7.20% | ~8.5 | 1.5 |
| **KNN** | 37.29% | - | 11.80% | 9.50% | 10.80% | 5.90% | ~0.5 | 5.4 |

---

## 5. Visual Comparison of Classification Results

*(Embed Matplotlib Output Images Here)*
*   `[Insert: Bar Chart comparing Accuracy, Precision, Recall, and F1-Score across Set A, Set B, and Set C]`
*   `[Insert: Confusion Matrix - Set A (Raw Images)]`
*   `[Insert: Confusion Matrix - Set B (Filtered Images)]`
*   `[Insert: Confusion Matrix - Set C (Edge Images)]`

---

## 6. Discussion

**Question 1: Edge Detection and Noise**
The Laplacian edge detector was the most sensitive to noise. Because it calculates the second derivative of pixel intensities without any inherent smoothing, it drastically amplified high-frequency changes, turning minor skin textures and artificial noise spikes into overwhelming false edges.

**Question 2: Effect of Filtering**
Median filtering successfully removed Salt-and-Pepper (impulse) noise without compromising edge sharpness, allowing detectors to find the true lesion boundaries. Gaussian filtering effectively suppressed Gaussian noise but acted as a low-pass filter, which inherently blurred the structural edges and resulted in thicker, less localized edge maps.

**Question 3: Canny Parameters**
Lowering the thresholds ($Low=30, High=100$) increased the number of detected edges but introduced significant noise by capturing irrelevant skin textures. Raising the thresholds ($Low=100, High=200$) reduced the edge count drastically, but caused the genuine lesion boundaries to become broken and fragmented. The 50/150 split provided the optimal balance.

**Question 4: Edge Maps and Classification**
Using edge-only images significantly reduced classification accuracy compared to raw images (dropping from 42.37% to 16.95% for EfficientNet-B0). Edge detection strips away crucial diagnostic information, leaving the models with only structural outlines that are often similar across different disease classes. Classical ML classifiers relying on deep feature extraction also performed near random-guessing levels (11-14%) on edge maps.

**Question 5: Information Loss**
When an image is converted to an edge map, critical dermatological features are lost. This includes color variations, melanin distribution, lesion shading, and subsurface textures—all of which are primary indicators used by both dermatologists and CNNs to differentiate between conditions like melanoma and benign keratosis.

**Question 6: Classical vs. Deep Features**
Classical edge detectors force a handcrafted, rigid spatial bias onto the image. Deep learning models, through convolutional layers, learn their own optimal, task-specific feature extractors dynamically. Allowing a CNN to process raw pixels enables it to identify complex, hierarchical patterns (combining color, texture, and edges) rather than being bottlenecked by a human-engineered edge map.

**Question 7: Best Representation**
Based on the results across Labs 01-03, the **Filtered images (Average Filter)** produced the most useful classification results, achieving the highest accuracy (43.22% after 1 Epoch). This representation retained all necessary color and texture information while applying just enough smoothing to eliminate background noise, providing the cleanest signal for the CNN to learn from.

---

## 7. Conclusion
This laboratory demonstrated that while classical edge detectors like Canny are highly effective at extracting structural boundaries when paired with appropriate noise-reduction filters, they are suboptimal as standalone inputs for deep learning and classical classifiers in dermatological applications. Converting skin lesion images to edge maps strips away critical color, texture, and shading information, causing classification accuracy to drop significantly across all model architectures (EfficientNet, ResNet, SVM, Random Forest, KNN). The highest performance is achieved by providing the models with mildly filtered raw images, allowing convolutional layers to learn their own optimal, task-specific feature hierarchies dynamically.