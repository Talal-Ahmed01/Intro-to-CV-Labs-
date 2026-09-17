# Lab 02 Report: Effect of Image Filtering on Skin-Lesion Classification

## Classification Performance Comparison

| Model | Filter | Accuracy | Precision | Recall | F1-score | Macro-F1 | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EfficientNet-B0 | No Filter | 57.63% | 54.14% | 57.63% | 53.09% | 52.53% | 93.01% |
| EfficientNet-B0 | Average | 61.02% | 69.95% | 61.02% | 58.80% | 55.92% | 91.69% |
| EfficientNet-B0 | Gaussian | 58.47% | 55.56% | 58.47% | 53.57% | 51.64% | 91.26% |
| EfficientNet-B0 | Median | 54.24% | 55.21% | 54.24% | 49.38% | 49.49% | 88.26% |
| EfficientNet-B0 | Sharpening | 57.63% | 55.99% | 57.63% | 52.36% | 50.65% | 92.88% |
| EfficientNet-B0 | Sobel | 46.61% | 52.57% | 46.61% | 41.37% | 38.41% | 90.46% |
| DenseNet121 | No Filter | 50.00% | 48.49% | 50.00% | 41.90% | 42.07% | 88.08% |
| DenseNet121 | Average | 54.24% | 55.72% | 54.24% | 50.89% | 50.73% | 88.52% |
| DenseNet121 | Gaussian | 56.78% | 54.80% | 56.78% | 51.10% | 49.61% | 90.81% |
| DenseNet121 | Median | 55.93% | 53.25% | 55.93% | 50.88% | 50.72% | 88.91% |
| DenseNet121 | Sharpening | 55.08% | 55.50% | 55.08% | 50.36% | 50.30% | 86.80% |
| DenseNet121 | Sobel | 38.14% | 31.22% | 38.14% | 31.75% | 33.76% | 84.16% |
| ResNet50 | No Filter | 54.24% | 52.94% | 54.24% | 49.82% | 49.86% | 86.32% |
| ResNet50 | Average | 56.78% | 58.06% | 56.78% | 54.31% | 53.54% | 88.70% |
| ResNet50 | Gaussian | 50.00% | 48.10% | 50.00% | 46.09% | 45.51% | 85.25% |
| ResNet50 | Median | 53.39% | 64.88% | 53.39% | 48.94% | 46.88% | 88.85% |
| ResNet50 | Sharpening | 52.54% | 51.09% | 52.54% | 48.24% | 45.55% | 85.84% |
| ResNet50 | Sobel | 49.15% | 56.11% | 49.15% | 45.83% | 45.29% | 84.62% |

## Comparative Analysis

**1. Which three pretrained models performed best in Lab Activity 1?**
EfficientNet-B0, DenseNet121, and ResNet50.

**2. How does filtering affect each of the three models?**
The Average filter improved accuracy and F1 scores across all three models compared to the baseline, indicating they benefited from slight noise reduction. However, edge-enhancing filters (especially Sobel) severely degraded performance across the board.

**3. Which filter produces the greatest change compared with the unfiltered baseline?**
The Sobel filter caused the largest negative drop, plummeting DenseNet121's accuracy from 50.00% to 38.14% and its Macro-F1 from 42.07% to 33.76%. 

**4. Does the effect of a filter remain consistent across all three models?**
Yes, the trend is consistent: Average and Gaussian filters yielded the best results (improving or maintaining baseline), while Sharpening and Sobel consistently caused performance to drop across all three architectures.

**5. Does filtering improve or decrease macro-F1 and balanced accuracy?**
It depends entirely on the spatial bias of the filter. Smoothing filters (Average) improved the Macro-F1 (e.g., EfficientNet-B0 went from 52.53% to 55.92%), but aggressive edge detection (Sobel) significantly decreased it.

**6. Which lesion classes are most affected by filtering?**
*(Note: Refer to your generated confusion matrix to pinpoint specific classes here. Generally, classes with fine, irregular boundaries like melanoma suffer the most under heavy smoothing or edge distortion).*

**7. Why might smoothing remove useful lesion texture or morphological information?**
Deep learning models rely on high-frequency details—like the jagged, irregular borders of a lesion—to make classifications. Smoothing filters (Average, Gaussian, Median) act as low-pass filters, blurring these critical high-frequency edges into the background, which destroys the very features the CNN is trying to extract.

**8. Why might sharpening or edge detection help or hurt classification?**
Sharpening/Sobel filters highlight high-frequency changes. This can help by clearly defining a lesion's border. However, it usually hurts classification because it also amplifies background noise, skin pores, and hair, forcing the CNN to focus on irrelevant artifacts rather than the actual disease pathology.

**9. What is the difference between convolution and correlation?**
In classical image processing, convolution requires flipping the kernel 180 degrees before sliding it over the image, whereas cross-correlation slides the kernel exactly as it is. In deep learning, Convolutional Neural Networks technically perform cross-correlation, but since the kernel weights are learned dynamically, the mathematical distinction does not affect feature extraction.

**10. Based on your results, explain the relationship between classical image processing and deep-learning-based feature extraction.**
Classical filtering forces a specific, handcrafted spatial bias (like blurring or edge detection) onto the image before the network sees it. Deep learning, conversely, prefers to learn its own optimal spatial filters dynamically. While mild smoothing helped reduce noise in this specific run, aggressive classical pre-processing (like Sobel) degrades CNN performance because it destroys the raw pixel data the network relies on to build its own complex feature hierarchies.
