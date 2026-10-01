# Lab-Assignment01 Intro to Computer Vision
**Name:** Talal Ahmed Tarar  
**Registration Number:** FA23-BAI-031  

---

## Required Visualization & Results[cite: 8]

| Image | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
| :--- | :--- | :--- | :--- | :--- |
| **Image 1** | Gaussian (5x5) | Canny (100-200) | 10.50 | 99.98 |
| **Image 2** | Gaussian (5x5) | Canny (100-200) | 113.50 | 63.15 |
| **Image 3** | Gaussian (5x5) | Canny (100-200) | 27.50 | 112.61 |
| **Image 4** | Gaussian (5x5) | Canny (100-200) | 6.00 | 149.11 |
| **Image 5** | Gaussian (5x5) | Canny (100-200) | 92.00 | 329.53 |

---

## Final Comparison[cite: 8]

| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Original + Sobel** | Poor (amplifies noise) | Thick, blurred | Highly fragmented | Very Poor |
| **Original + Canny** | Poor (detects texture/hair) | Sharp but noisy | Messy / False positives | Poor |
| **Average + Sobel** | Moderate | Very thick | Acceptable but imprecise | Fair |
| **Average + Canny** | Moderate | Sharp, some detail lost | Acceptable | Good |
| **Gaussian + Sobel** | Good | Thick edges | Continuous but thick | Fair |
| **Gaussian + Canny** | Good | Sharp, continuous | Highly accurate | Very Good |
| **Median + Sobel** | Excellent (removes hair) | Thick edges | Continuous but thick | Good |
| **Median + Canny** | Excellent | Sharp, single-pixel | Most precise | Excellent |

---

## Questions to Answer

**1. Why is Gaussian filtering applied before Canny detection?**
Canny edge detection relies on calculating intensity gradients. Because derivatives are highly sensitive to high-frequency noise, Gaussian filtering is applied first to smooth the image, preventing the detector from falsely identifying sensor noise or normal skin textures as strong edges[cite: 9].

**2. How did the three Canny threshold settings affect the result?**
*   **50-100 (Low):** Too sensitive. It captures faint intensity transitions, outlining the lesion but also capturing irrelevant background noise like hair and skin pores[cite: 9].
*   **100-200 (Medium):** Optimal. It strikes a balance by ignoring weak background gradients while successfully retaining the strong, true boundary of the lesion[cite: 9].
*   **150-250 (High):** Too strict. It discards genuine but softer lesion boundaries, resulting in broken, unclosed contours[cite: 9].

**3. Which threshold produced the best lesion boundary?**
The **100-200** threshold setting produced the best boundary[cite: 9]. It successfully suppressed false edges from healthy skin while preserving a continuous, distinct edge around the pathological tissue[cite: 9].

**4. Why are edges useful for detecting skin lesions?**
Edges isolate the sharp transition zone between healthy skin and the pathological lesion tissue[cite: 9]. Localizing this boundary enables the calculation of critical clinical metrics required for diagnosis, such as lesion area, perimeter, and shape asymmetry[cite: 9].

**5. What problems did you observe in detecting the lesion boundary?**
The most prominent issues were artifacts like body hair and skin reflections[cite: 9]. Hair creates strong, sharp gradients that intersect with the lesion, tricking the edge detector[cite: 9]. Additionally, lesions with fading, low-contrast borders occasionally resulted in fragmented boundaries rather than a perfectly closed loop[cite: 9].

**6. How could your method be improved?**
The pipeline could be improved by adding a morphological preprocessing step (such as the DullRazor algorithm) specifically designed to digitally remove hair before edge detection[cite: 9]. Furthermore, utilizing Active Contour Models (Snakes) or adaptive local thresholding could help enforce smooth, fully closed boundaries across fading gradient gaps[cite: 9].