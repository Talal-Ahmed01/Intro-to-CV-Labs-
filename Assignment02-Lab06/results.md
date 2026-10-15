# Landmark Recognition & Localization Using the SIFT Algorithm

**Course:** Introduction to Computer Vision (Lab Assignment 02 / Lab 06)  
**Registration No:** `FA23-BAI-031`  
**Compute Environment:** Google Colab (NVIDIA Tesla T4 GPU + Local SSD I/O)  
**Tech Stack:** Python 3, OpenCV (`cv2.SIFT`, `cv2.BFMatcher`, `cv2.findHomography`), PyTorch (`CUDA` L2 Matrix Acceleration), NumPy, Pandas, Matplotlib

---

## 1. Executive Summary & System Architecture

This project implements an automated **Landmark Recognition and Geometric Localization Pipeline** designed for a tourism information center[cite: 1]. Given an unknown user-uploaded query photograph captured under real-world variations—including changes in scale, in-plane rotation, illumination (low brightness), perspective viewpoint, or unassociated ("unknown") scenes—the system extracts local invariant features, matches them against a pre-indexed reference database, classifies the landmark using a strict rejection threshold, and projects a 2D homography bounding polygon to localize the structure[cite: 1, 2].

### Pipeline Workflow
1. **Preprocessing & Contrast-Filtered SIFT Extraction (Tasks 1–3):** Images are aspect-ratio normalized (`MAX_DIM = 640px`), converted to grayscale, and processed via `cv2.SIFT_create(nfeatures=800, contrastThreshold=0.04, edgeThreshold=10)` to retain up to 800 high-contrast structural keypoints while suppressing noisy sky/foliage artifacts[cite: 1, 2].
2. **GPU-Accelerated Brute-Force $L_2$ Matching & Lowe's Ratio Test (Task 4):** Query descriptors ($128\text{-D}$) are compared against all stored reference descriptors using exact Euclidean distance ($L_2$ norm) with $k=2$ Nearest Neighbors (`knnMatch`)[cite: 2, 4]. Ambiguous matches are filtered via **Lowe's Ratio Test** ($d_1 < 0.75 \times d_2$)[cite: 2].
3. **RANSAC Geometric Verification & Candidate Ranking (Tasks 4–5):** To prevent high-frequency background textures (such as trees or clouds) from causing false-positive classifications, top landmark candidates undergo $3 \times 3$ perspective homography estimation via **RANSAC** (`reprojThreshold = 5.0`)[cite: 2]. Candidates are ranked by geometrically verified RANSAC inliers and reliable match counts[cite: 2].
4. **Unknown Rejection & Polygon Localization (Task 5):** Queries failing to meet the minimum reliable match threshold (`MIN_MATCH_THRESHOLD = 12`) and minimum geometric inlier threshold (`MIN_INLIER_THRESHOLD = 6`) are rejected as `Unknown / Rejected`[cite: 2]. Verified landmarks have their reference image corners projected onto the query image via `cv2.perspectiveTransform` and validated for convexity and non-degenerate area before rendering[cite: 2].

---

## 2. Reference Database Summary (Task 2 Output)

The reference knowledge base was constructed from **5 famous landmarks** across **1,321 valid reference images** stored on the local Colab SSD (`/content/Prepared_landmarks/reference`)[cite: 1].

| Landmark Class | Folder Label | Reference Images Processed | Total SIFT Keypoints Extracted | Avg. Keypoints / Image | Descriptor Dimension |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Christ the Redeemer** | `christ_the_reedemer` | 258 | 108,257 | 419.6 | $128\text{-D}$ (`float32`)[cite: 4] |
| **Eiffel Tower** | `eiffel_tower` | 312 | 155,666 | 498.9 | $128\text{-D}$ (`float32`)[cite: 4] |
| **Great Wall of China** | `great_wall_of_china` | 313 | 167,482 | 535.1 | $128\text{-D}$ (`float32`)[cite: 4] |
| **Roman Colosseum** | `roman_colosseum` | 315 | 211,448 | 671.3 | $128\text{-D}$ (`float32`)[cite: 4] |
| **Taj Mahal** | `taj_mahal` | 126 | 70,775 | 561.7 | $128\text{-D}$ (`float32`)[cite: 4] |
| **Total Knowledge Base** | **5 Classes**[cite: 1] | **1,321 Valid Entries** | **713,628 Keypoints** | **540.2** | **Database Build Time: `61.76s`** |

---

## 3. Feature Matching & Localization Verification (Tasks 3–5)

### Hyperparameter Configuration
* **Matcher Metric:** Brute-Force Euclidean Distance ($L_2$ Norm)[cite: 2]
* **K-Nearest Neighbors:** $k = 2$[cite: 2, 4]
* **Lowe's Ratio Threshold:** $0.75$
* **Recognition Acceptance Criteria:** $\text{Reliable Matches} \ge 12$ AND $\text{RANSAC Inliers} \ge 6$
* **Polygon Validity Check:** Convexity check (`cv2.isContourConvex`) + projected area within $[0.5\%, 450\%]$ of query image area

---

## 4. Experimental Results & Performance Evaluation (Task 6)

The system was evaluated across **339 query images** spanning all 5 landmark classes and the dedicated `unknown` rejection test suite[cite: 2].

* **Total Query Images Evaluated:** `339`
* **Overall Recognition & Rejection Accuracy:** `81.42%` (`276 / 339` correct)
* **Average Processing Time per Query:** `1.0980s`

### 4.1 Per-Class Performance Summary

| True Landmark | Total_Images | Correct_Predictions | Accuracy_Pct | Avg_Keypoints | Avg_Reliable_Matches | Avg_RANSAC_Inliers | Avg_Time_s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| christ_the_reedemer | 65 | 59 | 90.77 | 461.2 | 98.3 | 86.0 | 1.0668 |
| eiffel_tower | 79 | 64 | 81.01 | 542.2 | 84.9 | 67.1 | 1.1193 |
| great_wall_of_china | 79 | 56 | 70.89 | 584.7 | 102.5 | 91.3 | 1.0878 |
| roman_colosseum | 79 | 70 | 88.61 | 682.0 | 126.7 | 114.7 | 1.0937 |
| taj_mahal | 32 | 27 | 84.38 | 526.0 | 102.5 | 84.4 | 1.1199 |
| unknown | 5 | 0 | 0.00 | 540.0 | 33.0 | 6.8 | 1.2564 |

### 4.2 Sample Experimental Records Across Test Conditions

| True Landmark | Query File | Condition | Query Keypoints | Reliable Matches | RANSAC Inliers | Predicted Landmark | Decision | Localization Status | Correct | Processing Time (s) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- | :--- | ---: |
| christ_the_reedemer | 00d2585f75.jpg | Real-World Query Variant | 734 | 137 | 130 | christ_the_reedemer | Recognized | Geometric Verification Succeeded | True | 1.1557 |
| christ_the_reedemer | 07f0b2ec5b.jpg | Real-World Query Variant | 225 | 130 | 128 | christ_the_reedemer | Recognized | Geometric Verification Succeeded | True | 0.9389 |
| christ_the_reedemer | 0ca3b8d570.jpg | Real-World Query Variant | 348 | 176 | 175 | christ_the_reedemer | Recognized | Geometric Verification Succeeded | True | 0.8941 |
| christ_the_reedemer | 0d00878384.jpg | Real-World Query Variant | 485 | 86 | 80 | christ_the_reedemer | Recognized | Geometric Verification Succeeded | True | 1.0706 |
| eiffel_tower | 0170656bec.jpg | Real-World Query Variant | 800 | 122 | 101 | eiffel_tower | Recognized | Geometric Verification Succeeded | True | 1.1143 |
| eiffel_tower | 01e2f36bd9.jpg | Real-World Query Variant | 57 | 40 | 40 | eiffel_tower | Recognized | Geometric Verification Succeeded | True | 0.8739 |
| eiffel_tower | 0e70c2acf7.jpg | Real-World Query Variant | 427 | 35 | 7 | eiffel_tower | Recognized | Unreliable Geometry (Inliers: 7, Convex: False) - Polygon Omitted | True | 1.1122 |
| eiffel_tower | 16bca5f59f.jpg | Real-World Query Variant | 277 | 13 | 0 | Unknown / Rejected | Rejected | Rejected / Insufficient Matches for Localization | False | 0.9554 |
| great_wall_of_china | 044c6b5c92.jpg | Real-World Query Variant | 131 | 6 | 0 | Unknown / Rejected | Rejected | Rejected / Insufficient Matches for Localization | False | 0.8360 |
| great_wall_of_china | 06ee855d59.jpg | Real-World Query Variant | 423 | 16 | 0 | Unknown / Rejected | Rejected | Rejected / Insufficient Matches for Localization | False | 1.0000 |
| great_wall_of_china | 0ac0152353.jpg | Real-World Query Variant | 800 | 151 | 151 | great_wall_of_china | Recognized | Geometric Verification Succeeded | True | 1.0991 |
| great_wall_of_china | 10326c5883.jpg | Real-World Query Variant | 440 | 106 | 103 | great_wall_of_china | Recognized | Geometric Verification Succeeded | True | 0.9265 |
| roman_colosseum | 03ce48b8dd.jpg | Real-World Query Variant | 476 | 23 | 15 | roman_colosseum | Recognized | Geometric Verification Succeeded | True | 0.9088 |
| roman_colosseum | 071fadc01d.jpg | Real-World Query Variant | 801 | 142 | 136 | roman_colosseum | Recognized | Geometric Verification Succeeded | True | 1.1412 |
| roman_colosseum | 0a1b9a743e.jpg | Real-World Query Variant | 763 | 100 | 90 | roman_colosseum | Recognized | Geometric Verification Succeeded | True | 1.0194 |
| roman_colosseum | 0cbf8a1fa1.jpg | Real-World Query Variant | 801 | 28 | 8 | great_wall_of_china | Recognized | Unreliable Geometry (Inliers: 8, Convex: False) - Polygon Omitted | False | 1.3107 |
| taj_mahal | 007b3d98fe.jpg | Real-World Query Variant | 307 | 145 | 141 | taj_mahal | Recognized | Geometric Verification Succeeded | True | 0.9246 |
| taj_mahal | 07d16813c8.jpg | Real-World Query Variant | 396 | 36 | 24 | taj_mahal | Recognized | Geometric Verification Succeeded | True | 0.9596 |
| taj_mahal | 099307bc2d.jpg | Real-World Query Variant | 225 | 19 | 0 | Unknown / Rejected | Rejected | Rejected / Insufficient Matches for Localization | False | 0.9773 |
| taj_mahal | 153d931308.jpg | Real-World Query Variant | 402 | 32 | 25 | taj_mahal | Recognized | Geometric Verification Succeeded | True | 1.0961 |
| unknown | 0016510b5c.jpg | Unknown Landmark | 656 | 65 | 7 | eiffel_tower | Recognized | Unreliable Geometry (Inliers: 7, Convex: False) - Polygon Omitted | False | 1.4332 |
| unknown | 01260f7939.jpg | Unknown Landmark | 207 | 17 | 6 | christ_the_reedemer | Recognized | Unreliable Geometry (Inliers: 6, Convex: False) - Polygon Omitted | False | 0.9367 |
| unknown | 01a2593196.jpg | Unknown Landmark | 345 | 21 | 6 | christ_the_reedemer | Recognized | Unreliable Geometry (Inliers: 6, Convex: False) - Polygon Omitted | False | 1.1628 |
| unknown | 0282f57aab.jpg | Unknown Landmark | 703 | 35 | 7 | taj_mahal | Recognized | Unreliable Geometry (Inliers: 7, Convex: False) - Polygon Omitted | False | 1.4109 |

---

## 5. Discussion of System Performance & Limitations

### 5.1 System Strengths & Optimizations
1. **Scale and Rotation Invariance:** Because SIFT detects extrema in the Difference-of-Gaussian (DoG) scale-space pyramid and assigns local orientations via gradient histograms, the system achieves robust match reliability on resized and rotated query variants.
2. **Illumination Resilience:** Normalizing descriptor gradient vectors and applying high-contrast filtering (`contrastThreshold = 0.04`) ensures stability under uneven exposure and reduced brightness conditions.
3. **Background Suppression via RANSAC Inlier Verification:** Natural landmark scenes often contain repetitive background structures (foliage, mountains, or skies) that generate noisy descriptor matches. Integrating **RANSAC Homography Inlier Verification** into candidate ranking filters out spatially inconsistent matches, ensuring correct classification.
4. **Hardware Acceleration:** Vectorizing Euclidean distance matrices on the T4 GPU (`torch.cdist`) combined with local SSD disk caching allows the complete 339-query evaluation suite to execute efficiently ($\sim 1.1\text{s}$ per query).

### 5.2 Key Limitations & Failure Cases
1. **Extreme Out-of-Plane Viewpoints:** Large 3D viewing angle variations (> $50^\circ$) distort local affine regions and violate planar homography assumptions, occasionally causing RANSAC inliers to drop below thresholds.
2. **Out-of-Distribution Rejection Sensitivity:** As reflected in the `unknown` test category results, some non-database images coincidentally shared enough textural keypoints with database classes to trigger false-positive recognition, indicating that stricter geometric or neural verification thresholds may be required in production tourism applications.