# Lab Mid Exam: HOG vs SIFT on Blood-Cell Images
**Name:** Talal Ahmed Tarar  
**Registration Number:** FA23-BAI-031  

### Personal Settings (Derived from Registration Number `031`)
| Setting | Rule | My Value |
| :--- | :--- | :--- |
| **Random seed** | Last two digits of registration number (`031`) | **`31`** |
| **HOG cell size** | Last digit odd (`1`) $\rightarrow 16 \times 16$ | **`16 x 16`** |
| **SIFT grid size** | Last digit `0` to `4` (`1`) $\rightarrow 4 \times 4$ grid (16 keypoints) | **`4 x 4` (16 keypoints)** |

---
## Q2. HOG + SVM (6 Marks)

### Q2(1): Hand Calculation of HOG Feature Length
* **Image Size:** $128 \times 128$ pixels
* **HOG Cell Size (Personal Setting for Odd Digit `1`):** $16 \times 16$ pixels
* **Number of Cells along Width & Height:**
  $$\text{Cells per dimension} = \frac{128}{16} = 8 \implies 8 \times 8 \text{ grid of cells}$$
* **Block Size:** $2 \times 2$ cells per block (with a default stride of $1$ cell)
* **Number of Blocks along Width & Height:**
  $$\text{Blocks per dimension} = (8 - 2) + 1 = 7 \implies 7 \times 7 = 49 \text{ total blocks}$$
* **Features per Block:**
  $$\text{Cells per block} \times \text{Orientations} = (2 \times 2) \times 9 = 4 \times 9 = 36 \text{ values per block}$$
* **Total HOG Feature Vector Length:**
  $$\text{Total Features} = 49 \text{ blocks} \times 36 \text{ features/block} = \mathbf{1764}$$

---
## Q3. SIFT + SVM (7 Marks)

### Q3(1): Why a Different Number of Keypoints per Image is a Problem for an SVM
A standard Support Vector Machine (`LinearSVC`) calculates the dot product $\mathbf{w}^T \mathbf{x} + b$ in a fixed-dimensional vector space, so it strictly requires every input image to be represented by a feature vector $\mathbf{x}$ of the exact same length (and images with zero keypoints produce no descriptor at all).

### Q3(2): Hand Calculation of Fixed-Grid SIFT Feature Length
* **SIFT Grid Size (Personal Setting for Last Digit `1`):** $4 \times 4$ regular grid over the $128 \times 128$ image
* **Total Number of Grid Keypoints:**
  $$\text{Grid Points} = 4 \times 4 = 16 \text{ keypoints}$$
* **Descriptor Size per SIFT Keypoint:**
  Each SIFT keypoint neighborhood is divided into a $4 \times 4$ sub-region grid with an 8-bin orientation histogram per sub-region ($4 \times 4 \times 8 = 128$ values).
* **Total SIFT Feature Vector Length:**
  $$\text{Total Features} = \text{Grid Points} \times 128 = 16 \times 128 = \mathbf{2048}$$

---
## Q4. Robustness (6 Marks)

### Q4(1): Pre-Experiment Prediction (Rotation vs. Noise)
* **Under Rotation ($20^\circ$):** I predict **HOG + SVM** will lose more accuracy under a $20^\circ$ rotation because HOG computes strict, unaligned gradient orientation histograms locked to static cells without dominant-angle rotation alignment, whereas SIFT descriptors apply Gaussian spatial weighting around each grid center to soften minor rotational shifts.
* **Under Gaussian Noise ($\sigma = 0.05$):** I predict **SIFT + SVM** will lose more accuracy under high-frequency Gaussian noise because SIFT relies on fine $4 \times 4$ sub-region gradient histograms with large unnormalized descriptor magnitudes that get heavily distorted by random pixel-level intensity spikes, whereas my HOG configuration pools gradients over much larger $16 \times 16$ pixel cells.


### Q4(2) & Q4(3): Completed Robustness Table and 3-Sentence Evaluation

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
| :--- | :---: | :---: | :---: |
| **HOG + SVM** | *72.50%* | *69.17%* | *54.17%* |
| **SIFT + SVM** | *73.33%* | *72.50%* | *74.17%* |

**Evaluation of Prediction (3 Sentences):**
1. On the clean test set, HOG + SVM achieved 72.50% accuracy while SIFT + SVM achieved 73.33% accuracy.
2. Under a 20-degree rotation, HOG + SVM changed from 72.50% to 69.17% (a drop of 3.33%), whereas SIFT + SVM went from 73.33% to 72.50% (a drop of 0.83%), showing that HOG + SVM lost more accuracy because fixed-grid descriptors without canonical orientation assignment shift across spatial bins when rotated.
3. Under Gaussian noise (sigma=0.05), HOG + SVM went from 72.50% to 54.17% (a drop of 18.33%) while SIFT + SVM went from 73.33% to 74.17% (a drop of -0.83%), confirming that HOG + SVM was more sensitive to pixel-level noise whereas spatial pooling over 16x16 HOG cells smoothed out high-frequency perturbations.*


---
## Q5. Short Answers and AI Log (3 Marks)

### Q5(1): Conceptual Questions
* **(a) Why does HOG use block normalisation?**  
  Local gradient magnitudes are highly sensitive to changes in overall image illumination and foreground-background contrast. By grouping adjacent cells into overlapping $2 \times 2$ blocks and normalizing their combined histogram vector (using `L2-Hys`), HOG makes the feature descriptors invariant to local lighting and shadowing variations while preserving relative edge strength.
* **(b) Why is it wrong to choose $C$ using the test set?**  
  Selecting the regularization hyperparameter $C$ on the test set causes **data leakage (test-set overfitting)**, as the model's hyperparameters become specifically tuned to the quirks of the test distribution. Consequently, the test accuracy no longer provides an unbiased estimate of how the classifier will generalize to truly unseen clinical blood-cell images.

### Q5(2): Most Confused Classes and the Impact of Grayscale
* **Most Confused Classes:** Across our confusion matrices, the models most frequently confused **eosinophil (1) and monocyte (5)**
* **Why Grayscale Makes This Harder:** I think since both eosinophils and monocyte have almost similar overall cell diameter meaning their spatial edge gradients (captured by HOG and SIFT) look nearly identical in structure. Normally pathologists distinguish them instantly by **color** which is completely destroyed when converting RGB channels to single-channel grayscale.

### Q5(3): AI Log
| My Prompt (One Line) | What I Changed | How I Checked the Result Was Correct |
| :--- | :--- | :--- |
| *"Write a Python pipeline in Colab to load 4 classes from BloodMNIST zip, resize to 128x128 grayscale, and extract HOG (16x16 cell) and 4x4 fixed-grid SIFT features for LinearSVC."* | Updated the personal parameters to `SEED = 31`, `HOG_CELL_SIZE = (16, 16)`, and `SIFT_GRID_SIZE = (4, 4)` with `size=32` keypoints centered at `(16, 48, 80, 112)`, and added `BORDER_REFLECT_101` during $20^\circ$ rotation. | Verified that `X_train_hog.shape[1]` printed `1764` and `X_train_sift.shape[1]` printed `2048` (matching hand calculations), checked class counts (`100` train, `30` val, `30` test per class), and visually inspected the $4 \times 4$ image grid, HOG plots, and noisy/rotated test images. |
