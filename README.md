cat > README.md <<'EOF'

\# Leaf Shape Classification Using GNU Octave



A machine learning project that classifies leaf images into six plant categories using shape, colour, and texture features.



\## Project Overview



This project extracts visual features from leaf images and uses the K-Nearest Neighbors (KNN) algorithm to predict the corresponding leaf category.



It is implemented using GNU Octave and its Image package.



\## Classes



The system supports six leaf categories:



\- Mango

\- Neem

\- Custard Apple

\- Hibiscus

\- Lemon

\- Betel Leaf



\## Technologies Used



\- GNU Octave

\- Image Processing Package

\- K-Nearest Neighbors (KNN)

\- Feature Extraction

\- Image Processing



\## Features



\- Image preprocessing and segmentation

\- Shape-based feature extraction

\- Colour feature extraction

\- Texture feature extraction

\- KNN classification

\- Leave-One-Out cross-validation

\- Prediction of new leaf images

\- Feature comparison experiments



\## Model Evaluation



| Evaluation | Result |

|---|---:|

| Dataset images | 182 |

| Leaf categories | 6 |

| LOO correct predictions | 160 |

| LOO accuracy | 87.91% |



The project also contains a small external test set with six recorded predictions. All six were classified correctly in that test run. This is a limited test set and should not be interpreted as proof of general performance.



\## Project Structure



```text

Leaf\_shape\_classification/

├── dataset/

│   ├── Mango/

│   ├── Neem/

│   ├── Custard\_Apple/

│   ├── Hibiscus/

│   ├── Lemon/

│   └── Betel\_Leaf/

├── output/

├── RESULTS/

├── tests/

├── main.m

├── train\_model.m

├── predict\_leaf.m

├── preprocess\_leaf.m

├── extract\_features.m

├── extract\_features\_baseline.m

├── compare\_feature.m

├── compare\_features.m

└── test\_all\_leaves.m

