# Leaf Shape Classification Using GNU Octave

An Image Processing and Machine Learning project that classifies leaf images into six plant categories using shape, colour, and texture feature extraction.

## Project Overview

This project uses digital image processing techniques to preprocess leaf images, extract important visual features, and classify leaves using the K-Nearest Neighbors (KNN) algorithm.

It is developed using GNU Octave and its Image Processing Package.

## Classes

The system supports six leaf categories:

- Mango
- Neem
- Custard Apple
- Hibiscus
- Lemon
- Betel Leaf

## Technologies Used

- GNU Octave
- Image Processing Package
- Digital Image Processing
- K-Nearest Neighbors (KNN)
- Feature Extraction
- MATLAB-compatible scripting

## Features

- Image preprocessing and segmentation
- Shape-based feature extraction
- Colour feature extraction
- Texture feature extraction
- KNN classification
- Leave-One-Out cross-validation
- Prediction of new leaf images
- Feature comparison experiments

## Model Evaluation

| Evaluation | Result |
|---|---:|
| Dataset images | 182 |
| Leaf categories | 6 |
| LOO correct predictions | 160 |
| LOO accuracy | 87.91% |

The project also includes a small external test set with six recorded predictions. All six were classified correctly in that test run. This limited test set should not be interpreted as proof of general performance.

## Project Structure

```text
Leaf_shape_classification/
├── dataset/
│   ├── Mango/
│   ├── Neem/
│   ├── Custard_Apple/
│   ├── Hibiscus/
│   ├── Lemon/
│   └── Betel_Leaf/
├── output/
├── RESULTS/
├── tests/
├── main.m
├── train_model.m
├── predict_leaf.m
├── preprocess_leaf.m
├── extract_features.m
├── extract_features_baseline.m
├── compare_feature.m
├── compare_features.m
└── test_all_leaves.m
```

## How to Run

1. Install GNU Octave.
2. Install the Image Processing Package.
3. Clone or download this repository.
4. Open the project folder in GNU Octave.
5. Run the main script:

```matlab
main
```

## Results

The project includes prediction outputs, model files, evaluation results, and screenshots in the `output/` and `RESULTS/` folders.

## Author

**Swetha A**

BCA Student | Developer

GitHub: [@swetha-a01](https://github.com/swetha-a01)
