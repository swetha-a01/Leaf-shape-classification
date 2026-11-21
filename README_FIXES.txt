LEAF CLASSIFICATION CODE FIXES

1. Replace main.m with the supplied main.m.
   Fix: menu option 1 called train_classifier(), but the provided training function is train_model().

2. Replace test_all_leaves.m with the supplied test_all_leaves.m.
   Fix: default test directory is now 'tests' instead of 'test'.

IMPORTANT: predict_leaf.m is NOT included or changed.

Keep the following files in the project root with these exact names:
  main.m
  train_model.m
  predict_leaf.m
  preprocess_leaf.m
  extract_features.m

Keep folders:
  dataset/Mango
  dataset/Neem
  dataset/Betel_Leaf
  dataset/Custard_Apple
  dataset/Hibiscus
  dataset/Lemon
  tests/
  output/

Run from the project root in Octave:
  pkg load image
  clear functions
  main

Choose 1 to train. Only after training finishes successfully, choose 2 to predict.

The uploaded extract_features(2).m is the 122-feature texture version. If this is the version you want to use, it must be named extract_features.m in the project root. Training and prediction must use the same feature extractor. A model trained with the 104-feature baseline is not compatible with the 122-feature version, so retraining is needed when switching feature versions.

I could inspect the source files, but GNU Octave was not available in this checking environment, so the scripts have not been executed here.
