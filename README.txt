LEAF CLASSIFICATION - OCTAVE FEATURE IMPROVEMENT

Files included:
- extract_features.m : rotation-tolerant shape and colour features
- train_model.m      : compares KNN values K=1,3,5,7,9 with leave-one-out validation

IMPORTANT:
1. Copy these two .m files into your existing project folder, replacing the old files.
2. Keep your current preprocess_leaf.m and predict_leaf.m unchanged for this first test.
3. Keep the dataset folder structure unchanged:
   dataset/Mango, dataset/Neem, dataset/Betel_Leaf,
   dataset/Custard_Apple, dataset/Hibiscus, dataset/Lemon
4. In Octave, set the current directory to the project root, then run:
      pkg load image
      clear functions
      train_model
      predict_leaf
5. Retraining is required because the feature vector has changed. train_model saves a new output/leaf_model.mat.
6. If Octave reports a feature-count mismatch, make sure predict_leaf loads the newly generated output/leaf_model.mat and that all files are from the same project folder.

This is an experiment, not a guaranteed accuracy improvement. The model compares candidate K values on LOO; that score is for tuning and is not an independent test score. Please record predictions on separate external test photos too.
