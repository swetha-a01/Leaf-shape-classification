clc;
clear;
close all;

% Load Image Processing package
pkg load image;

fprintf("\n");
fprintf("====================================================\n");
fprintf("        LEAF SHAPE CLASSIFICATION SYSTEM\n");
fprintf("====================================================\n");

fprintf("\n1. Train Classifier\n");
fprintf("2. Predict New Leaf\n");
fprintf("3. Exit\n");

choice = input("\nEnter your choice: ");

switch choice
  case 1
    % The training function in this project is named train_model.
    train_model();
  case 2
    % Keep the existing prediction function unchanged.
    predict_leaf();
  case 3
    fprintf("\nProgram ended.\n");
  otherwise
    fprintf("\nInvalid choice! Enter 1, 2, or 3.\n");
endswitch
