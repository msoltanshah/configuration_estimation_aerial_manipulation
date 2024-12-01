# Aerial Manipulation Configuration Estimation Using RGB Images

Aerial manipulation systems often face challenges with uncertainty and errors in estimating their configuration, specifically the pose of the UAV and its manipulator arm. This project aims to train an aerial robot to estimate its pose autonomously using only RGB images. The setup involves the robot hovering near a highlighted cylindrical object, utilizing an eye-to-hand camera for QR code tracking and an eye-in-hand camera for generating RGB data.

This repository provides tools and models for configuration estimation in aerial manipulation tasks, leveraging deep learning techniques. The primary focus is on regression-based neural networks, complemented by visualization and analysis utilities.

## Features

- Regression with Deep Neural Networks:

  - Implements a regression model for configuration estimation.

  - Includes early stopping for optimized training.

- Model Visualization:

  - Visualize feature maps and filters from the trained neural network.

- Prediction Utility:

  - Predict configurations using the trained model on new data.

- Dataset:

  - Includes train/ and test/ directories with sample images.

## Repository Structure

configuration_estimation_aerial_manipulation/

├── regression_deep_nn.py          # Implements the regression model

├── model_prediction.py            # Prediction script using trained models

├── visualize_feature_maps.py      # Script to visualize feature maps

├── visualize_filters.py           # Script to visualize learned filters

├── output_test.csv                # Output results for test data

├── output_train.csv               # Output results for training data

├── regression_early_stopping_1.h5 # Trained model file

├── regression_early_stopping_1_weights.h5 # Trained model weights

├── train/                         # Directory containing training images

└── test/                          # Directory containing test images

## Requirements

- Python 3.8+

- Required Libraries:

  - TensorFlow / Keras

  - Matplotlib

  - NumPy

  - Pandas

## Dataset

The train/ and test/ directories contain sample images for training and testing.

Modify these directories to use your custom datasets.
