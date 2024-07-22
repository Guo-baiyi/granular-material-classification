# Granular Materials Classification and Particle Size Identification

This project aims to classify granular materials and identify particle size distribution using deep learning techniques. Specifically, we use Convolutional Neural Networks (CNN) to predict the average particle size and standard deviation from images of granular materials.

## Project Overview

### Objectives
1. Segment and extract features from images of granular materials.
2. Train a CNN model to predict the average particle size and standard deviation.
3. Use the trained model to analyze new images and provide accurate particle size predictions.

### Dataset
The dataset consists of images of various granular materials. Each image is labeled with the average particle size and standard deviation. The images are pre-processed and augmented to improve the robustness of the model.

### Methodology
1. **Segmentation and Feature Extraction**: Using the SAM (Segment Anything Model) to segment particles and extract geometric features.
2. **Model Training**: Training a CNN model using the extracted features and corresponding labels.
3. **Prediction**: Using the trained model to predict particle size metrics for new images.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
