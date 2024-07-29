# Granular Materials Classification and Particle Size Identification

## Introduction

This project aims to classify granular materials and identify particle size distribution using deep learning techniques. We employ Convolutional Neural Networks (CNN) and the Segment Anything Model (SAM) to predict the average particle size and standard deviation from images of granular materials.

## Complete Project Overview

![未命名绘图 drawio (1)](https://github.com/user-attachments/assets/d07802a1-9712-4fba-b35f-c2ac5ce1ff33)


The primary goal of this project is to automate the process of particle size detection in a material pile. This is achieved using a Luxonis OAK-D Pro camera mounted on a mobile robot. The camera's depth sensing capability ensures it maintains a fixed distance of 30mm from the material pile, capturing an RGB image at this distance. A pre-trained CNN model is then used to predict particle sizes from the captured images.




## Particle Size Detection Approaches


### 1. **Regression-based Approach**:
1. Pre-process Image
2. Extract Features
3. Train and Apply Regression Model
4. Predict Particle Size
5. Post-process and Output Results




##### Methodology
- **Model Architecture**: A modified ResNet50 convolutional neural network (CNN) was employed. The fully connected layer of the ResNet50 was adjusted to output a single continuous value representing the predicted grain size.
- **Data Preparation**: The dataset consisted of images of granular materials with known grain sizes. These images were preprocessed to ensure consistency in size and normalized using specific mean and standard deviation values ([0.4312, 0.4072, 0.3674], [0.1762, 0.1749, 0.1686]).
- **Training**: The model was trained to minimize the mean squared error (MSE) between the predicted and actual grain sizes. The training process involved splitting the dataset into training and validation sets, applying data augmentation techniques such as random cropping, flipping, and rotation.
- **Evaluation**: The performance of the regression model was evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).

##### Results
- The regression model provided a continuous prediction of grain size with reasonable accuracy.
- Metrics such as MAE and R² were used to quantify the model's performance, indicating the model's effectiveness in predicting grain size based on image data.

##### Advantages
- **Simplicity**: Direct prediction of grain size as a continuous variable simplifies the process.
- **Efficiency**: Once trained, the model can quickly provide grain size predictions for new images.

##### Disadvantages
- **Data Dependency**: The accuracy of predictions heavily depends on the quality and quantity of the training data.
- **Generalization**: The model might struggle to generalize to images with grain sizes significantly different from those in the training set.

### 2. **Classification-based Approach**:
    - Pre-process Image
    - Apply Classification Model
    - Classify Particles by Size Range
    - Post-process and Output Results


### 3. **Segmentation-based Approach**:
1. Pre-process Image
2. Apply Segmentation Model - Segment Anything Model(SAM)
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/86446666-6a8f-4b8a-9443-53521f7cebca" alt="original picture" width="30%"/>
    <img src="https://github.com/user-attachments/assets/2f8d2cc6-4aaf-4f1d-af34-364ba35a0d2c" alt="Colored Segments" width="30%"/>
    <img src="https://github.com/user-attachments/assets/3c2ae483-e64e-407c-a944-777d7bbeb184" alt="Contours" width="30%"/>
</div>

3. Identify and Measure Particles - OpenCV


4. Post-process and Output Results


![image](https://github.com/user-attachments/assets/b9fa4502-3fdb-4767-8616-007653897915)


#### Methodology
- **Model Architecture**: The Segment Anything Model (SAM) (https://github.com/facebookresearch/segment-anything) was utilized for initial image segmentation. SAM is capable of identifying and segmenting individual grains within an image.
- **Data Preparation**: Images were processed to ensure they were suitable for segmentation. This included resizing, normalization, and augmentation to create diverse training data.
- **Segmentation and Measurement**: SAM was applied to segment the grains in the images. Post-segmentation, OpenCV techniques were employed to measure the size of each segmented grain. The measurements were then averaged to obtain an overall grain size for the image.
- **Evaluation**: The segmentation accuracy and the grain size measurements were compared to ground truth values to evaluate performance.

#### Results and Summary of Issues in Grain Size Detection with SAM
- The segmentation approach provided detailed information about the distribution and size of individual grains within an image.
- This method allowed for the visualization of segmented grains, aiding in the qualitative assessment of the segmentation accuracy.

##### Accuracy with Small Particles (< 5mm):

- SAM struggles to accurately segment and measure particles smaller than 5mm.
- These small particles often fall below the resolution limits of the segmentation model, resulting in missed detections or inaccurate measurements.
##### Accuracy with Large Particles (> 40mm):

- For particles larger than 40mm, SAM faces difficulties due to occlusions and overlapping particles.
- The presence of other particles in the foreground or background can cause the model to underestimate the actual size of these large particles.
##### Advantages
- **Detail-Oriented**: Provides detailed insights into the size and distribution of individual grains.
- **Flexibility**: Can handle images with varying grain sizes and distributions more effectively.

##### Disadvantages
- **Complexity**: The segmentation and measurement process is more complex and computationally intensive compared to direct regression.
- **Processing Time**: Segmenting each grain and measuring its size takes more time, especially for high-resolution images with many grains.






## Execution Flow for real-world implementation
1. **Innock-Robot Moves to Material Pile**
2. **Camera Ensures 30mm Distance**
3. **Capture RGB Image**
4. **Process Image with Pre-trained CNN Model**
5. **Predict Particle Size**
6. **Display Results**




### Dataset
The dataset consists of images of various granular materials. Each image is labeled with the average particle size and standard deviation. The images are pre-processed and augmented to improve the robustness of the model.

### Methodology
1. **Segmentation and Feature Extraction**: Using the SAM (Segment Anything Model) to segment particles and extract geometric features.
2. **Model Training**: Training a CNN model using the extracted features and corresponding labels.
3. **Prediction**: Using the trained model to predict particle size metrics for new images.


#### Comparative Analysis

- **Accuracy**: Both methods demonstrated good accuracy, with the regression model being faster and simpler, while the segmentation approach provided more detailed information.
- **Scalability**: The regression model is more scalable for large datasets due to its efficiency in processing, whereas the segmentation approach, while more informative, requires significant computational resources.
- **Application Suitability**: The choice of method depends on the specific requirements of the application. For quick and continuous predictions, the regression model is suitable. For detailed analysis and visualization, the segmentation approach is preferable.

### Conclusion

By exploring these approaches, the project highlights the potential and versatility of computer vision techniques in analyzing granular materials. Both methods have their unique strengths and can be applied based on the specific needs of the analysis, paving the way for further research and development in this field.









