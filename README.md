# Granular Materials Classification and Particle Size Identification

## Introduction

Coarse-grained granular soils with particle sizes between 80 mm and 0.075 mm are critical construction and building materials，often for (specific purposes such as concrete production, road base, and landscaping). In the future deployment of unmanned intelligent construction sites, accurately identifying granular material information is crucial. This includes determining particle size, material type, particle shape, and current state.

This project aims to enhance the management of intelligent construction sites using mobile smart robots by designing and implementing a material recognition and classification module. This module utilizes machine learning and computer vision algorithms, integrated with the RGB-D camera installed on the mobile robots, to detect the particle size and type of granular materials.

In this research project, Convolutional Neural Networks (CNN) and the Segment Anything Model (SAM) are used to process and recognize the characteristics of the particulate materials in the photographs, in order to predict the average particle size and particle size distribution of the particulate materials.
## Complete Project Overview

![未命名绘图 drawio (1)](https://github.com/user-attachments/assets/d07802a1-9712-4fba-b35f-c2ac5ce1ff33)


The primary goal of this project is to automate the process of particle size detection in a material pile. This is achieved using a Luxonis OAK-D Pro camera mounted on a mobile robot. The camera's depth sensing capability ensures it maintains a fixed distance of 30mm from the material pile, capturing an RGB image at this distance. A pre-trained CNN model is then used to predict particle sizes from the captured images.




## Particle Size Detection Approaches


### 1. **Regression-based Approach**:
1. Pre-process Image
2. labeling grain size
    1. Labeling with granularity values extracted by SAM and opencv
    2. Labeling with the median of the product's particle size values



| ID   | Product Name                       | Number Range               | Grainsize (Median) | Grainsize (OpenCV)         | Photo Count |
|------|------------------------------------|----------------------------|--------------------|----------------------------|-------------|
| 1    | Marmorsplitt_7-12mm                | IMG_1394-IMG_1405          | 9.5                | 7.445                      | 1000        |
| 2    | Marmorkiesel_Weiss_7-15mm          | IMG_1406-IMG_1415          | 11                 | 9.310                      | 1000        |
| 3    | Marmorkiesel_Weiss_15-25mm         | IMG_1416-IMG_1425          | 20                 | <span style="color:red">11.410</span>  | 1000        |
| 4    | Marmorkiesel_Weiss_25-40mm         | IMG_1426-IMG_1435          | 32.5               | <span style="color:red">14.605</span>  | 1000        |
| 5    | Marmorkiesel_Weiss_40-60mm         | IMG_1436-IMG_1445          | 50                 | <span style="color:red">18.345</span>  | 1000        |
| 6    | Marmorkiesel_Schwarz_40-60mm       | IMG_1446-IMG_1455          | 50                 | <span style="color:red">21.870</span>  | 1000        |
| 7    | Marmorsplitt_Veronarot_9-12mm      | IMG_1457-IMG_1466          | 10.5               | 7.975                      | 1000        |
| 8    | Marmorsplitt_Donaublau_8-12mm      | IMG_1467-IMG_1476          | 10                 | 7.965                      | 1000        |
| 9    | Marmorkiesel_Gruen_15-25mm         | IMG_1477-IMG_1486          | 20                 | <span style="color:red">13.590</span>  | 1000        |
| 10   | Schieferplaettchen_22-40mm         | IMG_1487-IMG_1496          | 31                 | <span style="color:red">13.895</span>  | 1000        |
| 11   | Bruchsteine_Veronarot_30-60mm      | IMG_1498-IMG_1507          | 45                 | <span style="color:red">15.990</span>  | 1000        |
| 12   | Quarzkies_Rund_Hell_16-32mm        | IMG_1509-IMG_1518          | 24                 | <span style="color:red">14.280</span>  | 1000        |
| 13   | Quarzkies_Rund_Hell_8-16mm         | IMG_1519-IMG_1528          | 12                 | 9.030                      | 1000        |
| 14   | Kies_2-8mm                         | IMG_1529-IMG_1538          | 5                  | 5.150                      | 1000        |
| 15   | Kies_8-16mm                        | IMG_1539-IMG_1548          | 12                 | 9.345                      | 1000        |
| 16   | Kies_16-32mm                       | IMG_1549-IMG_1558          | 24                 | <span style="color:red">13.705</span>  | 1000        |
| 17   | Basaltsplitt_8-12mm                | IMG_1559-IMG_1568          | 10                 | 8.710                      | 1000        |
| 18   | Granitsplitt_8-12mm                | IMG_1569-IMG_1578          | 10                 | 8.580                      | 1000        |
| 19   | Splitt_2-5mm                       | IMG_1579-IMG_1588          | 3.5                | <span style="color:red">6.685</span>   | 1000        |
| 20   | Kies_1-3mm                         | IMG_1636-IMG_1645          | 2                  | <span style="color:red">19.305</span>  | 1000        |








3. Train and Apply Regression Model

4. Predict Particle Size
![image](https://github.com/user-attachments/assets/2157a696-3fc2-41b4-b9a9-f3e24415f77e)
![image](https://github.com/user-attachments/assets/f88afcdb-4d6e-443d-b059-ebcb8db91332)


5. Post-process and Output Results
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/7ad68d11-1312-491f-a257-441efd716114" alt="prediction_scatter_plot" width="30%"/>
    <img src="https://github.com/user-attachments/assets/11553956-3d67-481e-8b06-957c6109bf58" alt="loss_curves" width="30%"/>
    <img src="https://github.com/user-attachments/assets/daa741c1-3606-40fc-8a15-4b32f9bcf3d5" alt="val_mse_curves" width="30%"/>
</div>


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
1. Pre-process Image
2. Apply Classification Model(ResNet50, VGG16)
3. Classify Particles by Size Range

#### Small (0-12mm)
| Product Name               | Particle Size | Source         | Data Quantity |
|----------------------------|---------------|----------------|---------------|
| Kies_1-3mm                 | 1-3mm         | Aachen OBI     | 1000          |
| Splitt_2-5mm               | 2-5mm         | Aachen Bauhaus | 1000          |
| Kies_2-8mm                 | 2-8mm         | Aachen Bauhaus | 1000          |
| Marmorsplitt_7-12mm        | 7-12mm        | Aachen Bauhaus | 1000          |
| Marmorsplitt_Donaublau_8-12mm | 8-12mm    | Aachen Bauhaus | 1000          |
| granitsplitt_8-12mm        | 8-12mm        | Aachen Bauhaus | 1000          |
| Basaltsplitt_8-12mm        | 8-12mm        | Aachen Bauhaus | 1000          |

#### Medium (7-32mm)
| Product Name               | Particle Size | Source         | Data Quantity |
|----------------------------|---------------|----------------|---------------|
| Marmorkiesel_Weiss_7-15mm  | 7-15mm        | Aachen Bauhaus | 1000          |
| Quarzkies_Rund_Hell_8-16mm | 8-16mm        | Aachen Bauhaus | 1000          |
| Kies_8-16mm                | 8-16mm        | Aachen Bauhaus | 1000          |
| Marmorkiesel_Weiss_15-25mm | 15-25mm       | Aachen Bauhaus | 1000          |
| Marmorkiesel_Gruen_15-25mm | 15-25mm       | Aachen Bauhaus | 1000          |
| Quarzkies_Rund_Hell_16-32mm| 16-32mm       | Aachen Bauhaus | 1000          |
| Kies_16-32mm               | 16-32mm       | Aachen Bauhaus | 1000          |

#### Large (22-60mm)
| Product Name               | Particle Size | Source         | Data Quantity |
|----------------------------|---------------|----------------|---------------|
| Marmorkiesel_Weiss_25-40mm | 25-40mm       | Aachen Bauhaus | 1000          |
| Schieferplaettchen_22-40mm | 22-40mm       | Aachen Bauhaus | 1000          |
| Marmorkiesel_Weiss_40-60mm | 40-60mm       | Aachen Bauhaus | 1000          |
| Marmorkiesel_Schwarz_40-60mm| 40-60mm      | Aachen Bauhaus | 1000          |
| Bruchsteine_Veronarot_30-60mm | 30-60mm    | Aachen Bauhaus | 1000          |

4. Post-process and Output Results

![loss_accuracy_curves_12_epochs](https://github.com/user-attachments/assets/d57fa104-32da-4386-a09b-1cd9795fdf03)


### 3. **Segmentation-based Approach**:
1. Pre-process Image
2. Apply Segmentation Model - Segment Anything Model(SAM)
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/86446666-6a8f-4b8a-9443-53521f7cebca" alt="original picture" width="30%"/>
    <img src="https://github.com/user-attachments/assets/2f8d2cc6-4aaf-4f1d-af34-364ba35a0d2c" alt="Colored Segments" width="30%"/>
    <img src="https://github.com/user-attachments/assets/3c2ae483-e64e-407c-a944-777d7bbeb184" alt="Contours" width="30%"/>
</div>

3. Identify and Measure Particles - OpenCV

![schieferplaettchen_22-40mm_distribution](https://github.com/user-attachments/assets/994c1ed7-3df9-442c-a867-334305cf2106)


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
#### Product Information

| Product Name                    | Particle Size | Source         | Data Quantity |
|---------------------------------|---------------|----------------|---------------|
| Marmorsplitt_7-12mm             | 7-12mm        | Aachen Bauhaus | 1000          |
| Marmorkiesel_Weiss_7-15mm       | 7-15mm        | Aachen Bauhaus | 1000          |
| Marmorkiesel_Weiss_15-25mm      | 15-25mm       | Aachen Bauhaus | 1000          |
| Marmorkiesel_Weiss_25-40mm      | 25-40mm       | Aachen Bauhaus | 1000          |
| Marmorkiesel_Weiss_40-60mm      | 40-60mm       | Aachen Bauhaus | 1000          |
| Marmorkiesel_Schwarz_40-60mm    | 40-60mm       | Aachen Bauhaus | 1000          |
| Marmorsplitt_Veronarot_9-12mm   | 9-12mm        | Aachen Bauhaus | 1000          |
| Marmorsplitt_Donaublau_8-12mm   | 8-12mm        | Aachen Bauhaus | 1000          |
| Marmorkiesel_Gruen_15-25mm      | 15-25mm       | Aachen Bauhaus | 1000          |
| Schieferplaettchen_22-40mm      | 22-40mm       | Aachen Bauhaus | 1000          |
| Bruchsteine_Veronarot_30-60mm   | 30-60mm       | Aachen Bauhaus | 1000          |
| Quarzkies_Rund_Hell_16-32mm     | 16-32mm       | Aachen Bauhaus | 1000          |
| Quarzkies_Rund_Hell_8-16mm      | 8-16mm        | Aachen Bauhaus | 1000          |
| Kies_2-8mm                      | 2-8mm         | Aachen Bauhaus | 1000          |
| Kies_8-16mm                     | 8-16mm        | Aachen Bauhaus | 1000          |
| Kies_16-32mm                    | 16-32mm       | Aachen Bauhaus | 1000          |
| Basaltsplitt_8-12mm             | 8-12mm        | Aachen Bauhaus | 1000          |
| granitsplitt_8-12mm             | 8-12mm        | Aachen Bauhaus | 1000          |
| Splitt_2-5mm                    | 2-5mm         | Aachen Bauhaus | 1000          |
| Kies_1-3mm                      | 1-3mm         | Aachen OBI     | 1000          |

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









