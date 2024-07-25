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

## Summary of Two Approaches to Obtain Grain Size

In this project, we explored two distinct approaches to obtain the grain size of granular materials using advanced computer vision techniques. These approaches include the use of regression models and segmentation methods. Below is a detailed summary of each approach.

#### 1. Regression-Based Approach

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

#### 2. Segmentation-Based Approach

##### Methodology
- **Model Architecture**: The Segment Anything Model (SAM) was utilized for initial image segmentation. SAM is capable of identifying and segmenting individual grains within an image.
- **Data Preparation**: Images were processed to ensure they were suitable for segmentation. This included resizing, normalization, and augmentation to create diverse training data.
- **Segmentation and Measurement**: SAM was applied to segment the grains in the images. Post-segmentation, OpenCV techniques were employed to measure the size of each segmented grain. The measurements were then averaged to obtain an overall grain size for the image.
- **Evaluation**: The segmentation accuracy and the grain size measurements were compared to ground truth values to evaluate performance.

##### Results
- The segmentation approach provided detailed information about the distribution and size of individual grains within an image.
- This method allowed for the visualization of segmented grains, aiding in the qualitative assessment of the segmentation accuracy.

##### Advantages
- **Detail-Oriented**: Provides detailed insights into the size and distribution of individual grains.
- **Flexibility**: Can handle images with varying grain sizes and distributions more effectively.

##### Disadvantages
- **Complexity**: The segmentation and measurement process is more complex and computationally intensive compared to direct regression.
- **Processing Time**: Segmenting each grain and measuring its size takes more time, especially for high-resolution images with many grains.

#### Comparative Analysis

- **Accuracy**: Both methods demonstrated good accuracy, with the regression model being faster and simpler, while the segmentation approach provided more detailed information.
- **Scalability**: The regression model is more scalable for large datasets due to its efficiency in processing, whereas the segmentation approach, while more informative, requires significant computational resources.
- **Application Suitability**: The choice of method depends on the specific requirements of the application. For quick and continuous predictions, the regression model is suitable. For detailed analysis and visualization, the segmentation approach is preferable.

### Conclusion

By exploring these two approaches, the project highlights the potential and versatility of computer vision techniques in analyzing granular materials. Both methods have their unique strengths and can be applied based on the specific needs of the analysis, paving the way for further research and development in this field.


### Introduction
Granular materials, consisting of a collection of discrete solid particles, play a crucial role in various industries such as construction, pharmaceuticals, agriculture, and food processing. Accurate examination and classification of these materials are essential for optimizing processes, improving product quality, and ensuring safety standards. Traditional methods of granular material analysis, including sieving and manual inspection, are often time-consuming, labor-intensive, and subject to human error. Recent advancements in computer vision and artificial intelligence (AI) offer promising solutions to these challenges, enabling more efficient and precise analysis.

### Background Information
Granular material examination typically involves determining particle size distribution, shape, and other morphological characteristics. Traditional techniques like sieve analysis, although widely used, have limitations in terms of resolution and efficiency. In recent years, computer vision has emerged as a powerful tool for material analysis, leveraging advanced image processing algorithms to extract detailed information from digital images.
The Segment Anything Model (SAM) is a state-of-the-art segmentation model that has shown remarkable performance in various image segmentation tasks. When applied to granular materials, SAM can accurately delineate individual particles, even in complex and overlapping scenarios. This segmentation capability is essential for precise feature extraction, which forms the basis for subsequent classification and analysis.
OpenCV, an open-source computer vision library, provides a comprehensive set of tools for feature extraction, including shape descriptors, texture analysis, and color-based segmentation. By integrating SAM with OpenCV, it is possible to develop a robust system for extracting detailed features from segmented particles, which can then be used to train machine learning models for classification and prediction.
Artificial Neural Networks (ANNs), a class of machine learning models inspired by the human brain, have demonstrated exceptional performance in various predictive tasks. ANNs can be trained to recognize patterns and relationships within the extracted features, enabling accurate prediction of particle size distribution and other material properties.

### Previous Research
Several studies have explored the use of computer vision and AI in granular material analysis. For instance, a study by Buitrón Cevallos et al. (2023) utilized AI for the characterization of granular media, demonstrating significant improvements in accuracy and efficiency compared to traditional methods (source: https://ngi.brage.unit.no/ngi-xmlui/bitstream/handle/11250/3069797/1-s2.0-S0266352X23002677-main.pdf?sequence=1). Another study by Hejft et al. (2017) applied computer image analysis to assess grain size composition, highlighting the potential of computer vision techniques in enhancing material analysis (source: https://bibliotekanauki.pl/articles/334621.pdf).
In another notable study, Bastidas-Rodríguez et al. (2016) employed computer vision techniques for the fractographic classification of metallic materials, demonstrating the applicability of these methods across different types of granular and particulate materials (source: http://www.scielo.org.co/scielo.php?pid=S0121-11292016000300007&script=sci_arttext). These studies underscore the growing recognition of computer vision and AI as valuable tools in material science.

### Research Justification
The primary motivation for this research is to address the limitations of traditional granular material examination methods by leveraging advanced computer vision and AI techniques. The proposed system aims to enhance accuracy, reduce inspection time, and provide a user-friendly platform for real-time analysis. This research is particularly relevant in the context of increasing demands for high-quality materials and the need for efficient and reliable analysis methods.
### Hypothesis
The hypothesis of this research is that integrating the Segment Anything Model (SAM), OpenCV for feature extraction, and Artificial Neural Networks (ANNs) can significantly improve the accuracy and efficiency of granular material examination and classification compared to traditional methods.

### Techniques and Objectives
The proposed system will utilize the following techniques:
1. **Segment Anything Model (SAM)**: For accurate particle segmentation in complex images.
2. **OpenCV**: For detailed feature extraction from segmented particles.
3. **Artificial Neural Networks (ANNs)**: For predicting particle size distribution based on extracted features.
The objectives of this research are:
1. To develop a comprehensive system for examining and classifying granular materials using computer vision.
2. To validate the system with both synthetic and real-world samples.
3. To compare the performance of the proposed system against traditional methods.
4. To integrate the system into a user-friendly application for real-time analysis.


### Importance in a Broader Context
This research contributes to the broader field of material science by providing a novel approach to granular material analysis, which can be applied across various industries. The integration of computer vision and AI techniques not only enhances accuracy and efficiency but also paves the way for future advancements in automated material analysis. By reducing dependency on manual inspection and traditional methods, the proposed system aligns with the ongoing trend towards automation and digitalization in industry.


### Contribution to Existing Work
This research builds upon existing studies in computer vision and AI, offering an integrated approach that combines advanced segmentation, feature extraction, and predictive modeling. The proposed system distinguishes itself by leveraging state-of-the-art techniques and providing a comprehensive solution for granular material examination and classification. It aims to set a new benchmark in the field, offering insights and tools that can be adapted and expanded for various applications.



