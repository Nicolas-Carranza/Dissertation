# Deep Learning for Automatic Seal Counting in Aerial Images

**Author:** Sophie Bickerton (170001861)  
**Supervisors:** Dr Kasim Terzic and Prof Len Thomas  
**Date:** April 2021

## Abstract

One of the focuses of the Sea Mammal Research Unit (SMRU) is in monitoring the populations of seals in the U.K.. The aim of this SH project is to research object detection and image classification techniques in order to apply this to the application of aerial images. In order to count the different types of seals, two stages of machine learning are connected together in order to perform the classification successfully. The first stage is using YOLOv3 to detect seals against the background of large aerial images. The detected seals are then cut out of the large aerial images into smaller tiles. These tiles are converted to vectors using a histogram of oriented gradients. The results of this are put through a second stage classification to detect what type of seal it is.

## Declaration

I declare that the material submitted for assessment is my own work except where credit is explicitly given to others by citation or acknowledgement. This work was performed during the current academic year except where otherwise stated. The main text of this project report is 11,169 words long. In submitting this project report to the University of St Andrews, I give permission for it to be made available for use in accordance with the regulations of the University Library. I also give permission for the title and abstract to be published and for copies of the report to be made and supplied at cost to any bona fide library or research worker, and to be made available on the World Wide Web. I retain the copyright in this work.

## Acknowledgements

I would like to thank my supervisors, Prof. Len Thomas and Dr. Kasim Terzic for their continued support and advice throughout this turbulent year. I appreciate their guidance and insights in this field and the results of this project were greatly impacted by our discussions throughout the course of the year.

---

## Contents

1. [Introduction](#introduction)
2. [Context Survey](#context-survey)
3. [Requirements Specification](#requirements-specification)
4. [Software Engineering Process](#software-engineering-process)
5. [Ethics](#ethics)
6. [Design](#design)
7. [Implementation](#implementation)
8. [Evaluation and Critical Appraisal](#evaluation-and-critical-appraisal)

---

## Introduction

The Seal Mammal Research Unit (SMRU) uses aerial images to count seals and classify their stage of development (pups, whitecoats, moulted, etc.) in order to ultimately estimate the population of seal in Scotland. Currently, around 40% of all grey seals live in UK waters so estimating this population is significant. This task is mainly completed by hand and takes a significant amount of time to complete each year.

To replace this process, the implementation will be to detect and classify seals from the provided aerial images. Steps towards this goal have been made previously by two students from St. Andrews, Dorottya Denes and Samuel Pavlik, whose work has been built upon in this project.

### Disruption Due to the COVID-19 Pandemic

This entirety of this project was conducted during the pandemic and as a consequence planning was done with the consideration that all work would have to be completed remotely.

---

## Context Survey

Machine learning is a branch of Artificial Intelligence (AI) which applies statistics to algorithms in order to find patterns within data. This project uses machine learning techniques in order to detect and classify seals.

### Standard Machine Learning Algorithms

#### Support Vector Machine (SVM)

The concept of the Support Vector Machine is to find a hyperplane which divides the different classes. With 2 input features (e.g., height and width) the hyperplane is a line which separates the classes, the more features mean the hyperplane is in more dimensions.

SVM uses the points closest to the hyperplanes, called support vectors, in order to calculate the optimal hyperplane. The result from maximizing the margin between the support vectors gives us an optimal separation.

The loss function for the SVM is called hinge loss:

$$L(x, y, f(x)) = \begin{cases}
0 & \text{if } y \cdot f(x) \geq 1 \\
1 - y \cdot f(x) & \text{otherwise}
\end{cases}$$

#### Random Forest

The concept of the random forest algorithm is that there are several decision trees which are produced from random samples of the dataset. The sampling technique is bootstrapping where there may be repeated entries in the training sample, this is also known as sampling with replacement. Each decision tree is trained on these samples and with an input predicts a class as an output. This is called a 'vote', a majority of the votes decides the class that is predicted by the algorithm.

When the decision trees are trained, they are done so without pruning, this means that the depth and overall size of the tree is not restricted.

### Deep Learning Overview

Deep learning is a subsection of machine learning which focuses on algorithms and techniques that focus on artificial neural networks. These techniques attempt to imitate the way the human brain processes data, spots patterns and trends.

#### Neural Network Architecture

A neural network can be formed by connecting many perceptrons together. The basic unit is the perceptron, made up of an input function, activation function and relevant weighting.

**Input Layer**

Each of the inputs in the input layer have weighting applied to them at the input function:

$$z = \sum_{i=1}^{n} w_i \cdot x_i + b$$

**Hidden Layer**

The activation function determines how the input is transformed into the output for each node in the network. Popular options include:

- **Sigmoid:** $\text{sigmoid} = \frac{1}{1 + e^{-z}}$
- **Tanh:** $\text{tanh} = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
- **ReLU:** $\text{ReLU} = \max(0, z)$
- **Softmax:** $\text{Softmax} = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}$

#### Universality Theorem

The universality theorem states: "no matter what function we want to compute, we know that there is a neural network which can do the job". The neural network, however, will only be able to approximate the function and the structure of this network will determine how good the approximation is.

#### Forward and Back Propagation

**Forward-Propagation**

The structure of the neural network has been explained, with an example of a feed-forward network. To understand forward-propagation:

For each hidden neuron in the first layer:
$$a_i^{(1)} = f(w_{i,1}^{(1)} x_1 + w_{i,2}^{(1)} x_2 + ... + b_0^{(1)})$$

where f is the activation function and b_0 is the bias for that layer.

**Back-propagation**

Back-propagation was introduced to enable neural networks to learn internally by adjusting weights and biases. Using back-propagation requires an error (also known as cost) function. This function will give a value to represent the error between the provided output and the expected output. This is typically Mean Squared Error (MSE) for linear regression or Cross-Entropy for classification tasks.

**Gradient Descent**

Gradient descent works from a given initial value and steps towards the minimum. At each step the function calculates the gradient and moves in the negative direction. The step size is determined by a value called the learning rate.

**Vanishing Gradient Problem**

When using an activation function such as sigmoid in multiple nodes in multiple layers, it can cause problems in back-propagation. The vanishing gradient problem occurs as the derivative of the sigmoid function is very small. If there are several layers that use the sigmoid as the activation function, then it results in the multiplication of increasingly small values.

**Dying ReLU Problem**

The ReLU activation function produces 0 as the result for input values that are less than or equal to 0. This also subsequently means that the derivative is also 0 for these values. Therefore, weights will not be altered, and this results in the model not being fully optimized.

### Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are a subsection of Artificial Neural Networks that are ideal for computer vision-based tasks. The name convolutional comes from the use of the mathematical process involving matrices called convolution.

#### Architecture

CNNs are made up of three key layers: a Convolutional layer, a Pooling layer and a Fully-Connected layer.

**Convolutional Layer**

An image is made up of pixels and these pixels have values corresponding to colour channels (RGB, Greyscale, etc.). The concept of the convolutional layer is to pass a filter (also known as a kernel) over each of the layers to produce an altered set of pixels as an output.

**Pooling Layer**

Using multiple filters in the convolutional layer can mean the output has a large depth and this can end in a large amount of data that needs to be processed. Pooling the data will substantially reduce what needs to be processed at the next layer.

**Fully-Connected Layer**

The fully-connected layer is most simply described as a feed-forward neural network. This layer sits at the end of the architecture and is used to produce the predicted output.

### Object Detection

Object detection consists of identifying the location of the object, with a bounding box, and then providing a label to said object. This can be simply described as a localization step followed by a classification step.

#### Performance Metrics

**Intersection over Union (IoU)**

When an object is detected, there is a bounding box supplied with it which signifies the location of the detected object. To determine whether or not a proposed bounding box is correct a metric called Intersection over Union (IoU) is applied:

$$\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$

**Precision and Recall**

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**Average Precision (AP) and Mean Average Precision (mAP)**

Average Precision (AP) summarizes the recall and precision and is calculated as:

$$\text{AP} = \sum_{i=0}^{n-1} [\text{Recall}(i) - \text{Recall}(i+1)] \cdot \text{Precision}(i)$$

The mean average precision (mAP) is calculated as the mean of the APs of each class:

$$\text{mAP} = \frac{1}{N} \sum_{k=1}^{N} \text{AP}(k)$$

#### YOLO

You only look once (YOLO) is a specialised CNN which is designed to improve object detection and classification. YOLOv3 is specifically relevant as it is used in the application and is the third iteration of the algorithm.

### Summary of Prior Work

Steps towards this goal have been made previously by Dorottya Denes and Samuel Pavlik. Their work used YOLO to detect and classify seals. Whilst the detection of a seal against the background landscape was very successful, when it came to classifying Whitecoats, Moulted and Dead Pups, the results were not as good as hoped. By implementing a two-stage algorithm, improved classification results are hoped for.

---

## Requirements Specification

### Primary Objectives

- Research a variety of deep learning techniques on object detection and classification
- Prepare the dataset of provided seal images
- Implement a binary classifier that will detect seals
- Train a classifier which will use the detected seals to recognize what type of seal has been detected

### Secondary Objectives

- Calculate confidence values for the system
- Experiment with tiling approach
- Experiment with weighting of the classifier
- Integrate results into the GIS system used by SMRU

### Primary Requirements

- Research object detection Convolutional Neural Network (CNN) based algorithms including YOLO and ResNet
- Research machine learning algorithms ideal for classification such as SVMs and Random Forest
- Cropping of larger TIFF files to small images
- Use binary classification for object detection
- Transform results into cropped seal images using HOG
- Use multiclass classification to detect seal type
- Evaluate results using precision-recall curve, ROC curve and confusion matrix

### Secondary Requirements

- Research alternate object detection methods
- Research and compare several transformation systems beyond HOG
- Use image augmentation
- Compare CNN classifier against traditional methods
- Calculate confidence intervals

### Tertiary Requirements

- Connect stages together for whole system training
- Integrate system into GIS

---

## Software Engineering Process

### Development Process

Throughout the development of the project, weekly meetings with supervisors were held in order to check in and keep an agile based approach. This enabled target setting each week for small, manageable goals to be completed. The agile process over the waterfall model was much preferred as it allowed small targets with balance of working on this project and any others.

### Resources and Technologies

**Python**

This was the language used throughout the project. All the existing code was in this language, so it was much easier to extend with the same language. Python is also ideal for machine learning projects due to the expansive libraries.

**Jupyter Lab**

This platform was used when developing the project. It allowed for a range of files to be used from interactive jupyter notebook to standard python and text files.

**TensorFlow**

This is a library which was used for the first stage of the project due to it being good at pipelining and highly parallel.

**External Server**

The project was conducted using a server hosted by one of supervisors which enabled the training to occur in a shorter duration.

---

## Ethics

There were no ethical considerations for this project as the aerial images were supplied by the SMRU and permission was given for them to be used.

---

## Design

The first stage is to use object detection to identify seals against the background. Aiming for a high recall on this first stage, the second stage is to take these identified images and classify them with a higher precision.

### Data Preparation

#### Image Cropping

The aerial images provided by the SMRU are far too large to enable an efficient training procedure. Therefore, the images need to be cropped into smaller tiles.

#### Tile Batches

Three main batches were implemented to look at how tiles from different islands impacted the result of the process as a whole:

1. Firth of Forth (FirFor) tiles
2. Orkney (Ork) tiles
3. Stratified sampling from all islands

#### Anchors

As a part of the YOLO parameters, anchors are required, which give the algorithm a sense of the size of bounding boxes that are expected to be predicted.

### Object Detection

The first stage of the model is to detect seals in the tiles which make up the training set. The aim is to provide as high a recall as possible so that the second stage will have all the possible seal images.

### Transformation Stage

This stage will take the bounding boxes of the predicted seals and transform this data such that it can be used to train the image classifier.

#### Prediction Cropping

The predicted bounding boxes are transformed into a smaller tile around the seal location.

#### Extracting HOG

The second step of the transformation process is to extract the HOG from the smaller tiles. This can then be used as an input to the image classifier.

### Image Classification

After transforming the smaller seal tiles, the aim of this stage is to classify the type of seal as well as any background tiles.

#### Support Vector Machine

The first method that was developed was to use a Support Vector Machine which has previously been used alongside HOG to produce successful results.

#### Random Forest

The second method to develop was to use a Random Forest based script in order to produce an image classifier.

#### Neural Networks

Another option to consider was using a multi-layer perceptron (MLP) classifier or ResNet based classifiers.

---

## Implementation

This project is implemented in Python and makes use of the TensorFlow framework where applicable.

### Data Preparation

#### Image Cropping

The image cropping scripts is called ImageCrop and was simply adjusted in order to cut the tiles required for training.

#### Tile Batches

Three main batches were implemented with different islands and data sources.

#### Anchors

Implementation of this section was based on code previously created. The calculation of the anchors works by using the size of the bounding boxes in the training dataset to calculate what the size of future bounding boxes looks like.

### Object Detection

Similar to the Data Preparation section, the work on object detection was building onto the pre-existing code to ensure it has a high recall.

#### Training

The key part of training algorithms to produce a high recall and reasonable precision is to alter the hyperparameters. The main hyperparameter to change was score_threshold as this adjusts the minimum probability the bounding boxes have to be to a positive detection.

#### Evaluation

Results at different thresholds:

| Threshold | Precision | Recall | AP    |
|-----------|-----------|--------|-------|
| 0.01      | 0.0143    | 0.9960 | 0.9141|
| 0.1       | 0.2836    | 0.9681 | 0.9100|
| 0.2       | 0.5760    | 0.9402 | 0.8972|
| 0.3       | 0.7627    | 0.8964 | 0.8672|

### Transformation Stage

#### Prediction Cropping

Cropping is linked to the eval.py script as the chosen model produces predicted bounding boxes. Tile sizes were tested from 60x60 up to 80x80.

#### Extracting HOG

The script extract.py produces data from HOG and RGB channels for each tile. Comparing using RGB showed that the colour channels impacted the classification.

### Image Classification

#### Support Vector Machine

Using Bayesian Optimization to tune hyperparameters:

| Class       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| Whitecoats  | 0.53      | 0.32   | 0.40     |
| Moulted     | 0.25      | 0.31   | 0.28     |
| Dead Pups   | 0.02      | 0.46   | 0.30     |

#### Random Forest

Using GridSearchCV to tune hyperparameters:

| Class       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| Whitecoats  | 0.53      | 0.49   | 0.51     |
| Moulted     | 0.00      | 0.00   | 0.00     |
| Dead Pups   | 0.00      | 0.00   | 0.00     |

#### Neural Networks

Using MLP with 'adam' solver:

| Class       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| Whitecoats  | 0.46      | 0.63   | 0.53     |
| Moulted     | 0.26      | 0.10   | 0.14     |
| Dead Pups   | 0.06      | 0.02   | 0.03     |

---

## Evaluation and Critical Appraisal

Overall, this project asked for an understanding of machine learning, deep learning as well as how to apply it to research code. Working with previous code was more of a challenge than initially anticipated.

The main surprise when exploring this implementation has been how unsuccessful hyperparameter tuning has been. When completing vast searches, the results often end with only the whitecoats class giving good results and the other classes being overlooked.

With more time, work could focus on the second stage of the process and working on different algorithms to improve performance. This can be achieved by:

1. Research into an alternative to the HOG method
2. Work on the image classification algorithms to fine tune it
3. Recognition that this is a difficult problem due to seals being the same colour and shape as rocks

---

## References

[Ahmadzadeh et al., 2017] Ahmadzadeh, A., et al. (2017). Improving the functionality of tamura directionality on solar images. In 2017 IEEE International Conference on Big Data.

[Albawi et al., 2017] Albawi, S., Mohammed, T. A., and Al-Zawi, S. (2017). Understanding of a convolutional neural network. In 2017 International Conference on Engineering and Technology.

[Caie et al., 2021] Caie, P. D., Dimitriou, N., and Arandjelović, O. (2021). Precision medicine in digital pathology via image analysis and machine learning.

[Denes, 2020] Denes, D. (2020). Neural Networks and Deep Learning. PhD thesis, University of St. Andrews.

(Additional references continue as in the original document)
