# Plant Disease Classification Using Deep Learning

This repository contains a deep learning project aimed at classifying plant diseases using Convolutional Neural Networks (CNNs). The model is trained on the "New Plant Diseases Dataset" from Kaggle and includes several categories of plant diseases as well as healthy plants.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training and Evaluation](#training-and-evaluation)
5. [Results](#results)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Acknowledgments](#acknowledgments)

## Project Overview

This project leverages CNNs to build a robust model capable of identifying different plant diseases from images. The model was trained on a large dataset and utilizes transfer learning with the MobileNetV2, VGG16, DenseNet and ResNet50 architecture, coupled with data augmentation techniques to improve generalization.

## Dataset

The dataset used for training and evaluation is the **New Plant Diseases Dataset** from Kaggle. It contains 70,295 images across multiple categories of plant diseases, including healthy plant samples. The dataset is split into training, validation, and test sets for model development and evaluation.

### Key Statistics:
- **Number of Classes:** 38
- **Total Images:** 70,295
- **Dataset Source:** [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/emmarex/plantdisease)

## Model Architecture

The CNN model was designed to handle high-dimensional image data, with the following key components:

- **Convolutional Layers:** For feature extraction from input images.
- **Batch Normalization & ReLU/LeakyReLU Activations:** To stabilize learning and accelerate convergence.
- **Pooling Layers:** To reduce dimensionality while preserving key features.
- **Dropout:** Applied to prevent overfitting.
- **Global Average Pooling:** Followed by a Dense layer with Softmax activation for multi-class classification.

## Training and Evaluation

The model was trained with the following settings:

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint for efficient training.

### Evaluation Metrics:
- Accuracy
- Loss
- Confusion Matrix
- Precision, Recall, F1-Score

## Results

The model achieved strong performance on the test set. The confusion matrix and classification report demonstrate the model's effectiveness in identifying different plant diseases.

### Key Performance Highlights:
- **Validation Accuracy:** 97%
- **Validation Loss:** 0.23
- **Test Accuracy:** 93%
- **Confusion Matrix & Classification Report:** Generated to analyze performance across classes.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/plant-disease-classification.git
   ```

2. Install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the dataset in the specified directory (refer to the `dataset/` structure in the project).

## Usage

Once you have installed the dependencies and set up the dataset, you can run the following steps to train and evaluate the model:

1. **Training the Model:**

   The training process uses the dataset from the `train/` and `valid/` directories. The model checkpoints and training history will be saved after every few epochs.

2. **Evaluation on Test Data:**

   After training, the model is evaluated on the test set. This produces accuracy metrics, confusion matrix, and classification report, which are saved for later inspection.

3. **Saving and Loading the Model:**

   The trained model is saved to a `.keras` file and can be loaded for inference or fine-tuning in the future.

## Acknowledgments

Special thanks to the Kaggle community for providing the dataset and to all contributors who helped in the development of this project.
