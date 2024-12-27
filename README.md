# Computer-Vision-CV-model

A Computer Vision (CV) system powered by a Convolutional Neural Network (CNN). It processes images by first preprocessing them—resizing to consistent dimensions, normalizing pixel values, and augmenting the data for better training. Using MobileNetV2, a pretrained CNN backbone, the model extracts hierarchical features like edges, textures, and shapes, which are crucial for distinguishing between classes. The extracted features are passed through a global average pooling layer to create a compact representation, followed by a dense layer with a sigmoid activation function that outputs a probability score between 0 and 1.

This project implements a Computer Vision (CV) system for image classification, powered by a Convolutional Neural Network (CNN). The model is designed to distinguish between two classes: IDs and Passports. The system includes data preprocessing, model training, evaluation, and testing functionalities.

# Features

# Data Preprocessing:

Resizing images to consistent dimensions (224x224).

Normalizing pixel values to a range of [0, 1].

Data augmentation (shearing, zooming, and horizontal flipping) to enhance training.

# Model Architecture:

Base Model: MobileNetV2, a pre-trained CNN backbone, extracts hierarchical features such as edges, textures, and shapes.

Custom Layers:

Global Average Pooling layer to create a compact feature representation.

Dense layer with a sigmoid activation function to output a probability score between 0 and 1 for binary classification.

# File Formats:

Model: Saved as .h5 format, which includes the entire model (architecture, weights, and optimizer state).

Model Architecture: Saved separately as .json format for modularity and reuse.

Testing:

Provided code to test the model on new images.

# Workflow

1. Data Preprocessing

The dataset is loaded using TensorFlow's ImageDataGenerator for real-time augmentation.

Images are split into training (80%) and validation (20%) subsets.

Images are resized to 224x224 pixels and normalized for consistency.

2. Model Training

MobileNetV2 is used as the backbone, with its pre-trained weights frozen initially.

The custom layers are added, and the model is compiled with:

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

The model is trained for 10 epochs with an initial learning rate of 0.001.

3. Fine-Tuning

Some layers of the pre-trained model are unfrozen for fine-tuning.

The learning rate is reduced to 1e-5, and the model is trained for an additional 5 epochs.

4. Saving the Model

The trained model is saved in two formats:

.h5: Complete model (architecture + weights).

.json: Model architecture only.

5. Testing

Testing functionality allows you to input new images and predict whether they belong to the "ID" or "Passport" class.

Example testing code is included in the repository.

How the Model Works

# Feature Extraction:

MobileNetV2 extracts features like edges, textures, and shapes from input images.

Feature Pooling:

A global average pooling layer reduces the feature maps to a compact representation.

Classification:

The dense layer outputs a probability score for each class.

Checking for Overfitting and Underfitting

Overfitting Indicators:

High accuracy on training data but low accuracy on validation data.

Large gap between training and validation loss.

# Underfitting Indicators:

Both training and validation accuracies are low.

Training and validation losses remain high.

Best Practices to Address Overfitting:

Use dropout layers to reduce overfitting.

Employ data augmentation for training.

Use early stopping to halt training when validation performance stops improving.

Best Practices to Address Underfitting:

Increase model capacity (e.g., add more layers or units).

Train for more epochs.

Ensure the learning rate is not too low.
