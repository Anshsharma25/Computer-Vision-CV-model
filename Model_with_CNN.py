# ..............................................................................#
# Date : 27 DEC 2024, 06:29 MYT                                                 #
# Author : Ansh sharma                                                           #
# PART II: "AI/ML Model Selection and Classification of Classes (ID, Passport)" #
#...............................................................................#

#...............................................................................#
"""
# Integrated: "AI/ML Model Selection and Classification of Classes (ID, Passport)"
# Output: (i)   Class Classification (ID, Passport), 
#         (ii)  AI/ML Model (.h5), 
#         (iii) Model Accuracy score.
"""

# Importing required libraries for data processing, natural language processing, geolocation, and logging

import os
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

import logging


#..............................INITIALIZATION.................................................................

# ---------------------------
# CONFIG USER DEFINED PATH
     
# ---------------------------
        
# Get the current date for validation purposes
current_date = datetime.now()
print("Initialized current date: %s", current_date)

# Define the project path where files will be stored
project_path = r"C:\Users\user\Downloads\solution_caseStudy"

# Format the current date to a string (e.g., '2024-11-20_15-45-30')
formatted_date = current_date.strftime("%Y-%m-%d_%H-%M-%S")


# Output Folder to Store Resultant files 
log_folder = os.path.join(project_path,'logs')
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
    logging.info(f"Folder '{log_folder}' created successfully.")
else:
    logging.info(f"Folder '{log_folder}' already exists.")
#...................................................................

# Create the log filename by appending the formatted date to the base name
log_filename = "Class_Classification_" + formatted_date + ".log"
# Create the log file path by combining the project path and log filename
log_path = os.path.join(project_path, log_folder, log_filename)

#.............................................................................................................

# Set up the logger
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

#...............................................................................................................

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

try:
    # Step 1: Data Preparation
    # Specify the path to the dataset directory
    data_dir = r'C:\Users\user\Downloads\datasets' # Replace with your dataset path

    # Data Augmentation and Preprocessing setup
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,      # Rescaling the image pixel values to [0, 1]
        validation_split=0.2,   # Reserve 20% of data for validation
        shear_range=0.2,        # Shear angle for data augmentation
        zoom_range=0.2,         # Random zoom for data augmentation
        horizontal_flip=True    # Randomly flip images horizontally
    )
    logger.info("Data preprocessing and augmentation setup completed.")    

    # Load training data using flow_from_directory
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),  # Resize all images to 224x224
        batch_size=32,
        class_mode='binary',     # Binary classification: ID vs Passport
        subset='training'        # Use subset 'training' for training data
    )

    # Load validation data using flow_from_directory
    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),   # Resize all images to 224x224
        batch_size=32,
        class_mode='binary',
        subset='validation'       # Use subset 'validation' for validation data
    )

    # Step 2: Model Building
    # Load MobileNetV2 pre-trained model without the top layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze pre-trained layers to avoid retraining

    # Transfer Learning: Build the custom model with additional layers for binary classification
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),  # Reduce overfitting
        Dense(1, activation='sigmoid')  # Sigmoid activation for binary output
    ])

    # Compile the model with Adam optimizer and binary crossentropy loss
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    logger.info("Model architecture and compilation completed.")


    # Step 3: Train the Model
    history = model.fit(
        train_data,
        epochs=10,  # Adjust as needed
        validation_data=val_data
    )
    logger.info("Model training completed.")

    # Step 4: Fine-tuning
    # Unfreeze some layers of the pre-trained model for fine-tuning
    base_model.trainable = True

    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Fine-tune the model for a few epochs
    history_fine_tune = model.fit(
        train_data,
        epochs=5,  # Fine-tune for a few epochs
        validation_data=val_data
    )

    logger.info("Fine-tuning of the model completed.")
    # Step 5: Save the Model
    model_save_path = os.path.join(data_dir, "id_passports_classifier.h5")
    model.save (model_save_path)
    logger.info(f"Model saved successfully at {model_save_path}")

    # Step 6: Evaluate the Model
    test_loss, test_acc = model.evaluate(val_data)
    logger.info(f"Test Accuracy: {test_acc * 100:.2f}%")

except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
