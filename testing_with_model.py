# .............................................................................#
# Date : 27 DEC 2024, 04:29 MYT                                                #
# Author : Ansh sharma                                                          #
# Part II A: Class Prediciton Function using Trained AI/ML MODEL
#..............................................................................#

#..............................................................................#
"""
# Integrated:  Class Prediciton Function using Trained AI/ML MODEL
# Output: (i)   Prediction Class, 
#         (ii)  Confidence Score, 
"""
# .............................................................................#

# Importing required libraries for data processing, natural language processing, geolocation, and logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the trained model
def load_trained_model(model_path): 
    try:
        # Load the model from the specified path
        model = load_model(model_path)
        return model
    except Exception as e:
        raise ValueError(f"Error loading the model: {e}")

# Function to load class indices from a saved file
def load_class_indices(indices_path):
    try:
        with open(indices_path, 'r') as f:
            class_indices = json.load(f)
        return class_indices
    except Exception as e:
        raise ValueError(f"Error loading class indices: {e}")

#................PASSPORT AND INTERNAL PASSPORT PREDICTION FUNCTION............................................
# Predict on a new image
def predict_image_pass_int(img_path, model):
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1] range
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)
        print(f"Prediction confidence: {prediction[0][0]:.4f}")

        # Dynamically calculate a threshold based on the prediction confidence
        confidence_score = prediction[0][0]

       #............................................................................................
       # Using dynamic scaling factor based on the confidence score
        if confidence_score < 0.3:
            threshold = 0.7  # Stricter threshold for very low confidence
        elif confidence_score < 0.45:
            threshold = 0.55  # Higher threshold for low confidence cases
        elif confidence_score < 0.7:
            threshold = 0.6  # Moderate threshold for mid-range confidence
        else:
            threshold = 0.45  # Lenient threshold for high confidence

        print(f"Dynamic threshold: {threshold:.4f}")

        # Ensure the model output matches expected behavior
        class_label = "Passport" if confidence_score > threshold else "Internal Passport"
        return class_label, confidence_score
    except Exception as e:
        return f"Error processing the image: {e}"

#........................................ID AND PASSPORT PRECITION FUNCTION............................................
# Predict on a new image
def predict_image(img_path, model): #, class_indices):
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1] range
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)
        print(f"Prediction confidence: {prediction[0][0]:.4f}")

        # Dynamically calculate a threshold based on the prediction confidence
        confidence_score = prediction[0][0]

        
        #.................MAIN CODE......................................................................

        # Using dynamic scaling factor based on the confidence score
        if confidence_score < 0.3:
            threshold = 0.7  # Stricter threshold for very low confidence
        elif confidence_score < 0.45:
            threshold = 0.55  # Higher threshold for low confidence cases
        elif confidence_score < 0.7:
            threshold = 0.6  # Moderate threshold for mid-range confidence
        elif confidence_score <= 0.8:
            threshold = 0.45  # Lenient threshold for high confidence
        else:
            # Directly assign 'Passport' for very high confidence
            class_label = "Passport"
            print(f"Confidence score is very high ({confidence_score:.2f}). Class label set to Passport.")
            return class_label  # Exit early, no need to calculate further

        print(f"Dynamic threshold: {threshold:.4f}")

        # Assign class label based on adjusted thresholds
        if confidence_score > 0.8:
            class_label = "Passport"
        elif confidence_score > threshold:
            class_label = "Passport"
        else:
            class_label = "ID"

        print(f"Assigned class label: {class_label} based on confidence score {confidence_score:.2f} and threshold {threshold:.2f}.")
           
        return class_label, confidence_score
    except Exception as e:
        return f"Error processing the image: {e}"

# Main execution
if _name_ == "_main_":
    
    # Define the project path
    project_path = r'C:\Users\user\Downloads\solution_caseStudy' # Relative path user can change

    # Define the Train model file path

    model_path = os.path.join(project_path, 
                              "models",
                              "id_passport_classifier_new.h5") # Model to classify ID and Passport
     
    # model_path = r"C:\Users\user\Downloads\dataset\id_passport_classifier_new.h5" # final model at 18 nov 5:33
    # model_path = r"C:\Users\user\Downloads\dataset\id_passport_classifier_improved.h5"
   
    model_path_passint = os.path.join(project_path, 
                                      "models", 
                                      "passport_intPass_classifier.h5") # Model to classify Passport and internal Passport

    class_indices_path = os.path.join(project_path, 
                                      "models",
                                      "class_indices_2class.json") # Json Class file for ID, Passport
    class_indices_path1 = os.path.join(project_path, 
                                       "models",
                                       "class_indices_passint.json") # Json Class file for Passport, Internal passport
    
    test_image_path = os.path.join(project_path, 
                                   "test", 
                                   "27.jpg") # Can change test folder name and file name 

    #............................................................
    # Method if to run on multiple images to predict the classes
    #.............................................................

    # folder = r'C:\Users\user\Downloads\images\annotated_images'
    
    # for file in os.listdir(folder):
    #     # Replace with the actual image path
    #     # test_image_path = r"C:\Users\user\Downloads\images\photo\images\id\svk\14.jpg"
    #     test_image_path = os.path.join(folder, file)
    #     try:
    #         # Load the model
    #         model = load_trained_model(model_path)
            
    #         # Load class indices
    #         class_indices = load_class_indices(class_indices_path)

        
    #         # Predict the class of the test image
    #         # result = predict_image(test_image_path, model,  class_indices)
    #         # print(f"The image is classified as: {result}")
    #         print("*")

    #         predicted_class, confidence = predict_image(test_image_path, model, class_indices)
    #         print(f"Image {file} is classified as: {predicted_class} with Confidence: {confidence:.2f}")
    #     except ValueError as model_error:
    #         print(model_error)
    #     except Exception as e:
    #         print(f"An error occurred: {e}")

#.............................................................................................................
# Single Image function
#.............................................................................................................

    try:
        # Load the model
        model = load_trained_model(model_path)
        
        # Load class indices
        # class_indices = load_class_indices(class_indices_path)
        

        # Predict function to determine the class for ID, Passport
        predicted_class, confidence = predict_image(test_image_path, 
                                                    model) # , class_indices) 
        
        # Predict function to determine the class for Passport, Internal Passport
        # predicted_class, confidence = predict_image_pass_int(test_image_path, model) # , class_indices)

        print(f"Image {test_image_path} is classified as: {predicted_class} with Confidence: {confidence:.2f}")
    except ValueError as model_error:
        print(model_error)
    except Exception as e:
        print(f"An error occurred: {e}")    
# this will not give the json file need to check and Do in a proper way to tackel this 
