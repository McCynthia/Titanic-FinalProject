# Titanic-FinalProject

## Overview
This project involves building machine learning models to predict the survival of passengers on the Titanic using various algorithms, including Random Forest, K-Nearest Neighbors (KNN), Logistic Regression, and Support Vector Machine (SVM). The models are trained on a preprocessed dataset, and their performances are evaluated using accuracy and classification reports.

## Prerequisites
Ensure you have the following Python packages installed:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn

You can install the required packages using pip: `pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn`

## Dataset
The datasets used in this project are ***train.csv*** and ***test.csv***. Ensure these files are present in the same directory as the script.

## Script Overview
The script performs the following steps:

1. Data Loading:
   
Loads the training and test datasets from CSV files.

2. Data Preprocessing:

Fills missing values.
Encodes categorical variables into numerical values.
Performs feature engineering (creating new features from existing ones).

3. Data Normalization:

Normalizes numerical features using StandardScaler.

4. Feature Selection:

Splits the data into features (X) and target (y).

5. Model Training and Evaluation:

- Splits the data into training and validation sets.
- Trains and evaluates the following models using GridSearchCV for hyperparameter tuning:
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Support Vector Machine (SVM)
- Prints the accuracy and classification report for each model.
- Visualizes the confusion matrices and accuracy comparison of the models.

6. Predictions:

- Makes predictions on the test dataset using the trained models.
- Saves the predictions to predictions.csv.

## How to Run
Make sure the train.csv and test.csv are in the same directory as the script.

Run the script using Python: `python titanic-ml.py`

## Output
Model Accuracy and Classification Reports: 
- Printed to the console.
- Confusion Matrix Visualizations: Saved as PNG files (rf_confusion_matrix.png, knn_confusion_matrix.png, lr_confusion_matrix.png, svm_confusion_matrix.png).
- Accuracy Comparison Plot: Saved as accuracy_comparison.png.
- Predictions CSV: The script saves the predictions of each model to predictions.csv.

