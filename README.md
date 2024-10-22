# Breast Cancer Classification using Machine Learning
This project focuses on classifying breast cancer tumors as malignant or benign using Logistic Regression, leveraging the Breast Cancer Wisconsin Dataset from the sklearn library.

# Project Overview
Breast cancer is one of the most common cancers in women, and early detection plays a crucial role in treatment. In this project, I built a machine learning model using Logistic Regression to predict whether a breast cancer tumor is malignant or benign based on features from biopsy results.

# Dataset
**Dataset Source**: The Breast Cancer Wisconsin Dataset (Diagnostic), available in the sklearn.datasets library.
**Features**: 30 numeric features describing characteristics of the cell nuclei present in the image.
**Target**: Binary classification (0 = Malignant, 1 = Benign).

# Technologies Used
Python: Main programming language

# Libraries:
* NumPy and Pandas for data manipulation
* scikit-learn for model building and evaluation
  
# Project Workflow
**Data Loading**: Loaded the Breast Cancer Wisconsin dataset using sklearn.datasets.
**Data Preprocessing**: Converted the dataset into a DataFrame and performed basic exploratory data analysis (EDA).
**Data Splitting**: Split the dataset into training and testing sets (80% training, 20% testing).
**Model Building**: Trained a Logistic Regression model using the training data.
**Model Evaluation**:
  * Training Accuracy: Measured using the accuracy_score function.
  * Testing Accuracy: Measured to evaluate model performance on unseen data.

# Results
The Logistic Regression model achieved good accuracy on both the training and testing datasets. You can use this project as a starting point for further exploration of different algorithms or feature engineering techniques.

# Future Work
* Experiment with other machine learning models like Random Forest or Support Vector Machine (SVM).
* Implement precision, recall, and F1-score as additional evaluation metrics.
* Perform hyperparameter tuning to optimize model performance.
