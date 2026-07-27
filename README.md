# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project is a Machine Learning-based Credit Card Fraud Detection system that identifies whether a credit card transaction is legitimate or fraudulent.

The project uses the Logistic Regression algorithm to classify transactions based on the features present in the credit card dataset.

## Objective

The main objective of this project is to build a machine learning model that can detect fraudulent credit card transactions accurately.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Google Colab
- Logistic Regression

## Dataset

The dataset contains credit card transaction details.

The target column is `Class`:

- `0` → Legitimate Transaction
- `1` → Fraudulent Transaction

Since the original dataset is highly imbalanced, undersampling is performed to create a balanced dataset.

## Project Workflow

1. Import required Python libraries
2. Load the credit card dataset
3. Explore and understand the dataset
4. Check for missing values
5. Separate legitimate and fraudulent transactions
6. Analyze transaction amounts
7. Handle the imbalanced dataset using undersampling
8. Separate features and target
9. Split the dataset into training and testing data
10. Train a Logistic Regression model
11. Make predictions
12. Evaluate the model using accuracy score

## Handling Imbalanced Data

The dataset contains significantly more legitimate transactions than fraudulent transactions.

To balance the dataset, 492 legitimate transactions are randomly sampled and combined with the fraudulent transactions.

