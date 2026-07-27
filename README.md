# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project is a Machine Learning-based Credit Card Fraud Detection system that identifies whether a credit card transaction is legitimate or fraudulent.

The project uses the **Logistic Regression** algorithm to classify credit card transactions.

---

## Objective

The main objective of this project is to build a Machine Learning model that can detect fraudulent credit card transactions and distinguish them from legitimate transactions.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Google Colab
- Logistic Regression

---

## Dataset

The dataset contains credit card transaction details.

The target column is **Class**:

- `0` → Legitimate Transaction
- `1` → Fraudulent Transaction

The original dataset is highly imbalanced because the number of legitimate transactions is much higher than the number of fraudulent transactions.

---

## Project Workflow

1. Import the required Python libraries
2. Load the credit card dataset
3. Explore and understand the dataset
4. Check for missing values
5. Separate legitimate and fraudulent transactions
6. Analyze the transaction data
7. Handle the imbalanced dataset using undersampling
8. Separate features and target
9. Split the dataset into training and testing data
10. Train the Logistic Regression model
11. Make predictions
12. Evaluate the model using accuracy score

---

## Handling Imbalanced Data

The dataset contains significantly more legitimate transactions than fraudulent transactions.

To create a balanced dataset, legitimate transactions are randomly sampled and combined with fraudulent transactions.

```python
legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample, fraud], axis=0)
```

This creates a more balanced dataset for training the Machine Learning model.

---

## Feature and Target Separation

The input features and target variable are separated.

```python
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
```

Here:

- `X` contains the transaction features.
- `Y` contains the transaction class.

---

## Train-Test Split

The dataset is divided into training and testing data.

- **80%** → Training Data
- **20%** → Testing Data

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)
```

The training data is used to train the model, while the testing data is used to evaluate its performance.

---

## Machine Learning Model

The **Logistic Regression** algorithm is used for classification.

python
model = LogisticRegression(max_iter=5000)

model.fit(X_train, Y_train)


The model learns from the training data to classify transactions as legitimate or fraudulent.

---

## Model Evaluation

The performance of the model is evaluated using the **Accuracy Score**.

### Training Data Accuracy

```python
X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(
    X_train_prediction,
    Y_train
)

print('Accuracy on Training data : ', training_data_accuracy)
```

### Testing Data Accuracy


X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(
    X_test_prediction,
    Y_test
)

print('Accuracy on Test data : ', test_data_accuracy)
```

---

## Results

The Logistic Regression model is trained on the balanced credit card transaction dataset.

The model's performance is evaluated on both training and testing datasets using the accuracy score.

---

## Conclusion

This project demonstrates the use of Machine Learning for detecting fraudulent credit card transactions.

The imbalanced dataset is handled using **undersampling**, and a **Logistic Regression** model is trained to classify transactions as legitimate or fraudulent.

The model is evaluated using accuracy scores on both training and testing data, demonstrating how Machine Learning can be applied to credit card fraud detection.

---

## Future Improvements

The project can be further improved by:

- Comparing Logistic Regression with other Machine Learning algorithms.
- Using Precision, Recall and F1-Score for evaluation.
- Creating a Confusion Matrix.
- Using techniques such as SMOTE for handling imbalanced data.
- Developing a web application for real-time fraud prediction.

---

## Author

Konanki Charan
