# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data Import the dataset to initiate the analysis.

2.Explore Data Examine the dataset to identify patterns, distributions, and relationships.

3.Select Features Determine the most important features to enhance model accuracy and efficiency.

4.Split Data Separate the dataset into training and testing sets for effective validation.

5.Train Model Use the training data to build and train the model.

6.Evaluate Model Measure the model’s performance on the test data with relevant metrics.

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: LAAKSHIT D
RegisterNumber:  212222230071
*/
```
```py
#Import necessary libraries

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Step 1: Data Loading

data = pd.read_csv('tumor.csv')

#Step 2: Data Exploration
#Display the first few rows and column names for verification
print(data.head())
print(data.columns)

#Step 3: Select features and target variable

#Drop id and other non-feature columns, using diagnosis as the target
x = data.drop(columns=['Class']) # Remove any irrelevant columns
y = data['Class'] # The target column indicating benign or malignant diagnosis

#Step 4: Data Splitting

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Step 5: Model Training
#Initialize and train the Decision Tree Classifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#Step 6: Model Evaluation
#Predicting on the test set

y_pred = model.predict(X_test)

#Calculate accuracy and print classification metrics

accuracy = accuracy_score(y_test, y_pred)
print("LAAKSHIT D")
print("212222230071")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

#Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlOrRd")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:

<img width="654" height="310" alt="image" src="https://github.com/user-attachments/assets/6f992b5e-4e62-4021-a273-640d0df527fe" />

<img width="458" height="244" alt="image" src="https://github.com/user-attachments/assets/82001db9-9549-4917-a91f-1c32f42ffc89" />

<img width="614" height="518" alt="image" src="https://github.com/user-attachments/assets/a4bd5015-5cd3-4866-9370-150cb556137a" />


## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
