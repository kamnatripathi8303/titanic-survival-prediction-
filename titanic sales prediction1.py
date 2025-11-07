# ---------------------------------------------------
# PROJECT: TITANIC SURVIVAL PREDICTION
# ---------------------------------------------------

# STEP 1: IMPORT REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------
# STEP 2: LOAD DATASET
# You can use Titanic dataset from Kaggle or seaborn built-in dataset.
# If you don't have the file, seaborn provides an inbuilt Titanic dataset:
titanic = sns.load_dataset('titanic')
print("✅ Dataset Loaded Successfully!")
print(titanic.head())

# ---------------------------------------------------
# STEP 3: DATA PREPROCESSING

# Drop columns that are not useful for prediction
titanic.drop(['deck', 'embark_town', 'alive'], axis=1, inplace=True)

# Handle missing values
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# Convert categorical columns to numeric using LabelEncoder
le = LabelEncoder()
for col in ['sex', 'embarked', 'class', 'who', 'adult_male', 'alone']:
    titanic[col] = le.fit_transform(titanic[col])

print("\n✅ Data Cleaned and Encoded Successfully!")
print(titanic.info())

# ---------------------------------------------------
# STEP 4: DEFINE FEATURES AND TARGET
X = titanic.drop('survived', axis=1)
y = titanic['survived']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# STEP 5: TRAIN THE MODEL (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\n✅ Model Trained Successfully!")

# ---------------------------------------------------
# STEP 6: MAKE PREDICTIONS
y_pred = model.predict(X_test)

# ---------------------------------------------------
# STEP 7: EVALUATE THE MODEL
print("\n📊 MODEL EVALUATION:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------
# STEP 8: VISUALIZE RESULTS

# Confusion Matrix Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='g')
plt.title("Titanic Survival Prediction - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Survival count plot
plt.figure(figsize=(6,4))
sns.countplot(x='survived', data=titanic, palette='Set2')
plt.title("Titanic Survival Count")
plt.xlabel("0 = Not Survived, 1 = Survived")
plt.ylabel("Number of Passengers")
plt.show()

# ---------------------------------------------------
# STEP 9: CONCLUSION
print("\n✅ Project Completed Successfully!")
print("The Logistic Regression model predicts Titanic survival with good accuracy.")
