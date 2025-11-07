# -------------------------------------------------------
# PROJECT: MOVIE RATING PREDICTION WITH PYTHON
# -------------------------------------------------------

# STEP 1: IMPORT REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------------
# STEP 2: CREATE / LOAD MOVIE DATASET
# (You can also use a real dataset from Kaggle like IMDb or TMDB dataset)

# For demo purpose, we’ll create a small sample dataset
data = {
    'Genre': ['Action', 'Drama', 'Comedy', 'Action', 'Romance', 'Horror', 'Drama', 'Comedy', 'Action', 'Horror'],
    'Director': ['Nolan', 'Spielberg', 'Judd', 'Nolan', 'Cameron', 'Jordan', 'Spielberg', 'Judd', 'Nolan', 'Jordan'],
    'Lead_Actor': ['Bale', 'Hanks', 'Rogen', 'DiCaprio', 'Winslet', 'Peele', 'Hanks', 'Rogen', 'Hardy', 'Peele'],
    'Budget_Million': [200, 150, 60, 180, 100, 40, 160, 55, 190, 30],
    'Runtime_Min': [150, 160, 120, 155, 130, 110, 158, 115, 152, 108],
    'Audience_Score': [9.0, 8.5, 7.0, 8.8, 8.2, 6.5, 8.4, 7.2, 8.9, 6.8]
}

df = pd.DataFrame(data)
print("✅ Dataset Created Successfully!")
print(df.head())

# -------------------------------------------------------
# STEP 3: DATA PREPROCESSING

# Encode categorical features
le = LabelEncoder()
for col in ['Genre', 'Director', 'Lead_Actor']:
    df[col] = le.fit_transform(df[col])

# Separate features (X) and target (y)
X = df.drop('Audience_Score', axis=1)
y = df['Audience_Score']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------
# STEP 4: SPLIT INTO TRAINING & TESTING SETS
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# STEP 5: TRAIN THE MODEL (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)
print("\n✅ Model Trained Successfully!")

# -------------------------------------------------------
# STEP 6: MAKE PREDICTIONS
y_pred = model.predict(X_test)

# -------------------------------------------------------
# STEP 7: EVALUATE MODEL PERFORMANCE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 MODEL PERFORMANCE:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("R² Score:", round(r2, 2))

# -------------------------------------------------------
# STEP 8: VISUALIZE RESULTS
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color='purple')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.show()

# -------------------------------------------------------
# STEP 9: CONCLUSION
print("\n✅ PROJECT COMPLETED SUCCESSFULLY!")
print("This model can be improved using larger datasets and advanced algorithms like Random Forest or XGBoost.")
