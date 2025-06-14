import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os
import pickle

# Load your dataset
df = pd.read_csv("data.csv")

# Show first few rows
print("First 5 rows:")
print(df.head())

# Dataset info
print("\nDataset Info:")
print(df.info())

# Prepare input and output
X = df[['area', 'bedrooms', 'bathrooms', 'parking']]  # input features
y = df['price']  # target output

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict and evaluate
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("\nModel Evaluation:")
print(f"R² Score: {r2}")
print(f"RMSE: {rmse}")

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# ✅ Save model using protocol=4 for Streamlit Cloud compatibility
with open("output/house_model.pkl", "wb") as file:
    pickle.dump(model, file, protocol=4)

print("\n✅ Model saved successfully to output/house_model.pkl")
