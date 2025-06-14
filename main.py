import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Load dataset
df = pd.read_csv('dataset/house.csv')

# Display basic info
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Define input and output
X = df[['area', 'bedrooms', 'bathrooms', 'parking']]
y = df['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # ✅ Fix here

print("\nModel Evaluation:")
print(f"R² Score: {r2}")
print(f"RMSE: {rmse}")

# Ensure output folder
os.makedirs('output', exist_ok=True)

# Save model safely
with open('output/house_model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\n✅ Model saved successfully to output/house_model.pkl")
