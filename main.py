import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import numpy as np

# Load dataset
df = pd.read_csv('dataset/house.csv')
print("\nFirst 5 rows:\n", df.head())
print("\nDataset Info:\n", df.info())

# Prepare features and target
X = df[['area', 'bedrooms', 'bathrooms', 'parking']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict and evaluate
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))  # FIXED

print("\nModel Evaluation:")
print("R² Score:", r2)
print("RMSE:", rmse)

# Save the model using protocol=4 for cross-platform compatibility
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, 'house_model.pkl')
with open(model_path, 'wb') as file:
    pickle.dump(model, file, protocol=4)

print(f"\n✅ Model saved successfully to {model_path}")
