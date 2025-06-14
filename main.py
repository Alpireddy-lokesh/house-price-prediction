# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_squared_error
# import pickle
# import os
# import numpy as np

# # Load the dataset
# df = pd.read_csv("data.csv")

# # Features and target
# X = df[['area', 'bedrooms', 'bathrooms', 'parking']]
# y = df['price']

# # Train the model
# model = LinearRegression()
# model.fit(X, y)

# # Predict and evaluate
# y_pred = model.predict(X)
# r2 = r2_score(y, y_pred)
# mse = mean_squared_error(y, y_pred)
# rmse = np.sqrt(mse)

# print("Model Evaluation:")
# print("R² Score:", r2)
# print("RMSE:", rmse)

# # Save the model
# os.makedirs("output", exist_ok=True)
# with open("output/house_model.pkl", "wb") as file:
#     pickle.dump(model, file, protocol=4)

# print("✅ Model saved successfully to output/house_model.pkl")
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import os

# Load the data
df = pd.read_csv("data.csv")

# ✅ Use the correct column names
X = df[["area", "bedrooms", "bathrooms", "parking"]]
y = df["price"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
os.makedirs("output", exist_ok=True)
with open(os.path.join("output", "house_model.pkl"), "wb") as file:
    pickle.dump(model, file, protocol=4)

print("✅ Model trained and saved to output/house_model.pkl")
