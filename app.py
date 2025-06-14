import streamlit as st
import numpy as np
import pickle
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "output", "house_model.pkl")
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit app layout
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")

# Inputs
area = st.slider("Area (in sq. ft.)", min_value=500, max_value=10000, step=100)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3])
parking = st.selectbox("Parking Capacity", [0, 1, 2, 3])

# Prediction button
if st.button("Predict Price"):
    try:
        input_data = np.array([[area, bedrooms, bathrooms, parking]])  # ✅ Only 4 features
        prediction = model.predict(input_data)
        st.success(f"🏷️ Estimated House Price: ₹ {int(prediction[0]):,}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
