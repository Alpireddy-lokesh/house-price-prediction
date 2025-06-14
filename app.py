import streamlit as st
import pickle
import os
import numpy as np

# Set the correct model path
model_path = os.path.join("output", "house_model.pkl")

# Check if model file exists
if not os.path.exists(model_path):
    st.error("❌ Model file not found at 'output/house_model.pkl'. Please make sure it is uploaded to GitHub and committed.")
else:
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Streamlit App UI
    st.title("🏡 House Price Prediction App")
    st.write("Enter the details below to predict the house price:")

    # Inputs (adjust based on your model's features)
    area = st.slider("Area (in sq ft)", 500, 10000, 1500)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
    stories = st.selectbox("Stories", [1, 2, 3, 4])
    parking = st.selectbox("Parking Capacity", [0, 1, 2, 3])

    # Add more features as needed...
    # Example: mainroad, guestroom, basement, etc.

    # Button to predict
    if st.button("Predict Price"):
        try:
            # Prepare the input data for prediction (update order/length as per model)
            input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])

            # Predict
            prediction = model.predict(input_data)[0]

            st.success(f"💰 Estimated House Price: ₹ {prediction:,.2f}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
