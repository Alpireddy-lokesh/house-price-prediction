import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('output/house_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set Streamlit page configuration
st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")

# App title
st.title("🏠 House Price Prediction App")
st.markdown("Enter the details of the house to predict its price 💰")

# Input sliders
area = st.slider("Area (in sq. ft)", 500, 5000, 1000, step=100)
bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
bathrooms = st.slider("Number of Bathrooms", 1, 10, 2)
parking = st.slider("Parking Capacity", 0, 5, 1)

# Predict button
if st.button("Predict Price"):
    # Create input dataframe with feature names to avoid warning
    input_df = pd.DataFrame([[area, bedrooms, bathrooms, parking]],
                            columns=["area", "bedrooms", "bathrooms", "parking"])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display result
    st.success(f"🏷️ Estimated House Price: ₹ {int(prediction):,}")

# Footer
st.markdown("---")
st.markdown("🔧 Built by Lokesh | Powered by Streamlit + Scikit-learn")
