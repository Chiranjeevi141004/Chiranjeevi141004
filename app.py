import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Function to download model from Google Drive
def download_model_from_drive():
    file_id = "1u-8F7OpmTYDiRgxxXNmKGbogbIFORTuS"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(BytesIO(response.content))
    else:
        st.error("Failed to download the model. Check the file ID and permissions.")
        return None

# Load the model
st.title("Big Mart Sales Prediction")
st.write("Predict the sales of an item in a retail outlet based on various features.")

# Input fields for user
item_weight = st.number_input("Item Weight", min_value=0.0, value=5.0)
item_visibility = st.number_input("Item Visibility", min_value=0.0, value=0.02)
item_mrp = st.number_input("Item MRP", min_value=0.0, value=100.0)
outlet_establishment_year = st.number_input("Outlet Establishment Year", min_value=1980, max_value=2025, value=2000)
item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
outlet_type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])

# Encoding categorical variables
label_encoders = {
    "Item_Fat_Content": LabelEncoder().fit(["Low Fat", "Regular"]),
    "Outlet_Size": LabelEncoder().fit(["Small", "Medium", "High"]),
    "Outlet_Location_Type": LabelEncoder().fit(["Tier 1", "Tier 2", "Tier 3"]),
    "Outlet_Type": LabelEncoder().fit(["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"]),
}

# Transform categorical features
item_fat_content = label_encoders["Item_Fat_Content"].transform([item_fat_content])[0]
outlet_size = label_encoders["Outlet_Size"].transform([outlet_size])[0]
outlet_location_type = label_encoders["Outlet_Location_Type"].transform([outlet_location_type])[0]
outlet_type = label_encoders["Outlet_Type"].transform([outlet_type])[0]

# Prepare input data
data = np.array([[item_weight, item_visibility, item_mrp, outlet_establishment_year, item_fat_content, outlet_size, outlet_location_type, outlet_type]])

# Load the model and predict
if st.button("Predict Sales"):
    model = download_model_from_drive()
    if model:
        prediction = model.predict(data)[0]
        st.success(f"Predicted Sales: ${prediction:.2f}")
    else:
        st.error("Could not load the model. Try again later.")
