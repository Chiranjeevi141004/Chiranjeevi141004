import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("best_model.pkl")

# Define categorical features for encoding
cat_cols = ["Item_Fat_Content", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type", "Item_Type"]

# Streamlit UI
st.title("Big Mart Sales Prediction App")
st.write("Enter the item details to predict the sales.")

# User Inputs
item_weight = st.number_input("Item Weight", min_value=0.0, format="%.2f")
item_visibility = st.number_input("Item Visibility", min_value=0.0, format="%.4f")
item_mrp = st.number_input("Item MRP", min_value=0.0, format="%.2f")
outlet_age = st.number_input("Outlet Age", min_value=0, format="%d")

# Dropdowns for categorical variables
item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
outlet_location = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
outlet_type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])
item_type = st.selectbox("Item Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned", "Breads", "Starchy Foods", "Others", "Seafood"])

# Prediction function
def predict_sales():
    # Convert input data to DataFrame
    input_data = pd.DataFrame([[item_weight, item_visibility, item_mrp, outlet_age, item_fat_content, outlet_size, outlet_location, outlet_type, item_type]],
                              columns=["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Age", "Item_Fat_Content", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type", "Item_Type"])

    # Encoding categorical features
    input_data = pd.get_dummies(input_data)

    # Ensure the input columns match the trained model features
    model_features = joblib.load("model_features.pkl")  # Load stored feature names
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing columns with default value 0

    input_data = input_data[model_features]  # Ensure correct column order

    # Predict sales
    prediction = model.predict(input_data)
    return round(prediction[0], 2)

if st.button("Predict Sales"):
    sales_prediction = predict_sales()
    st.success(f"Predicted Sales: ₹ {sales_prediction}")
