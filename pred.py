import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from custom_features import add_custom_features

# Load model
pipeline = joblib.load("catboost_pipeline.pkl")

st.title("Airbnb Price Prediction App by Jolly Gupta ✨")

uploaded_file = st.file_uploader("📤 Upload Airbnb CSV File", type=['csv'])

if uploaded_file is not None:
    # Read CSV with encoding fix (Excel issue sometimes)
    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.write("📋 Columns detected:", df.columns.tolist())  # Debug display

    # Check required columns
    required_cols = ['id', 'host_since']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

    # Drop price column if it exists
    df = df.drop(columns='price', errors='ignore')
    ids = df['id']

    # Define training columns
    columns_used_in_training = [
        'host_is_superhost', 'host_identity_verified', 'city', 'latitude',
        'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
        'bedrooms', 'beds', 'bed_type', 'cleaning_fee', 'extra_people',
        'number_of_reviews', 'review_scores_rating', 'cancellation_policy',
        'host_since'
    ]

    # Keep only existing columns (safe fallback)
    df_pred = df[[col for col in columns_used_in_training if col in df.columns]].copy()

    # Convert boolean-like strings to string
    bool_cols = df_pred.select_dtypes(include='bool').columns
    df_pred[bool_cols] = df_pred[bool_cols].astype(str)

    # Add custom features
    df_pred = add_custom_features(df_pred)

    # Make prediction
    predicted_prices = pipeline.predict(df_pred)

    # Combine output
    df_result = pd.DataFrame({
        'id': ids,
        'predicted_price': predicted_prices
    })

    # Show result
    st.subheader("💸 Predicted Airbnb Prices")
    st.dataframe(df_result)

    # Download button
    csv = df_result.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "predicted_prices.csv", "text/csv")
