import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from custom_features import add_custom_features

# Load model pipeline
pipeline = joblib.load("catboost_pipeline.pkl")

st.title("Airbnb Price Prediction App By Jolly Gupta")

uploaded_file = st.file_uploader("Hey Jolly Upload Airbnb CSV File", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    ids = df['id']
    df = df.drop(columns='price', errors='ignore')

    # Select only the columns used in training
    columns_used_in_training = [
        'host_is_superhost', 'host_identity_verified', 'city', 'latitude',
        'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
        'bedrooms', 'beds', 'bed_type', 'cleaning_fee', 'extra_people',
        'number_of_reviews', 'review_scores_rating', 'cancellation_policy',
        'host_since'
    ]

    df_pred = df[columns_used_in_training].copy()

    # Fix boolean columns
    bool_cols = df_pred.select_dtypes(include='bool').columns
    df_pred[bool_cols] = df_pred[bool_cols].astype(str)

    # Add custom features
    df_pred = add_custom_features(df_pred)

    # Predict
    predicted_prices = pipeline.predict(df_pred)

    # Combine
    df_result = pd.DataFrame({
        'id': ids,
        'predicted_price': predicted_prices
    })

    st.subheader("Predicted Prices")
    st.write(df_result)

    csv = df_result.to_csv(index=False)
    st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")

