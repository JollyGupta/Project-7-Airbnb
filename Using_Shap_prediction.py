import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime
from custom_features import add_custom_features

# Load your saved pipeline (must include preprocessor + model)
pipeline = joblib.load("catboost_pipeline.pkl")
model = pipeline.named_steps['model']

# Clean & prepare dataframe before prediction
def clean_df_for_prediction(df):
    df = df.copy()

    # Replace boolean values
    df = df.replace({'t': 1, 'f': 0, True: 1, False: 0})

    # Convert money columns
    for col in ['cleaning_fee', 'extra_people']:
        if col in df.columns:
            df[col] = df[col].replace('[\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert object columns to string
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

    return df.fillna(0)

# Streamlit setup
st.set_page_config(page_title="Airbnb SHAP Price Predictor", layout="wide")
st.title("🏠 Airbnb Price Prediction App")
st.markdown("Upload Airbnb listing data to predict price and explain results using SHAP.")

uploaded_file = st.file_uploader("📤 Upload Airbnb CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    ids = df['id']
    df = df.drop(columns='price', errors='ignore')

    # Define columns used during training
    columns_used = [
        'host_is_superhost', 'host_identity_verified', 'city', 'latitude',
        'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
        'bedrooms', 'beds', 'bed_type', 'cleaning_fee', 'extra_people',
        'number_of_reviews', 'review_scores_rating', 'cancellation_policy',
        'host_since'
    ]

    df_pred = df[columns_used].copy()
    df_pred = clean_df_for_prediction(df_pred)
    df_pred = add_custom_features(df_pred)

    # ✅ Predict
    predicted_prices = pipeline.predict(df_pred)

    # Show results
    df_result = pd.DataFrame({
        'id': ids,
        'predicted_price': predicted_prices
    })

    st.subheader("💸 Predicted Prices")
    st.dataframe(df_result)

    # Download button
    csv = df_result.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "predictions.csv", "text/csv")

    # =============================
    # SHAP EXPLANATION
    # =============================
    st.header("🧠 Explainable AI: SHAP")

    # Transform input like training
    transformed_X = pipeline.named_steps['preprocessor'].transform(df_pred)

    # SHAP Explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(transformed_X)

    row_index = st.slider("Select Row for SHAP Waterfall", 0, len(df_pred) - 1, 0)
    st.subheader(f"🔍 SHAP for ID {ids.iloc[row_index]}")

    fig1, ax1 = plt.subplots()
    shap.plots.waterfall(shap_values[row_index], show=False)
    st.pyplot(fig1)

    st.subheader("🌍 Global Feature Importance (SHAP Beeswarm)")
    fig2, ax2 = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig2)

   
  
   
  
    
