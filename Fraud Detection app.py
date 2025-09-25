import streamlit as st
import joblib
import numpy as np
import pandas as pd

rf_model = joblib.load(r"D:\AI\Credit Card Fraud Detection\rf_model.joblib")
xgb_model = joblib.load(r"D:\AI\Credit Card Fraud Detection\xgb_model.joblib")
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("Credit Card Fraud Detector")
RF_WEIGHT = 0.4
XGB_WEIGHT = 0.6

def predict_single(features):
    X = np.array(features).reshape(1, -1)
    rf_prob = rf_model.predict_proba(X)[0][1]
    xgb_prob = xgb_model.predict_proba(X)[0][1]
    combined_prob = (RF_WEIGHT * rf_prob) + (XGB_WEIGHT * xgb_prob)
    combined_pred = int(combined_prob >= 0.5)
    if combined_pred == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
    return {
        "RF_Probability": float(rf_prob),
        "XGB_Probability": float(xgb_prob),
        "Combined_Probability": float(combined_prob),
        "Final_Prediction": combined_pred
    }

def predict_batch(df):
    results = []
    for i in range(len(df)):
        features = df.iloc[i].values
        results.append(predict_single(features))
    return pd.DataFrame(results)

st.sidebar.header("⚙️ Options")
mode = st.sidebar.radio("Choose input mode:", ["Single Transaction", "Batch CSV Upload"])
if mode == "Single Transaction":
    st.subheader("Enter Transaction Features")
    feature_input = st.text_area("Enter comma-separated features", "")
    if st.button("Predict"):
        if feature_input:
            features = list(map(float, feature_input.split(",")))
            result = predict_single(features)
            st.json(result)
        else:
            st.warning("Please enter features separated by commas.")
else:
    st.subheader("Upload Transactions CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📂 Uploaded Data Preview:", df.head())
        if st.button("Run Predictions"):
            results_df = predict_batch(df)
            output = pd.concat([df, results_df], axis=1)
            st.write("✅ Prediction Results", output)
            fraud_cases = output[output["Final_Prediction"] == 1]
            st.metric("Total Transactions", len(output))
            st.metric("Fraudulent Cases Detected", len(fraud_cases))
            st.subheader("⚠️ Fraud Cases")
            st.dataframe(fraud_cases)
            csv = output.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "fraud_predictions.csv", "text/csv")
st.markdown("---")
st.caption("Developed by Mariam Yasir | Internship Project")