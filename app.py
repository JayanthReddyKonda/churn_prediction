import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📉",
    layout="centered",
)

st.title("Customer Churn Prediction")
st.caption("Predict churn probability using a trained ANN + saved ColumnTransformer.")

# -----------------------------
# Cached loading
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("churn_model.keras")
    preprocessor = joblib.load("column_transformer.joblib")
    return model, preprocessor

model, preprocessor = load_artifacts()

# -----------------------------
# Input form
# -----------------------------
with st.form("churn_form"):
    st.subheader("Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        RowNumber = st.number_input("Row Number", min_value=0, value=0, step=1)
        CustomerId = st.text_input("Customer ID", value="00000000")
        Surname = st.text_input("Surname", value="SURNAME")
        CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=600, step=1)
        Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        Gender = st.selectbox("Gender", ["Female", "Male"])
        Age = st.number_input("Age", min_value=0, max_value=120, value=40, step=1)

    with col2:
        Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=2, step=1)
        Balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
        NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1, step=1)
        HasCrCard = st.selectbox("Has Credit Card", [0, 1], index=1)
        IsActiveMember = st.selectbox("Is Active Member", [0, 1], index=1)
        EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

    submitted = st.form_submit_button("Predict Churn")

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_df = pd.DataFrame([{
        "RowNumber": RowNumber,
        "CustomerId": CustomerId,
        "Surname": Surname,
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
    }])

    try:
        X = preprocessor.transform(input_df)

        # Keras expects numeric array-like input
        prediction_prob = float(model.predict(X, verbose=0).ravel()[0])
        prediction = 1 if prediction_prob >= 0.5 else 0

        st.subheader("Result")
        if prediction == 1:
            st.error("The customer is likely to churn.")
        else:
            st.success("The customer is not likely to churn.")

        st.metric("Churn Probability", f"{prediction_prob:.2%}")

        with st.expander("Show transformed input"):
            st.write(X)

    except Exception as e:
        st.error(f"Prediction failed: {e}")