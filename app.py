import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Scoring Prediction System")
st.markdown("Predict whether a customer is creditworthy.")

# ---------------------------
# LOAD MODEL
# ---------------------------
model_paths = {
    "Logistic Regression": "models/trained/logistic_regression_model.pkl",
    "Decision Tree": "models/trained/decision_tree_model.pkl",
    "Random Forest": "models/trained/random_forest_model.pkl",
    "XGBoost": "models/trained/xgboost_model.pkl"
}

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(model_paths.keys())
)

model = joblib.load(model_paths[selected_model])

# Load scaler
try:
    scaler = joblib.load("models/trained/scaler.pkl")
except:
    scaler = None

# ---------------------------
# INPUTS
# ---------------------------
col1, col2 = st.columns(2)

with col1:

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    person_income = st.number_input(
        "Annual Income",
        min_value=0,
        value=50000
    )

    person_emp_length = st.number_input(
        "Employment Length (Years)",
        min_value=0.0,
        value=5.0
    )

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0,
        value=10000
    )

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        value=10.0
    )

with col2:

    loan_percent_income = st.number_input(
        "Loan Percent Income",
        min_value=0.0,
        value=0.2
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length",
        min_value=0,
        value=5
    )

    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "EDUCATION",
            "MEDICAL",
            "PERSONAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    cb_person_default_on_file = st.selectbox(
        "Previous Default",
        ["N", "Y"]
    )

# ---------------------------
# PREDICT
# ---------------------------
if st.button("Predict Credit Risk"):

    input_df = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }])

    # Same preprocessing used during training
    input_df = pd.get_dummies(
        input_df,
        columns=[
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
            "cb_person_default_on_file"
        ],
        drop_first=True
    )

    # Match training columns
    input_df = input_df.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Scale numerical columns
    numerical_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length"
    ]

    if scaler is not None:
        input_df[numerical_cols] = scaler.transform(
            input_df[numerical_cols]
        )

    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Low Risk / Creditworthy")
    else:
        st.error("❌ High Risk / Not Creditworthy")

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(input_df)[0]

        prob_df = pd.DataFrame(
            {
                "Probability": probs
            },
            index=[
                "High Risk",
                "Low Risk"
            ]
        )

        st.subheader("Prediction Confidence")
        st.bar_chart(prob_df)

        st.write(
            f"High Risk Probability: {probs[0]*100:.2f}%"
        )

        st.write(
            f"Low Risk Probability: {probs[1]*100:.2f}%"
        )

st.markdown("---")
st.caption(f"Model Selected: {selected_model}")