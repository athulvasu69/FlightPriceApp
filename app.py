import streamlit as st
import pandas as pd
import joblib

model = joblib.load("flight_model_v2.pkl")
encoders = joblib.load("encoders_v2.pkl")

st.set_page_config(page_title="Flight Price Predictor")

st.title("✈️ Flight Ticket Price Prediction")

input_data = {}

columns = [
    'airline',
    'from',
    'to',
    'dep_time',
    'arr_time',
    'stop',
    'time_taken'
]

for col in columns:
    options = list(encoders[col].classes_)
    selected = st.selectbox(
        col.replace("_", " ").title(),
        options
    )
    input_data[col] = encoders[col].transform([selected])[0]

passengers = st.number_input(
    "👥 Number of Passengers",
    min_value=1,
    max_value=20,
    value=1,
    step=1
)

if st.button("Predict Ticket Price"):
    input_df = pd.DataFrame([input_data], columns=columns)

    price_per_person = model.predict(input_df)[0]
    total_price = price_per_person * passengers

    st.success(f"💰 Price Per Passenger: ₹ {price_per_person:,.0f}")
    st.info(f"👥 Total Price for {passengers} Passenger(s): ₹ {total_price:,.0f}")
