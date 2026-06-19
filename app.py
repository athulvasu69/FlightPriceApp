import streamlit as st
import pandas as pd
import joblib

model = joblib.load("flight_price_model_small.pkl")
encoders = joblib.load("encoders.pkl")

st.title("✈️ Flight Price Prediction")

input_data = {}

columns = [
    'airline', 'ch_code', 'num_code', 'dep_time', 'from',
    'time_taken', 'stop', 'arr_time', 'to', 'day', 'month', 'year'
]

for col in columns:
    if col in encoders:
        options = list(encoders[col].classes_)
        value = st.selectbox(col, options)
        input_data[col] = encoders[col].transform([value])[0]

    elif col == "day":
        input_data[col] = st.number_input("day", min_value=1, max_value=31, value=11)

    elif col == "month":
        input_data[col] = st.number_input("month", min_value=1, max_value=12, value=2)

    elif col == "year":
        input_data[col] = st.number_input("year", min_value=2020, max_value=2030, value=2022)

    else:
        input_data[col] = st.number_input(col, value=0)

if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data], columns=columns)
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Flight Price: ₹ {prediction:,.2f}")
