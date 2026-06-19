import streamlit as st
import pandas as pd
import joblib

model = joblib.load("flight_price_model_small.pkl")

st.title("Flight Price Prediction")

airline = st.number_input("airline", step=1)
ch_code = st.number_input("ch_code", step=1)
num_code = st.number_input("num_code", step=1)
dep_time = st.number_input("dep_time", step=1)
from_city = st.number_input("from", step=1)
time_taken = st.number_input("time_taken")
stop = st.number_input("stop", step=1)
arr_time = st.number_input("arr_time", step=1)
to = st.number_input("to", step=1)

day = st.number_input("day", 1, 31)
month = st.number_input("month", 1, 12)
year = st.number_input("year", 2024, 2030)

if st.button("Predict"):

    data = pd.DataFrame([[

        airline,
        ch_code,
        num_code,
        dep_time,
        from_city,
        time_taken,
        stop,
        arr_time,
        to,
        day,
        month,
        year

    ]], columns=[

        'airline',
        'ch_code',
        'num_code',
        'dep_time',
        'from',
        'time_taken',
        'stop',
        'arr_time',
        'to',
        'day',
        'month',
        'year'

    ])

    prediction = model.predict(data)

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")
