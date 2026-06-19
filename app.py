import streamlit as st
import pandas as pd
import joblib

# Load saved model and encoders
model = joblib.load("flight_model_v2.pkl")
encoders = joblib.load("encoders_v2.pkl")

st.set_page_config(page_title="Flight Price Predictor")

st.title("✈️ Flight Ticket Price Comparison")

st.write("Enter flight details once and compare Air India and Vistara prices.")

# Model column order
columns = [
    'airline',
    'from',
    'to',
    'dep_time',
    'arr_time',
    'stop',
    'time_taken'
]

# User inputs
from_city = st.selectbox("From", list(encoders['from'].classes_))
to_city = st.selectbox("To", list(encoders['to'].classes_))

dep_time = st.selectbox("Departure Time", list(encoders['dep_time'].classes_))
arr_time = st.selectbox("Arrival Time", list(encoders['arr_time'].classes_))

stop = st.selectbox("Stops", list(encoders['stop'].classes_))
time_taken = st.selectbox("Time Taken", list(encoders['time_taken'].classes_))

passengers = st.number_input(
    "Number of Passengers",
    min_value=1,
    max_value=20,
    value=1,
    step=1
)

# Predict button
if st.button("Compare Prices"):

    if from_city == to_city:
        st.warning("From and To cities cannot be the same.")
    else:
        results = []

        for airline in encoders['airline'].classes_:

            input_data = {
                'airline': encoders['airline'].transform([airline])[0],
                'from': encoders['from'].transform([from_city])[0],
                'to': encoders['to'].transform([to_city])[0],
                'dep_time': encoders['dep_time'].transform([dep_time])[0],
                'arr_time': encoders['arr_time'].transform([arr_time])[0],
                'stop': encoders['stop'].transform([stop])[0],
                'time_taken': encoders['time_taken'].transform([time_taken])[0]
            }

            input_df = pd.DataFrame([input_data], columns=columns)

            price_per_person = model.predict(input_df)[0]
            total_price = price_per_person * passengers

            results.append({
                "Airline": airline,
                "Price Per Passenger": f"₹ {price_per_person:,.0f}",
                "Total Price": f"₹ {total_price:,.0f}"
            })

        results_df = pd.DataFrame(results)

        st.subheader("Price Comparison")
        st.table(results_df)
