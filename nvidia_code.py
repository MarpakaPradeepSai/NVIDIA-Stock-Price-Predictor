import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load the trained LSTM model
model_lstm = load_model('NVIDIA_LSTM_5(1.38).h5')

# Function to get the stock data
def get_stock_data(ticker='NVDA'):
    # Fetch historical stock data with no start date limit (max data)
    data = yf.download(ticker, period='max')
    data = data.asfreq('B')  # Convert to business day frequency
    return data

# Function to make predictions considering only business days
def predict_next_business_days(model, data, look_back=5, days=2):
    # Create a business day frequency range
    future_dates = pd.bdate_range(start=datetime.now(), periods=days)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    last_sequence = data_scaled[-look_back:]
    predictions = []

    for _ in range(days):
        X_input = np.reshape(last_sequence, (1, look_back, 1))
        prediction = model.predict(X_input)
        predictions.append(prediction[0, 0])
        
        # Update the sequence for the next prediction
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    
    # Inverse transform the predictions to the original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Create a DataFrame with business day dates
    prediction_dates = future_dates
    prediction_data = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted Close Price': predictions.flatten()
    })
    
    return prediction_data

# Streamlit app layout
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 50px;  /* Change font size here */
    }
    .image-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .image-container img {
        width: 620px; /* Change the width as needed */
        height: auto; /* Maintain aspect ratio */
    }
    
    .stButton > button {
        background-color: red; /* Change the button color to red */
        color: white; /* Change the button text color to white */
        width: auto; /* Maintain the default button size */
    }
    </style>
    <h1 class="title">Stock Price Predictor 📈📉💰</h1>
    <div class="image-container">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png" alt="NVIDIA Logo">
    </div>
""", unsafe_allow_html=True)

# Get the current date
current_date = datetime.now().strftime('%Y-%m-%d')
st.write(" ")

# Initialize session state for the button text
if 'button_text' not in st.session_state:
    st.session_state.button_text = 'Forecast'

# User input for number of days to forecast
num_days = st.slider("Select number of business days to forecast", min_value=1, max_value=30, value=5)

st.write(f"Current Date: {current_date}")

if st.button(st.session_state.button_text, use_container_width=True):
    # Load stock data
    stock_data = get_stock_data()
    dates = stock_data.index

    # Prepare data for the entire dataset
    close_prices = stock_data['Close'].values.reshape(-1, 1)
    
    # Include all columns in the DataFrame
    historical_data = stock_data.reset_index()

    # Predict the next num_days business day prices
    prediction_data = predict_next_business_days(model_lstm, close_prices, look_back=30, days=num_days)
    
    # Display historical data with all columns
    st.write("### NVIDIA-Historical Stock Data")
    st.dataframe(historical_data)
    st.write(" ")
    # Plot the historical and predicted stock prices
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, stock_data['Close'], label='Historical Close Prices', color='blue')
    ax.plot(prediction_data['Date'], prediction_data['Predicted Close Price'], label='Predicted Close Prices', linestyle='--', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('NVIDIA Stock Prices')
    ax.legend()
    
    st.pyplot(fig)
    st.write(" ")
    # Plot only the predicted stock prices
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.plot(prediction_data['Date'], prediction_data['Predicted Close Price'], marker='o', color='blue')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Predicted Price')
    ax2.set_title(f'Predicted Stock Prices for the Next {num_days} Business Days')
    
    st.pyplot(fig2)

    # Show predictions table with dynamic title
    st.write(f"### Predictions for {num_days} Business Days")
    st.dataframe(prediction_data)

    # Update the button text only if the button is clicked for the first time
    if st.session_state.button_text == 'Forecast':
        st.session_state.button_text = 'Forecast Again'
