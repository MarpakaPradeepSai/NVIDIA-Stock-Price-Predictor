import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the NVIDIA model
model_file = 'NVIDIA_LSTM_5(1.38).h5'
try:
    model = load_model(model_file)
    print("NVIDIA model loaded successfully.")
except Exception as e:
    print(f"Error loading NVIDIA model: {e}")

# Function to get the stock data
def get_stock_data(ticker='NVDA'):
    data = yf.download(ticker, period='max')
    return data

# Function to generate a list of business days
def generate_business_days(start_date, num_days):
    """
    Generate a list of business days starting from start_date for num_days.
    """
    return pd.bdate_range(start=start_date, periods=num_days).tolist()

# Function to make predictions for business days
def predict_next_business_days(model, data, look_back=5, days=5):
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
    return predictions

# Streamlit app layout
st.markdown("<h1 style='text-align: center; font-size: 49px;'>Stock-Price-Predictor 📈📉💰</h1>", unsafe_allow_html=True)

# Center the NVIDIA logo image
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png" width="560">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)  # Add a gap between rows

# Set NVIDIA as the selected stock
stock = 'NVDA'

# User input for number of business days to forecast
num_days = st.slider("Select number of business days to forecast", min_value=1, max_value=30, value=5)

# Display current date
current_date = datetime.now().strftime('%Y-%m-%d')
st.write(f"Current Date: {current_date}")

# Apply custom CSS to change button colors and add green outline when clicked
st.markdown(
    """
    <style>
    div.stButton > button#forecast-button {
        background-color: green; /* Green color for the Forecast button */
        color: white;
    }
    div.stButton > button#forecast-button:focus,
    div.stButton > button#forecast-button:hover,
    div.stButton > button#forecast-button:active {
        color: white; /* Keep text white on hover, focus, and active states for the Forecast button */
        outline: 2px solid green; /* Green outline for the clicked button */
    }

    div.stButton > button:not(#forecast-button) {
        background-color: red; /* Red color for all other buttons */
        color: white;
    }
    div.stButton > button:not(#forecast-button):focus,
    div.stButton > button:not(#forecast-button):hover,
    div.stButton > button:not(#forecast-button):active {
        color: white; /* Keep text white on hover, focus, and active states for the other buttons */
        outline: 2px solid green; /* Green outline for the clicked button */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use unique key for the "Forecast" button
if st.button(f'Predict Next {num_days} Days Stock Prices for {stock}', key='forecast-button'):
    # Load stock data
    stock_data = get_stock_data(stock)
    close_prices = stock_data['Close'].values.reshape(-1, 1)
    dates = stock_data.index

    # Display the historical data
    st.markdown(f"### Historical Data for NVIDIA")
    st.dataframe(stock_data)

    # Predict the next num_days business days
    look_back = 5
    predictions = predict_next_business_days(model, close_prices, look_back=look_back, days=num_days)
    
    # Create dates for the predictions
    last_date = dates[-1]
    prediction_dates = generate_business_days(last_date + timedelta(days=1), num_days)

    st.write(" ")
    
    # Prepare data for plotting the historical and predicted prices
    fig, ax = plt.subplots()
    ax.plot(dates, close_prices, label='Historical Prices')
    ax.plot(prediction_dates, predictions, label='Predicted Prices', linestyle='--', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'{stock} Stock Prices', fontsize=10, fontweight='bold')
    ax.legend()

    st.pyplot(fig)

    st.write(" ")
    
    # Plot only the predicted stock prices
    fig2, ax2 = plt.subplots()
    ax2.plot(prediction_dates, predictions, marker='o', color='blue')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Predicted Price')
    ax2.set_title(f'Predicted Stock Prices for the Next {num_days} Business Days ({stock})', fontsize=10, fontweight='bold')
    
    # Use DayLocator to specify spacing of tick marks and set the format for the date labels
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.xticks(rotation=45)
    
    st.pyplot(fig2)
    
    st.write(" ")
    
    # Show predictions in a table format
    prediction_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted Price': predictions.flatten()
    })
    st.markdown(f"##### Predicted Stock Prices for the Next {num_days} Business Days ({stock})")
    st.table(prediction_df)

# Center the GIF image at the end of the app
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZmF0cnVzNzN5MTU3dXZ6MTVmcjhjMmFndDdqaDdsNGpkdmdnZG96MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/wIVA0zh5pt0G5YtcAL/giphy.webp" width="500">
    </div>
    """,
    unsafe_allow_html=True
)
