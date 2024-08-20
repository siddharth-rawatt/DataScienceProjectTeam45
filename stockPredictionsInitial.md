import yfinance as yf
import pmdarima as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

def detect_seasonal_period(time_series, max_lag=24):
    """
    Automatically detect the most significant seasonal period based on ACF peaks.

    Parameters:
    - time_series: The time series data.
    - max_lag: The maximum number of lags to check for seasonality.

    Returns:
    - seasonal_period: The detected seasonal period, or None if no significant seasonality is found.
    """
    acf_values = acf(time_series, nlags=max_lag)
    peaks = [i for i in range(1, len(acf_values) - 1) if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]]

    if peaks:
        # Select the first significant peak as the seasonal period
        seasonal_period = peaks[0]
        return seasonal_period
    else:
        return None

def fetch_and_predict(ticker, forecast_days):
    """
    Fetch historical stock data and forecast future prices using ARIMA/SARIMA.

    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
    - forecast_days: Number of days to forecast (e.g., 14 for two weeks)

    Returns:
    - forecast: Predicted stock prices for the next 'forecast_days'
    """
    # Fetch stock data
    data = yf.download(ticker)
    
    # Use the 'Close' prices for prediction
    stock_prices = data['Close']
    
    # This only uses the last 100 days of data
    stock_prices = stock_prices[-100:]
    
    # Automatically detect the seasonal period
    seasonal_period = detect_seasonal_period(stock_prices)
    
    if seasonal_period:
        print(f"Seasonality detected with period: {seasonal_period}. Using SARIMA model.")
        auto_model = pm.auto_arima(
            stock_prices, 
            seasonal=True, 
            m=seasonal_period,  # Start with detected period, adjust manually if necessary
            stepwise=True, 
            suppress_warnings=True, 
            trace=True,
            error_action='ignore',
            start_p=1, max_p=3,  # Adjust the AR terms
            start_q=1, max_q=3,  # Adjust the MA terms
            start_P=1, max_P=2,  # Adjust seasonal AR terms
            start_Q=0, max_Q=2,  # Adjust seasonal MA terms
            d=None, D=1,  # Let the model automatically determine differencing
            max_order=None,  # Limit the overall complexity
            maxiter=100  # Increase iterations if necessary
        )
    else:
        print("No seasonality detected. Using ARIMA model.")
        auto_model = pm.auto_arima(
            stock_prices, 
            seasonal=False,  # Set to False for a non-seasonal ARIMA model
            stepwise=False, 
            suppress_warnings=True, 
            trace=True,
            error_action='ignore',
            start_p=1, max_p=3,  # Adjust the AR terms
            start_q=1, max_q=3,  # Adjust the MA terms
            d=1,  # Let the model automatically determine the order of differencing (d)
            max_order=None,  # Do not limit the overall complexity
            maxiter=100  # Increase the number of iterations if necessary
        )
    
    # Forecast the specified number of days
    forecast, conf_int = auto_model.predict(n_periods=forecast_days, return_conf_int=True, alpha=0.20)  # 80% confidence interval
    
    # Create a date index for the forecasted values
    forecast_dates = pd.date_range(stock_prices.index[-1], periods=forecast_days+1, freq='D')[1:]
    
    # Generate in-sample predictions (last year in this case)
    in_sample_predictions = auto_model.predict_in_sample()
    
    # Skip the first seasons value in in-sample predictions due to it being lost in differencing
    in_sample_predictions = in_sample_predictions[seasonal_period:]
    
    # Create a date index for the in-sample predictions
    in_sample_dates = stock_prices.index[seasonal_period:]
    
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Actual vs. In-Sample Predictions for the last year
    plt.subplot(1, 2, 1)
    plt.plot(stock_prices.index, stock_prices, label='Actual Prices')
    plt.plot(in_sample_dates, in_sample_predictions, label='In-Sample Predictions', color='red')
    plt.title('Actual Prices vs. In-Sample Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Plot 2: Forecast
    plt.subplot(1, 2, 2)
    plt.plot(stock_prices.index, stock_prices, label='Actual Prices')
    plt.plot(forecast_dates, forecast, label='Forecasted Prices', color='red')
    plt.fill_between(forecast_dates, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='80% Confidence Interval')
    plt.title('Forecast for the Next 14 Days')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print model details
    print("Model Summary:")
    print(auto_model.summary())
    
    return forecast

# Example usage:
ticker = 'DE'  # Deere & Co which is seasonal
forecast_days = 14  # Predict the next 14 days

fetch_and_predict(ticker, forecast_days)
