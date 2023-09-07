import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from tradingview_ta import TA_Handler, Interval
import plotly.express as px
import numpy as np
import requests     
import statsmodels.api as sm

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import mysql.connector
from mysql.connector import Error
import bcrypt

def create_connection():
    """Create a connection to the MySQL database."""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='stock',
            user='root',
            password='jadu@8134'
        )
        return connection
    except Error as e:
        print(f"Error: {e}")
        return None

def create_user(username, email, password):
    """Create a new user and store their information in the database."""
    try:
        connection = create_connection()
        if connection:
            # Hash the password before storing it in the database
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            cursor = connection.cursor()
            query = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
            data = (username, email, hashed_password)
            cursor.execute(query, data)
            connection.commit()
            cursor.close()
            connection.close()
            return True
    except Error as e:
        print(f"Error: {e}")
    return False

def verify_user(username, password):
    """Verify the user's login credentials."""
    try:
        connection = create_connection()
        if connection:
            cursor = connection.cursor()
            query = "SELECT password FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            
            if result and bcrypt.checkpw(password.encode('utf-8'), result[0].encode('utf-8')):
                return True
    except Error as e:
        print(f"Error: {e}")
    return False

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def fetch_live_stock_price(symbol):
    stock_info = yf.Ticker(symbol)
    live_data = stock_info.history(period='1d')
    if not live_data.empty:
        return live_data['Close'][0]
    return None

def plot_candlestick_chart(data):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'])])
    fig.update_layout(title='Candlestick Chart',
                      xaxis_title='Date',
                      yaxis_title='Stock Price',
                      xaxis_rangeslider_visible=False)
    return fig

def plot_line_chart(data, column, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name=column))
    fig.update_layout(title=title,
                      xaxis_title='Date',
                      yaxis_title=column)
    return fig

def plot_histogram(data, column, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data[column], nbinsx=30))
    fig.update_layout(title=title,
                      xaxis_title=column,
                      yaxis_title='Frequency')
    return fig

def plot_scatter(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['High'], y=data['Low'], mode='markers', name='High vs. Low'))
    fig.update_layout(title='Scatter Plot - High vs. Low',
                      xaxis_title='High Price',
                      yaxis_title='Low Price')
    return fig

def plot_heatmap(data, title):
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(title)
    return fig

def calculate_yearly_profit(data):
    data['Year'] = data.index.year
    data['Yearly Profit'] = (data['Close'] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
    yearly_profit = data.groupby('Year')['Yearly Profit'].last().reset_index()
    return yearly_profit

def plot_yearly_profit_bar_chart(yearly_profit):
    fig = px.bar(yearly_profit, x='Year', y='Yearly Profit', title='Yearly Profit (%)')
    return fig

def calculate_signals(data, short_window=50, long_window=200):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0
    
    # Create short and long-term moving averages
    signals['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    
    # Create Buy and Sell signals
    signals.loc[signals['Short_MA'] > signals['Long_MA'], 'Signal'] = 1.0  # Buy Signal
    signals.loc[signals['Short_MA'] <= signals['Long_MA'], 'Signal'] = -1.0  # Sell Signal
    
    return signals

def plot_signals_chart(data, signals):
    fig = go.Figure()

    # Add the stock price
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Stock Price'))

    # Add Buy and Sell signals on the chart
    buy_signals = signals[signals['Signal'] == 1.0]
    sell_signals = signals[signals['Signal'] == -1.0]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=data.loc[buy_signals.index, 'Close'], mode='markers', 
                             marker=dict(color='green', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=data.loc[sell_signals.index, 'Close'], mode='markers', 
                             marker=dict(color='red', size=10), name='Sell Signal'))

    fig.update_layout(title='Stock Price with Buy/Sell Signals',
                      xaxis_title='Date',
                      yaxis_title='Stock Price')
    
    return fig

def calculate_signals(data, short_window=50, long_window=200):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0
    
    # Create short and long-term moving averages
    signals['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    
    # Create Buy and Sell signals
    signals.loc[signals['Short_MA'] > signals['Long_MA'], 'Signal'] = 1.0  # Buy Signal
    signals.loc[signals['Short_MA'] <= signals['Long_MA'], 'Signal'] = -1.0  # Sell Signal
    
    return signals

def plot_signals_chart(data, signals):
    fig = go.Figure()

    # Add the stock price
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Stock Price'))

    # Add Buy and Sell signals on the chart
    buy_signals = signals[signals['Signal'] == 1.0]
    sell_signals = signals[signals['Signal'] == -1.0]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=data.loc[buy_signals.index, 'Close'], mode='markers', 
                             marker=dict(color='green', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=data.loc[sell_signals.index, 'Close'], mode='markers', 
                             marker=dict(color='red', size=10), name='Sell Signal'))

    fig.update_layout(title='Stock Price with Buy/Sell Signals',
                      xaxis_title='Date',
                      yaxis_title='Stock Price')
    
    return fig

def calculate_signal_percentage(signals):
    buy_count = len(signals[signals['Signal'] == 1.0])
    sell_count = len(signals[signals['Signal'] == -1.0])
    total_signals = len(signals)
    
    buy_percentage = (buy_count / total_signals) * 100
    sell_percentage = (sell_count / total_signals) * 100
    
    return buy_percentage, sell_percentage

def plot_donut_chart(buy_percentage, sell_percentage):
    fig = go.Figure(data=[go.Pie(labels=['Buy', 'Sell'], values=[buy_percentage, sell_percentage],
                                 hole=0.4)])
    fig.update_layout(title='Percentage of Buy and Sell Signals')
    
    return fig

def calculate_volatility(data, window=30):
    data['Volatility'] = data['Close'].pct_change().rolling(window=window, min_periods=1).std() * np.sqrt(252)
    return data

def plot_volatility(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Volatility'], mode='lines', name='Volatility'))
    fig.update_layout(title='Volatility',
                      xaxis_title='Date',
                      yaxis_title='Volatility')
    return fig

def identify_risk_events(data, threshold=0.02):
    data['High_Risk_Event'] = (data['Close'].pct_change().abs() >= threshold).astype(int)
    return data

def plot_risk_events(data):
    fig = go.Figure(data=[go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Stock Price')])
    fig.add_trace(go.Scatter(x=data[data['High_Risk_Event'] == 1].index,
                             y=data[data['High_Risk_Event'] == 1]['Close'],
                             mode='markers',
                             marker=dict(color='red', size=8),
                             name='High Risk Event'))
    fig.update_layout(title='Stock Price with High-Risk Events',
                      xaxis_title='Date',
                      yaxis_title='Stock Price')
    return fig

def main():
    st.title('Stock Market Explorer - DataStock 2.0')

# Data Source - Yahoo Finance
    st.sidebar.title('Data Source - Yahoo Finance')
    st.sidebar.markdown("The stock data is sourced from Yahoo Finance.")
    st.sidebar.markdown("[Explore Yahoo Finance](https://finance.yahoo.com/)")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Authentication
    st.sidebar.title('User Authentication')

    # Check if the user is logged in or not
    logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None

    if st.session_state.user:
        logged_in = True
        if st.sidebar.button('Logout'):
            st.session_state.user = None
            logged_in = False

    # Check if the user is logged in or not
    logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None

    if st.session_state.user:
        logged_in = True
        st.sidebar.write(f"Logged in as {st.session_state.user}")
    else:
        login_username = st.sidebar.text_input('Username')
        login_password = st.sidebar.text_input('Password', type='password')
        if st.sidebar.button('Login'):
            if verify_user(login_username, login_password):
                st.session_state.user = login_username
                logged_in = True
            else:
                st.sidebar.error("Invalid username or password")

    if not logged_in:
        signup_username = st.sidebar.text_input('Username :')
        signup_email = st.sidebar.text_input('Email')
        signup_password = st.sidebar.text_input('Password :', type='password')
        signup_confirm_password = st.sidebar.text_input('Confirm Password', type='password')
        if st.sidebar.button('Sign Up'):
            if signup_password == signup_confirm_password:
                if create_user(signup_username, signup_email, signup_password):
                    st.sidebar.success("Account created successfully. Please login.")
                else:
                    st.sidebar.error("Error creating the account. Try again later.")
            else:
                st.sidebar.error("Passwords do not match.")

    if not logged_in:
        return
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Sidebar - Stock Symbol and Date Range Selection
    st.sidebar.title('Stock Selection')
    symbol = st.sidebar.text_input('Enter stock symbol (e.g., AAPL for Apple)', 'AAPL')
    start_date = st.sidebar.text_input('Start Date (YYYY-MM-DD)', '2020-01-01')
    end_date = st.sidebar.text_input('End Date (YYYY-MM-DD)', '2023-01-01')

    # Fetch Stock Data
    data = fetch_stock_data(symbol, start_date, end_date)

    if data.empty:
        st.error("Error: No data found for the given symbol and date range.")
        return

    st.subheader('Stock Data')
    st.write(data.tail())

    live_price = fetch_live_stock_price(symbol)
    if live_price is not None:
        st.subheader('Live Stock Price')
        st.write(f"The current price of {symbol} is: {live_price:.2f}")

    # Candlestick Chart
    st.subheader('Candlestick Chart')
    candlestick_chart = plot_candlestick_chart(data)
    st.plotly_chart(candlestick_chart)

    # Line Chart - Closing Price
    st.subheader('Line Chart - Closing Price')
    line_chart = plot_line_chart(data, 'Close', 'Closing Price')
    st.plotly_chart(line_chart)

    # Histogram - Daily Price Change
    st.subheader('Histogram - Daily Price Change')
    data['Daily Price Change'] = data['Close'].pct_change()
    histogram = plot_histogram(data, 'Daily Price Change', 'Daily Price Change Distribution')
    st.plotly_chart(histogram)

    # Scatter Plot - High vs. Low
    st.subheader('Scatter Plot - High vs. Low')
    scatter_plot = plot_scatter(data)
    st.plotly_chart(scatter_plot)

    # Heatmap - Correlation Matrix
    st.subheader('Correlation Heatmap')
    heatmap = plot_heatmap(data, 'Correlation Matrix')
    st.pyplot(heatmap)

    # Line Chart - Yearly Profit
    yearly_profit = calculate_yearly_profit(data)
    st.subheader('Yearly Profit Bar Chart')
    yearly_profit_bar_chart = plot_yearly_profit_bar_chart(yearly_profit)
    st.plotly_chart(yearly_profit_bar_chart)

    st.sidebar.title('Alert Price')
    given_price = st.sidebar.number_input('Enter the Alert price', value=data['Close'].iloc[-1])

    # Compare given price with current stock price
    current_price = data['Close'].iloc[-1]
    if live_price > given_price:
        st.sidebar.success(f"The current price ({live_price:.2f}) is above the Alert price ({given_price:.2f}).")
    elif live_price < given_price:
        st.sidebar.error(f"The current price ({live_price:.2f}) is below the Alert price ({given_price:.2f}).")
    else:
        st.sidebar.info(f"The current price ({live_price:.2f}) is equal to the Alert price ({given_price:.2f}).")

    # Moving Average Crossover Strategy
    st.subheader('Moving Average Crossover Strategy')

    # Calculate Buy/Sell signals
    signals = calculate_signals(data)

    # Display Buy/Sell signals chart
    signals_chart = plot_signals_chart(data, signals)
    st.plotly_chart(signals_chart)

    # Add explanation of the strategy
    st.markdown("""
    **Moving Average Crossover Strategy:**
    
    - **Buy Signal:** When the short-term moving average (e.g., 50-day MA) crosses above the long-term moving average (e.g., 200-day MA), it suggests a Buy signal.
    
    - **Sell Signal:** When the short-term moving average crosses below the long-term moving average, it suggests a Sell signal.
    
    You can modify the parameters (e.g., short_window, long_window) of the Moving Average Crossover strategy 
    to see how it affects the signals.
    """)

    # Calculate and display Buy/Sell signal percentages
    buy_percentage, sell_percentage = calculate_signal_percentage(signals)
    st.subheader('Percentage of Buy and Sell Signals')
    donut_chart = plot_donut_chart(buy_percentage, sell_percentage)
    st.plotly_chart(donut_chart)

    # Display information about the given price comparison
    st.sidebar.title('Alert Price Comparison')
    st.sidebar.write("Alert Price:", given_price)
    st.sidebar.write("Current Stock Price:", live_price)

    # Check if there are Buy signals in the data
    if 1.0 in signals['Signal'].values:
        latest_buy_price = data.loc[signals[signals['Signal'] == 1.0].index[-1], 'Close']
        st.sidebar.write(f"Buy Signal Price: {latest_buy_price:.2f}")
    
    # Check if there are Sell signals in the data
    if -1.0 in signals['Signal'].values:
        latest_sell_price = data.loc[signals[signals['Signal'] == -1.0].index[-1], 'Close']
        st.sidebar.write(f"Sell Signal Price: {latest_sell_price:.2f}")

# Calculate Volatility
    st.subheader('Volatility')
    volatility_data = calculate_volatility(data)
    st.plotly_chart(plot_volatility(volatility_data))

    # Identify High-Risk Events
    st.subheader('High-Risk Events')
    risk_events_data = identify_risk_events(data)
    st.plotly_chart(plot_risk_events(risk_events_data))
    
if __name__ == '__main__':
    main()





