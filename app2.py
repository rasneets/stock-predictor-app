import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
# ================================
# Data Functions
# ================================
from datetime import datetime, timedelta

def fetch_stock_data(symbol='BHARTIARTL.NS', start='2005-01-01', end='2025-01-01', extra_days=100):
    # Extend the start date backwards by `extra_days`
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    start_extended = (start_dt - timedelta(days=extra_days*2)).strftime("%Y-%m-%d")  # buffer for weekends/holidays

    df = yf.download(symbol, start=start_extended, end=end)

    close = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()

    df['MA20'] = ta.trend.sma_indicator(close, window=20)
    df['MA50'] = ta.trend.sma_indicator(close, window=50)
    df['RSI'] = ta.momentum.rsi(close, window=14)
    df['Volume_MA20'] = volume.rolling(window=20).mean()
    df['ATR_14'] = ta.volatility.average_true_range(high, low, close, window=14)
    df['ADX_14'] = ta.trend.adx(high, low, close, window=14)
    df['BB_WIDTH'] = ta.volatility.bollinger_wband(close, window=20, window_dev=2)

    df.dropna(inplace=True)

    # Trim to original date range + extra 100 for LSTM window
    df = df[df.index >= start_dt - timedelta(days=extra_days)]
    df = df[df.index <= pd.to_datetime(end)]

    return df

def prepare_data(df):
    features = ['Close', 'MA20', 'MA50', 'RSI','Volume_MA20','ATR_14','ADX_14','BB_WIDTH']
    df = df[features]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    total_len = len(data_scaled)
    test_size = int(total_len * 0.2)
    test_data = data_scaled[:test_size + 100]
    train_data = data_scaled[test_size:]

    return train_data, test_data, scaler

def create_sequences(data, window=100):
    x, y = [], []
    for i in range(window, len(data)):
        x.append(data[i - window:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(60, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(80, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(120))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def load_models(symbols, model_dir='models'):
    models = []
    for symbol in symbols:
        try:
            model = load_model(os.path.join(model_dir, f"{symbol}_model.h5"))
            with open(os.path.join(model_dir, f"{symbol}_scaler.pkl"), 'rb') as f:
                scaler = pickle.load(f)
            models.append({'symbol': symbol, 'model': model, 'scaler': scaler})
        except Exception as e:
            print(f"Error loading model for {symbol}: {e}")
    return models

def get_price_group(price):
    price=float(price)
    if price < 800:
        return 'group_less'
    elif price < 3000:
        return 'group_mid'
    else:
        return 'group_more'

# ================================
# Prediction Functions
# ================================

def predict_tomorrow_price(symbol, trained_models, window=100):
    df = fetch_stock_data(symbol)
    if df.empty or len(df) < 100:
        st.error(f"Not enough data to predict for {symbol}. Need at least 100 days of data.")
    
    else:

        features = ['Close', 'MA20', 'MA50', 'RSI', 'Volume_MA20', 'ATR_14', 'ADX_14', 'BB_WIDTH']
        df = df[features].dropna()
        if len(df) < window: return None

        best_r2 = -np.inf
        best_prediction = None

        for entry in trained_models:
            model = entry['model']
            scaler = entry['scaler']
            try:
                data_scaled = scaler.transform(df)
            except Exception as e:
                print(f"Scaler transform failed for symbol {symbol}: {e}")
                continue

            last_window = data_scaled[-window:]
            last_window = last_window.reshape(1, window, data_scaled.shape[1])
            y_pred_scaled = model.predict(last_window, verbose=0).flatten()[0]

            close_scale = scaler.scale_[0]
            close_min = scaler.min_[0]
            y_pred = y_pred_scaled / close_scale + close_min

            x_recent, y_true = create_sequences(data_scaled, window)
            y_pred_recent = model.predict(x_recent, verbose=0).flatten()

            y_pred_recent = y_pred_recent / close_scale + close_min
            y_true = np.array(y_true) / close_scale + close_min
            r2 = r2_score(y_true, y_pred_recent)

            if r2 > best_r2:
                best_r2 = r2
                best_prediction = y_pred

        return best_prediction

def predict_new_stock_best(symbol, start, end, trained_models, window=100):
    df = fetch_stock_data(symbol, start, end)
    if df.empty: return None, None, None
    features = ['Close', 'MA20', 'MA50', 'RSI', 'Volume_MA20', 'ATR_14', 'ADX_14', 'BB_WIDTH']
    df = df[features].dropna()
    if len(df) < window: return None, None, None

    best_r2 = -np.inf
    best_result = None

    for entry in trained_models:
        model, scaler = entry['model'], entry['scaler']
        try:
            data_scaled = scaler.transform(df)
        except:
            continue

        x_new, y_true = create_sequences(data_scaled, window)
        if len(x_new) == 0: continue

        y_pred_scaled = model.predict(x_new, verbose=0).flatten()

        close_scale = scaler.scale_[0]
        close_min = scaler.min_[0]
        y_pred = y_pred_scaled / close_scale + close_min
        y_true_rescaled = y_true / close_scale + close_min

        r2 = r2_score(y_true_rescaled, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_result = {
                'symbol': entry['symbol'],
                'r2': r2,
                'y_pred': y_pred,
                'y_true': y_true_rescaled,
                'df': df
            }

    if best_result:
        return best_result['y_pred'], best_result['y_true'], best_result['df']
    else:
        return None, None, None

# ================================
# Streamlit App
# ================================

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📊 NSE Stock Predictor")

tab1, tab2 = st.tabs(["📆 Predict Range", "📅 Predict Tomorrow"])

group_less = ['MAXHEALTH.NS', 'IDEA.NS', 'ITC.NS', 'TATAMOTORS.NS']
group_mid = ['ICICIBANK.NS', 'HDFCBANK.NS','SIEMENS.NS', 'HEROMOTOCO.NS']
group_more = [ 'MRF.NS', 'BAJFINANCE.NS', 'POWERINDIA.NS', 'PAGEIND.NS']

all_models = {
    'group_less': load_models(group_less),
    'group_mid': load_models(group_mid),
    'group_more': load_models(group_more),
}

with tab2:
    st.header("🕒 Predict Tomorrow's Price")
    # symbol = st.text_input("Enter Stock Symbol (e.g., HDFCBANK.NS)", value="HDFCBANK.NS")
    symbol = st.text_input("Enter Stock NSE Symbol", value="", placeholder="HDFCBANK")
    symbol=symbol + '.NS'


    if st.button("Predict"):
        df = yf.download(symbol, period="200d")
        if df.empty:
            st.error("Invalid symbol or no data available.")
        else:
            latest_price = float(df['Close'].iloc[-1])
            # st.write(f"Latest price for {symbol}: {latest_price}")
            group_name = get_price_group(latest_price)
            # st.write(f"Price group detected: {group_name}")
            models = all_models.get(group_name)
            
            # st.write(f"Models available for prediction: {[m['symbol'] for m in models]}")
            predicted_price = predict_tomorrow_price(symbol, models)

            group_name = get_price_group(latest_price)
            models = all_models.get(group_name)
            predicted_price = predict_tomorrow_price(symbol, models)
            symbol = symbol.replace(".NS", "")
            print(symbol)  

            if predicted_price:
                st.success(f"📌 Tomorrow's predicted close price for **{symbol}**: ₹{predicted_price:.2f}")
            else:
                st.warning("Prediction failed. Possibly not enough data.")

with tab1:
    st.header("📆 Predict Price Range")
    symbol = st.text_input("Enter Stock NSE Symbol", value="", placeholder='HDFCBANK', key='range_symbol')
    symbol=symbol+'.NS'
    col1, col2 = st.columns(2)
    with col1:
        
        start = st.date_input("Start Date (Default : 2024-01-01)", value=pd.to_datetime("2024-01-01"))

    with col2:
        
        end = st.date_input("End Date (Default : 2025-01-01)", value=pd.to_datetime("2025-01-01"))

    if st.button("Predict & Show Graph"):
        df_check = yf.download(symbol, period="10d")
        if df_check.empty:
            st.error("Invalid symbol or no data available.")
        else:
            latest_price = float(df_check['Close'].iloc[-1])
          #  st.write("Latest price type:", type(latest_price))
          #  st.write("Latest price value:", latest_price)

            group_name = get_price_group(latest_price)
            models = all_models.get(group_name)

            y_pred, y_true, df = predict_new_stock_best(symbol, str(start), str(end), trained_models=models)
            if y_pred is not None:
                symbol = symbol.replace(".NS", "")
                print(symbol)  
                st.success(f"✅ Prediction completed.")
                


                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_true, name='Actual', line=dict(color='royalblue',width=2)))
                fig.add_trace(go.Scatter(y=y_pred, name='Predicted', line=dict(color='orange',width=2)))

                fig.update_layout(
                    title=f"{symbol} - Actual vs Predicted",
                    height=500,
                    autosize=True,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="#0d1b2a",      # Optional: match dark theme
                    paper_bgcolor="#0d1b2a",
                    font=dict(color='white'),
                    xaxis_title="Trading Days",          # Add X axis label here
                    yaxis_title="Price (INR)"
                )

                st.plotly_chart(fig, use_container_width=True, config={
                    'staticPlot': True,
                    'displayModeBar': False
                })

            else:
                st.warning("Prediction failed. Check data availability or model mismatch.")


st.markdown(
    """
    <style>
    .caution {
    
        text-align: center;
        color:  red;
        padding: 10px;
        font-size: 14px;
        margin-top: 60px;
    }
    
    .footer {
       text-align: center;
        color: gray;
        
        font-size: 14px;
        
    }
    </style>
    <div class="caution">
        This application is for project/demo purposes only. Not recommended for real trading decisions. We are not responsible for any loss/profit incurred.
    </div>
    <div class="footer">
        Developed by <strong>Rasneet Singh</strong> 

    </div>
    """,
    unsafe_allow_html=True
)
