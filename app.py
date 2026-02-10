import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-positive {
        color: #04b88f;
        font-weight: bold;
    }
    .prediction-negative {
        color: #ff2b2b;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Stock Analysis Tool")
st.sidebar.markdown("---")

# Input sections
ticker = st.sidebar.text_input("Stock Ticker", "AAPL", placeholder="e.g., AAPL, GOOGL, TSLA").upper()
period = st.sidebar.selectbox(
    "Historical Data Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)
prediction_days = st.sidebar.slider(
    "Prediction Days (7-90)",
    min_value=7,
    max_value=90,
    value=30,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**⚠️ Disclaimer:**
This tool is for educational purposes only and is NOT financial advice.
Past performance does not guarantee future results.
""")

# Functions
@st.cache_data
def fetch_stock_data(ticker, period):
    """Fetch stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            st.error(f"No data found for ticker: {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal']
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Bollinger Bands
    bb_sma = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = bb_sma + (bb_std * 2)
    df['BB_Lower'] = bb_sma - (bb_std * 2)
    df['BB_Middle'] = bb_sma
    
    return df

def predict_prices(df, days):
    """Predict future prices using Linear Regression"""
    # Prepare data
    df_pred = df[['Close']].dropna()
    X = np.arange(len(df_pred)).reshape(-1, 1)
    y = df_pred['Close'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future
    future_X = np.arange(len(df_pred), len(df_pred) + days).reshape(-1, 1)
    predictions = model.predict(future_X)
    
    # Create forecast dataframe
    last_date = df_pred.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predictions
    })
    
    return forecast_df, model

def determine_trend(df):
    """Determine current trend"""
    if len(df) < 50:
        return "Insufficient Data", "#FFA500"
    
    sma_20 = df['SMA_20'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    
    if sma_20 > sma_50:
        return "📈 Uptrend", "#04b88f"
    elif sma_20 < sma_50:
        return "📉 Downtrend", "#ff2b2b"
    else:
        return "➡️ Sideways", "#FFA500"

# Main content
st.title("📊 Stock Analysis & Prediction Tool")

analyze_button = st.button("🔍 Analyze Stock", use_container_width=True, key="analyze")

if analyze_button:
    with st.spinner(f"Analyzing {ticker}..."):
        # Fetch data
        data = fetch_stock_data(ticker, period)
        
        if data is not None:
            # Calculate indicators
            data = calculate_technical_indicators(data)
            
            # Get current info
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            current_price = float(data['Close'].iloc[-1])
            previous_close = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
            price_change = current_price - previous_close
            price_change_pct = (price_change / previous_close) * 100 if previous_close != 0 else 0
            
            # Display Key Metrics
            st.markdown("### 📈 Key Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)" if not pd.isna(price_change) else "N/A")
            
            with col2:
                high_52w = info.get('fiftyTwoWeekHigh', None)
                st.metric("52W High", f"${high_52w:.2f}" if high_52w is not None and not pd.isna(high_52w) else "N/A")
            
            with col3:
                low_52w = info.get('fiftyTwoWeekLow', None)
                st.metric("52W Low", f"${low_52w:.2f}" if low_52w is not None and not pd.isna(low_52w) else "N/A")
            
            with col4:
                trend, trend_color = determine_trend(data)
                st.markdown(f'<div style="color: {trend_color}; font-size: 20px; font-weight: bold;">{trend}</div>', 
                           unsafe_allow_html=True)
                st.markdown("**Trend**")
            
            with col5:
                volume = data['Volume'].iloc[-1]
                if pd.isna(volume) or not np.isfinite(volume):
                    st.metric("Volume", "N/A")
                else:
                    st.metric("Volume", f"{int(volume):,.0f}")
            
            st.markdown("---")
            
            # Price Chart with Technical Indicators
            st.markdown("### 📊 Price & Technical Indicators")
            
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=("Price & Moving Averages", "RSI", "MACD")
            )
            
            # Price candlestick
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color='#04b88f',
                    decreasing_line_color='#ff2b2b'
                ),
                row=1, col=1
            )
            
            # Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='rgba(200,200,200,0.5)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='rgba(200,200,200,0.5)', width=1),
                    fillcolor='rgba(200,200,200,0.2)',
                    fill='tonexty',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Moving Averages
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    name='SMA 50',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # RSI
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", row=2, col=1)
            
            # MACD
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    name='MACD',
                    line=dict(color='blue', width=2)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Signal'],
                    name='Signal',
                    line=dict(color='red', width=2)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color=data['MACD_Histogram'].apply(lambda x: '#04b88f' if x > 0 else '#ff2b2b'),
                    showlegend=False
                ),
                row=3, col=1
            )
            
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            
            fig.update_layout(height=900, hovermode='x unified', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Technical Indicators Summary
            st.markdown("### 🎯 Technical Indicators Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rsi = data['RSI'].iloc[-1]
                if pd.isna(rsi) or not np.isfinite(rsi):
                    st.metric("RSI (14)", "N/A")
                else:
                    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.metric("RSI (14)", f"{rsi:.2f}", rsi_status)
            
            with col2:
                macd = data['MACD'].iloc[-1]
                signal = data['Signal'].iloc[-1]
                if pd.isna(macd) or pd.isna(signal) or not np.isfinite(macd) or not np.isfinite(signal):
                    st.metric("MACD", "N/A")
                else:
                    macd_status = "Bullish" if macd > signal else "Bearish"
                    st.metric("MACD", f"{macd:.4f}", macd_status)
            
            with col3:
                sma_20 = data['SMA_20'].iloc[-1]
                sma_50 = data['SMA_50'].iloc[-1]
                if pd.isna(sma_20) or pd.isna(sma_50) or not np.isfinite(sma_20) or not np.isfinite(sma_50):
                    st.metric("SMA 20/50", "N/A")
                else:
                    golden_cross = "Golden Cross" if sma_20 > sma_50 else "Death Cross"
                    st.metric("SMA 20/50", f"{sma_20:.2f}", golden_cross)
            
            with col4:
                try:
                    bb_position = ((data['Close'].iloc[-1] - data['BB_Lower'].iloc[-1]) / 
                                  (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1])) * 100
                    if pd.isna(bb_position) or not np.isfinite(bb_position):
                        st.metric("BB Position", "N/A")
                    else:
                        st.metric("BB Position", f"{bb_position:.1f}%", "Near Upper" if bb_position > 80 else "Near Lower" if bb_position < 20 else "Middle")
                except (ZeroDivisionError, TypeError, ValueError):
                    st.metric("BB Position", "N/A")
            
            st.markdown("---")
            
            # Price Prediction
            st.markdown(f"### 🔮 Price Prediction ({prediction_days} Days)")
            
            forecast_df, model = predict_prices(data, prediction_days)
            
            current_price = data['Close'].iloc[-1]
            predicted_price = forecast_df['Predicted_Price'].iloc[-1]
            price_diff = predicted_price - current_price
            price_diff_pct = (price_diff / current_price) * 100 if current_price != 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if pd.isna(current_price) or not np.isfinite(current_price):
                    st.metric("Current Price", "N/A")
                else:
                    st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                if pd.isna(predicted_price) or not np.isfinite(predicted_price):
                    st.metric(f"Predicted Price ({prediction_days}d)", "N/A")
                else:
                    st.metric(f"Predicted Price ({prediction_days}d)", f"${predicted_price:.2f}")
            
            with col3:
                if pd.isna(price_diff) or not np.isfinite(price_diff) or pd.isna(price_diff_pct) or not np.isfinite(price_diff_pct):
                    st.metric(f"Price Difference", "N/A")
                elif price_diff >= 0:
                    st.markdown(f'<div class="prediction-positive">Expected: +${price_diff:.2f} ({price_diff_pct:.2f}%)</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-negative">Expected: ${price_diff:.2f} ({price_diff_pct:.2f}%)</div>', 
                               unsafe_allow_html=True)
            
            # Prediction Chart
            fig_pred = go.Figure()
            
            # Historical data
            fig_pred.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Historical Price',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            forecast_dates = pd.concat([
                pd.Series([data.index[-1]]),
                forecast_df['Date']
            ])
            forecast_prices = pd.concat([
                pd.Series([data['Close'].iloc[-1]]),
                forecast_df['Predicted_Price'].reset_index(drop=True)
            ])
            
            fig_pred.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_prices,
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_pred.update_layout(
                title=f"{ticker} Price Prediction - {prediction_days} Days",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.markdown("---")
            
            # Data Export
            st.markdown("### 💾 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_historical = data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal']].to_csv()
                st.download_button(
                    label="📥 Download Historical Data",
                    data=csv_historical,
                    file_name=f"{ticker}_historical_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                csv_forecast = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast Data",
                    data=csv_forecast,
                    file_name=f"{ticker}_forecast_{prediction_days}d.csv",
                    mime="text/csv"
                )
            
            st.markdown("---")
            
            # Interpretation Guide
            with st.expander("📚 How to Interpret These Results"):
                st.markdown("""
                **RSI (Relative Strength Index)**
                - Ranges from 0 to 100
                - > 70: Stock might be overbought (potential to fall)
                - < 30: Stock might be oversold (potential to rise)
                - 30-70: Neutral zone
                
                **MACD (Moving Average Convergence Divergence)**
                - Bullish: MACD > Signal line
                - Bearish: MACD < Signal line
                - Helps identify trend changes
                
                **Moving Averages (SMA 20/50)**
                - Golden Cross: SMA 20 crosses above SMA 50 (bullish)
                - Death Cross: SMA 20 crosses below SMA 50 (bearish)
                - Used to identify trends
                
                **Bollinger Bands**
                - Price touching upper band: Potential resistance
                - Price touching lower band: Potential support
                - Band width: Indicates volatility
                
                **Price Predictions**
                - Based on Linear Regression of historical data
                - Not guaranteed - for educational purposes only
                - Longer predictions are less reliable
                - Always consult financial advisors
                """)

else:
    st.info("👈 Enter a stock ticker in the sidebar and click 'Analyze Stock' to get started!")
    
    st.markdown("""
    ## Welcome to Stock Predictor! 📈
    
    This tool provides:
    - **Real-time stock data** from Yahoo Finance
    - **Technical analysis** with multiple indicators
    - **Price predictions** using machine learning
    - **Interactive charts** and data visualization
    
    ### Try These Stocks:
    - **Tech**: AAPL, GOOGL, MSFT, NVDA, TSLA
    - **Finance**: JPM, BAC, V
    - **ETFs**: SPY (S&P 500), QQQ (NASDAQ), DIA (Dow Jones)
    
    ### ⚠️ Important Disclaimer:
    This tool is for **educational purposes only** and is **NOT financial advice**.
    Always do your own research and consult with financial advisors before investing.
    """)
