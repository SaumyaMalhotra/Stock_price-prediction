import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Theme and Typography Injections
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Shift layout upwards */
    .block-container {
        padding-top: 1.5rem !important;
    }
    
    /* Metrics panel customization */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar header styling */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Glowing card styled containers */
    .card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 0.75rem;
    }
    
    /* Prediction button style */
    div.stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.8rem !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5) !important;
        transform: translateY(-1px);
    }
    div.stButton > button:active {
        transform: translateY(1px);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Loading with Caching
# ---------------------------------------------------------
@st.cache_resource
def load_prediction_assets():
    """Loads the pre-trained Keras model and the MinMaxScaler."""
    model_path = "stock_lstm_model.keras"
    scaler_path = "scaler.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure it is in the root directory.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file '{scaler_path}' not found. Please ensure it is in the root directory.")
        
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Load assets and handle potential errors gracefully
try:
    model, scaler = load_prediction_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Error loading model assets: {str(e)}")
    assets_loaded = False

# ---------------------------------------------------------
# Data Processing Helpers
# ---------------------------------------------------------
@st.cache_data
def process_data(file_source):
    """Reads stock CSV, parses dates, and sorts chronologically."""
    df = pd.read_csv(file_source)
    
    # Sort columns by name for case-insensitive check or map them
    # Yahoo Finance column names: Date, Open, High, Low, Close, Adj Close, Volume
    col_mapping = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_mapping)
    
    # Parse Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    return df

def validate_columns(df):
    """Validates that all required columns are present in the DataFrame."""
    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required if col not in df.columns]
    return missing

# ---------------------------------------------------------
# App Layout & Title
# ---------------------------------------------------------
st.markdown("## Stock Price Prediction")
st.markdown("##### Predict the next day's closing price using a trained  neural network.")
st.markdown("---")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-header">Prediction Settings</div>', unsafe_allow_html=True)

# Timestep configuration
timestep = st.sidebar.selectbox(
    "Sequence Length (Timesteps)",
    options=[30, 60, 100],
    index=1,
    help="Number of prior days the LSTM uses to predict the next day."
)

st.sidebar.markdown('<div class="sidebar-header">Data Source</div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader(
    "Upload stock historical CSV data",
    type=["csv"],
    help="CSV must contain Date, Open, High, Low, Close, Volume"
)

# Run Prediction trigger button
predict_triggered = st.sidebar.button("Predict Next Day Close")

# Chart display settings in sidebar
st.sidebar.markdown('<div class="sidebar-header">Chart Settings</div>', unsafe_allow_html=True)
display_days = st.sidebar.slider(
    "Days to display in charts",
    min_value=30,
    max_value=365,
    value=90,
    help="Number of historical days to show in the visualizations."
)

# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------
# Choose between uploaded file or default AAPL.csv
default_data_path = os.path.join("data", "AAPL.csv")
data_source = None
is_default = False

if uploaded_file is not None:
    data_source = uploaded_file
elif os.path.exists(default_data_path):
    data_source = default_data_path
    is_default = True
else:
    st.info("Welcome! Please upload a historical stock price CSV file in the sidebar to begin.")
    st.stop()

# Load and validate the dataset
try:
    df = process_data(data_source)
    
    # Default dataset notice removed as requested
except Exception as e:
    st.error(f"Error parsing CSV file: {str(e)}")
    st.stop()

# Verify headers
missing_cols = validate_columns(df)
if missing_cols:
    st.error(f"Missing required columns: The dataset is missing: `{missing_cols}`. "
             "Please ensure the CSV contains Date, Open, High, Low, Close, and Volume columns.")
    st.stop()

# Verify row count
if len(df) < timestep:
    st.error(f"Insufficient data rows: The uploaded dataset has {len(df)} rows, "
             f"but the selected sequence length (timestep) requires at least {timestep} rows. "
             f"Please choose a smaller sequence length in the sidebar or upload a larger dataset.")
    st.stop()

# ---------------------------------------------------------
# Dashboard Data Overview
# ---------------------------------------------------------
with st.container():
    # Overview cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Records", value=f"{df.shape[0]:,}")
    with col2:
        missing_vals = df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().sum().sum()
        st.metric(label="Missing Values", value=str(missing_vals))
    with col3:
        latest_date = df['Date'].max().strftime('%Y-%m-%d')
        st.metric(label="Latest Date In Data", value=latest_date)
    with col4:
        last_close_val = df['Close'].iloc[-1]
        st.metric(label="Latest Close Price", value=f"${last_close_val:,.2f}")

    # Collapsible details
    with st.expander("View Raw Data & Statistics", expanded=False):
        tab_df, tab_stats = st.tabs(["First 5 Rows & Dimensions", "Summary Statistics"])
        with tab_df:
            st.markdown(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            st.dataframe(df.head(), use_container_width=True)
        with tab_stats:
            st.dataframe(df.describe(), use_container_width=True)

# ---------------------------------------------------------
# Interactive Plotly Charts
# ---------------------------------------------------------
st.markdown("### Market Visualization")
chart_tab1, chart_tab2, chart_tab3 = st.tabs([
    "OHLC Candlestick Chart", 
    "Closing Price Trend", 
    "Trading Volume"
])

# Limit data to show in charts for performance and readability
df_chart = df.tail(display_days).copy()

with chart_tab1:
    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(
        x=df_chart['Date'],
        open=df_chart['Open'],
        high=df_chart['High'],
        low=df_chart['Low'],
        close=df_chart['Close'],
        name='Candlestick'
    ))
    fig_candle.update_layout(
        title=f"OHLC Candlestick (Last {display_days} Trading Days)",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=450,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_candle, use_container_width=True)

with chart_tab2:
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df_chart['Date'], 
        y=df_chart['Close'], 
        mode='lines',
        name='Close',
        line=dict(color='#3b82f6', width=2.5)
    ))
    fig_line.update_layout(
        title=f"Closing Price Trend (Last {display_days} Trading Days)",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_line, use_container_width=True)

with chart_tab3:
    fig_vol = go.Figure()
    # Color bars based on close price move vs open
    colors = ['#10b981' if row['Close'] >= row['Open'] else '#f43f5e' for _, row in df_chart.iterrows()]
    fig_vol.add_trace(go.Bar(
        x=df_chart['Date'],
        y=df_chart['Volume'],
        name='Volume',
        marker_color=colors
    ))
    fig_vol.update_layout(
        title=f"Trading Volume (Last {display_days} Trading Days)",
        yaxis_title="Volume",
        xaxis_title="Date",
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# ---------------------------------------------------------
# Prediction & Inference Section
# ---------------------------------------------------------
if predict_triggered:
    if not assets_loaded:
        st.error("Prediction aborted. Model assets are not loaded.")
        st.stop()
        
    st.markdown("---")
    st.markdown("### Forecast Prediction")
    
    with st.spinner("Processing sequences and generating prediction..."):
        try:
            # 1. Extract feature columns in correct order: Open, High, Low, Close, Volume
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            features = df[feature_cols].copy()
            
            # Forward fill/backward fill missing values if any in prediction window
            if features.isnull().any().any():
                features = features.ffill().bfill()
                
            features_array = features.values
            
            # 2. Scale using scaler.transform (NOT fit_transform)
            scaled_data = scaler.transform(features_array)
            
            # 3. Take last sequence of length 'timestep'
            input_seq = scaled_data[-timestep:]
            
            # 4. Reshape to (1, timestep, 5)
            input_seq_reshaped = input_seq.reshape(1, timestep, 5)
            
            # 5. Predict using model
            pred_scaled = model.predict(input_seq_reshaped)
            
            # 6. Inverse transform Close value (column index 3)
            # Create a dummy array of shape (1, 5) to align with scaler shape
            dummy = np.zeros((1, 5))
            dummy[0, 3] = pred_scaled.flatten()[0]
            
            predicted_close = scaler.inverse_transform(dummy)[0, 3]
            
        except Exception as e:
            st.error(f"Error running prediction pipeline: {str(e)}")
            st.stop()
            
    # Calculate Metrics & Differences
    last_actual_close = df['Close'].iloc[-1]
    diff = predicted_close - last_actual_close
    pct_change = (diff / last_actual_close) * 100
    
    # ----------------- Metrics Panel (Extra Analytics) -----------------
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            label="Latest Closing Price", 
            value=f"${last_actual_close:,.2f}"
        )
    with metric_col2:
        st.metric(
            label="Predicted Next-Day Close", 
            value=f"${predicted_close:,.2f}"
        )
    with metric_col3:
        st.metric(
            label="Difference (Forecast vs. Actual)", 
            value=f"${diff:+,.2f}",
            delta=f"${diff:+.2f}"
        )
    with metric_col4:
        st.metric(
            label="Expected Change", 
            value=f"{pct_change:+.2f}%",
            delta=f"{pct_change:+.2f}%"
        )
        
    # ----------------- Visualizations: History vs Prediction -----------------
    st.markdown("#### Prediction Trajectory Map")
    
    # Determine the next business day date
    last_actual_date = df['Date'].iloc[-1]
    next_trading_date = last_actual_date + pd.offsets.BDay(1)
    
    # Filter historical slice to show connection (last 10 days for focused view)
    plot_history_len = min(20, len(df))
    df_history_slice = df.tail(plot_history_len).copy()
    
    fig_pred = go.Figure()
    
    # 1. Historical line
    fig_pred.add_trace(go.Scatter(
        x=df_history_slice['Date'],
        y=df_history_slice['Close'],
        mode='lines+markers',
        name='Historical Close',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=6, color='#1e3a8a')
    ))
    
    # 2. Last Actual Highlight
    fig_pred.add_trace(go.Scatter(
        x=[last_actual_date],
        y=[last_actual_close],
        mode='markers',
        name='Last Actual Close',
        marker=dict(color='#10b981', size=12, symbol='circle', line=dict(color='white', width=1))
    ))
    
    # 3. Predicted Point
    fig_pred.add_trace(go.Scatter(
        x=[next_trading_date],
        y=[predicted_close],
        mode='markers+text',
        name='Predicted Next Close',
        text=[f"${predicted_close:.2f}"],
        textposition="top center",
        marker=dict(color='#f43f5e', size=14, symbol='star', line=dict(color='white', width=1))
    ))
    
    # 4. Connecting Trajectory Line
    fig_pred.add_trace(go.Scatter(
        x=[last_actual_date, next_trading_date],
        y=[last_actual_close, predicted_close],
        mode='lines',
        name='Prediction Trajectory',
        line=dict(color='#f43f5e', width=2, dash='dash')
    ))
    
    fig_pred.update_layout(
        title=f"Forecast Trajectory (Connecting {last_actual_date.strftime('%Y-%m-%d')} to {next_trading_date.strftime('%Y-%m-%d')})",
        yaxis_title="Stock Price ($)",
        xaxis_title="Trading Date",
        template="plotly_dark",
        hovermode="x unified",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    # Add spacing at the bottom of the page for better visual layout
    st.markdown('<div style="margin-bottom: 3rem;"></div>', unsafe_allow_html=True)
    
else:
    st.markdown("---")
    st.info("Ready for prediction. Click the Predict Next Day Close button in the sidebar to generate the prediction.")