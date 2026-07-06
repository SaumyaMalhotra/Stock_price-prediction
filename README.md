Stock Price Prediction Using LSTM and Streamlit
Overview

This project presents a machine learning-based stock price prediction system implemented using Long Short-Term Memory (LSTM) neural networks. The application is deployed using Streamlit to provide an interactive interface for data visualization and prediction.

The model is trained on historical stock market data (Apple Inc.) and can be adapted for other financial time series datasets.

## Features

- Interactive stock price visualization dashboard
- OHLC candlestick chart for historical analysis
- Real-time styled UI for financial insights
- Model-based next-day stock prediction
- Data statistics and trend analysis

## Dashboard Preview

![Stock Prediction Dashboard](assets/dashboard.png)


Technologies Used
Python
TensorFlow / Keras
Pandas
NumPy
Scikit-learn
Streamlit
Matplotlib
Project Structure
stock-lstm-app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── stock_lstm_model.keras
│   └── scaler.pkl
│
├── data/
│   └── AAPL.csv
│
└── notebooks/
    └── model_training.ipynb

    
Installation Instructions
1. Clone the Repository
git clone https://github.com/your-username/stock-lstm-app.git
cd stock-lstm-app
2. Create a Virtual Environment (Optional)
python -m venv venv
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
Running the Application

To launch the Streamlit application, execute the following command:

streamlit run app.py
Dataset Information

The dataset used in this project consists of historical Apple Inc. stock price data sourced from Kaggle.

The dataset includes the following attributes:

Date
Open
High
Low
Close
Volume
Model Description

The predictive model is built using an LSTM neural network designed for sequential time-series forecasting. The model architecture includes:

LSTM layers for learning temporal dependencies
Dropout layers to reduce overfitting
Dense output layer for regression prediction
MinMaxScaler for feature normalization
Workflow
Data acquisition and preprocessing
Feature scaling using MinMaxScaler
Sequence generation for time-series input
Model training using LSTM architecture
Model evaluation and prediction
Deployment via Streamlit interface
Future Enhancements
Integration of real-time stock market APIs
Extension to multi-stock prediction systems
Implementation of advanced architectures such as GRU or Transformers
Cloud deployment using Streamlit Cloud or AWS
Author

Saumya Malhotra

License

This project is licensed under the MIT License.
