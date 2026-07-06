📈 Stock Price Prediction Using LSTM & Streamlit

🚀 Overview

This project is a machine learning-based stock price prediction system built using Long Short-Term Memory (LSTM) neural networks. The model is trained on historical stock data of Apple Inc. and deployed using Streamlit, providing an interactive and user-friendly web dashboard for analysis and forecasting.

The system can also be extended to other financial time-series datasets with minimal changes.

✨ Features
📊 Interactive stock price visualization dashboard
📉 OHLC candlestick chart for detailed technical analysis
⚡ Clean and responsive Streamlit UI
🔮 Next-day stock price prediction using LSTM model
📈 Historical trend analysis and statistical insights

![image alt](https://github.com/SaumyaMalhotra/Stock_price-prediction/blob/c5c26db814e4c167f2f1094184e35cb5bdb7317f/dashboard.png)

🧰 Tech Stack
Python 🐍
TensorFlow / Keras
Pandas
NumPy
Scikit-learn
Streamlit
Matplotlib

📁 Project Structure
stock-lstm-app/
│
├── app.py<br><br>
├── requirements.txt<br><br>
├── README.md<br><br>
│
├── model/<br><br>
│   ├── stock_lstm_model.keras<br><br>
│   └── scaler.pkl
│
├── data/
│   └── AAPL.csv
│
└── notebooks/
    └── model_training.ipynb
    
⚙️ Installation
1. Clone the Repository
git clone https://github.com/your-username/stock-lstm-app.git
cd stock-lstm-app
2. Create Virtual Environment (Optional but Recommended)
python -m venv venv

Activate environment:

Windows:
venv\Scripts\activate

macOS / Linux:
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt
▶️ Run the Application

Start the Streamlit app using:

streamlit run app.py

Then open the URL shown in the terminal (usually http://localhost:8501).

📊 Dataset Information

The dataset contains historical stock price data of Apple Inc., sourced from Kaggle.

Key Columns:
Date
Open
High
Low
Close
Volume

🧠 Model Architecture

The prediction system uses an LSTM-based deep learning model designed for sequential forecasting:

LSTM layers → capture temporal patterns
Dropout layers → reduce overfitting
Dense output layer → regression output
MinMaxScaler → feature normalization

🔄 Workflow
Data collection & preprocessing
Feature scaling using MinMaxScaler
Sequence creation for time-series input
LSTM model training
Model evaluation & prediction
Deployment using Streamlit

Author
Saumya Malhotra
