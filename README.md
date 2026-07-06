# 📈 Stock Price Prediction Using LSTM & Streamlit

## 🚀 Overview

This project is a machine learning-based stock price prediction system built using **Long Short-Term Memory (LSTM)** neural networks. The model is trained on historical stock data of **Apple Inc.** and deployed using **Streamlit**, providing an interactive web dashboard for analysis and forecasting.

The system can also be be extended to other financial time-series datasets with minimal changes.

---

## ✨ Features

- 📊 Interactive stock price visualization dashboard
- 📉 OHLC candlestick chart for technical analysis
- ⚡ Clean and responsive Streamlit UI
- 🔮 Next-day stock price prediction using LSTM
- 📈 Historical trend analysis and statistical insights

### Dashboard Preview

![image alt](https://github.com/SaumyaMalhotra/Stock_price-prediction/blob/c5c26db814e4c167f2f1094184e35cb5bdb7317f/dashboard.png)

---

## 🧰 Tech Stack

- Python 🐍
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

## 📁 Project Structure

```text
stock-lstm-app/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   ├── stock_lstm_model.keras
│   └── scaler.pkl
├── data/
│   └── AAPL.csv
└── notebooks/
    └── model_training.ipynb
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock-lstm-app.git
cd stock-lstm-app
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

### Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

## 📊 Dataset

Historical stock price data of **Apple Inc.**, sourced from **Kaggle**.

### Features

- Date
- Open
- High
- Low
- Close
- Volume

---

## 🧠 Model Architecture

- LSTM layers for capturing temporal dependencies
- Dropout layers to reduce overfitting
- Dense output layer for regression
- MinMaxScaler for feature normalization

---

## 🔄 Workflow

1. Data collection
2. Data preprocessing
3. Feature scaling using MinMaxScaler
4. Time-series sequence creation
5. LSTM model training
6. Model evaluation
7. Future stock price prediction
8. Streamlit deployment

---

## 👨‍💻 Author

**Saumya Malhotra**
