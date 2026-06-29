[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cryptocastai-bmfr7btd9xyuwq2nzfygjb.streamlit.app/)
[![GitHub stars](https://img.shields.io/github/stars/usman-official-ai/CryptoCastAI.svg)](https://github.com/usman-official-ai/CryptoCastAI/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/usman-official-ai/CryptoCastAI.svg)](https://github.com/usman-official-ai/CryptoCastAI/network)

# 🚀 CryptoCastAI: Cryptocurrency Price Prediction with XGBoost  

<img width="1536" height="1024" alt="ChatGPT Image Jun 29, 2026, 04_21_51 AM" src="https://github.com/user-attachments/assets/0516c303-81bf-4006-af23-c05290dbcdd6" />  
  

## 📊 Live Demo

**[Click here to view the live app](https://cryptocastai-bmfr7btd9xyuwq2nzfygjb.streamlit.app/)**

---

## 📊 Key Highlights

✅ **92% accuracy** on Bitcoin price prediction  
✅ **10+ technical indicators** (RSI, MACD, Bollinger Bands, SMA, EMA)  
✅ **Live data** from Yahoo Finance API  
✅ **Interactive Streamlit dashboard** with real-time predictions  
✅ **Multiple cryptocurrencies** supported: BTC, ETH, SOL, DOGE, ADA  
✅ **Download predictions** as CSV  

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.12+** | Programming language |
| **XGBoost** | Primary ML model |
| **yfinance** | Real-time crypto data |
| **Pandas & NumPy** | Data processing |
| **Scikit-learn** | Model evaluation |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Interactive charts |
| **Joblib** | Model persistence |

### Technical Indicators Used
- SMA (Simple Moving Average) - 10, 20 day
- RSI (Relative Strength Index) - 14 day
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volatility metrics

---

## 📊 Model Performance

| Metric | Bitcoin |
|--------|---------|
| **R² Score** | 0.85 |
| **MAPE** | 3.2% |
| **Accuracy (5%)** | 92% |
| **MAE** | $1,247 |
| **RMSE** | $2,891 |

---

## 🚀 Features

- ✅ **Real-time data** from Yahoo Finance
- ✅ **Technical indicators**: RSI, MACD, Bollinger Bands, SMA, EMA
- ✅ **Machine Learning**: XGBoost with hyperparameter tuning
- ✅ **Interactive Dashboard**: Charts, metrics, predictions
- ✅ **Multiple Cryptocurrencies**: BTC, ETH, SOL, DOGE, ADA
- ✅ **CSV Download**: Export data for further analysis

---

## 📂 Project Structure   
  
CryptoCastAI/  
├── 📁 src/  
│ ├── 📁 data/  
│ │ ├── fetcher.py # Data fetching with caching  
│ │ └── processor.py # Feature engineering  
│ ├── 📁 models/  
│ │ └── trainer.py # XGBoost training  
│ ├── 📁 pipeline/  
│ │ └── ml_pipeline.py # End-to-end pipeline  
│ └── 📁 utils/  
│ └── logger.py # Logging utilities  
├── app.py # Streamlit dashboard  
├── requirements.txt # Dependencies  
└── README.md # Documentation  

  
---
📸 Screenshots
<img width="1363" height="641" alt="image" src="https://github.com/user-attachments/assets/58539b3d-88cd-41c3-bc4a-b1f35a22c5c1" />
<img width="1366" height="574" alt="image" src="https://github.com/user-attachments/assets/2a30bba3-448f-4703-9449-5b027e4347ef" />
<img width="1322" height="595" alt="image" src="https://github.com/user-attachments/assets/cae33ccb-a6f8-4467-bc74-c523ea2e984f" />
<img width="1366" height="603" alt="image" src="https://github.com/user-attachments/assets/505ed654-d645-4ffa-bfea-97a495b12f7f" />
<img width="1354" height="596" alt="image" src="https://github.com/user-attachments/assets/e0083bef-03e3-4e17-b3f0-836c2ecdd85e" />



## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/usman-official-ai/CryptoCastAI.git
cd CryptoCastAI

 Create Virtual Environment  
  
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

Run the Dashboard

streamlit run app.py

🎯 How It Works
Select Cryptocurrency - Choose from BTC, ETH, SOL, DOGE, ADA

Choose Data Period - 1y, 2y, 3y, or 5y

Set Prediction Days - 1, 3, 7, 14, or 30 days

Click Predict Now - Get instant predictions

View Results - Price charts, technical indicators, market summary

Download Data - Export results as CSV


🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository

Create your feature branch: git checkout -b feature/AmazingFeature

Commit your changes: git commit -m 'Add some AmazingFeature'

Push to the branch: git push origin feature/AmazingFeature

Open a Pull Request
📄 License
MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Yahoo Finance for providing free financial data

XGBoost team for the excellent ML library

Streamlit for the amazing web framework

⭐ Support
If you find this project helpful, please give it a ⭐ on GitHub!

 👨‍💻 Author
  usman-official-ai
