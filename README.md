# Volatility Forecasting Dashboard

## Overview
The **Volatility Forecasting Dashboard** is an interactive web application designed to analyze and predict market volatility for a given stock ticker. It integrates stock market data, economic indicators, SEC filings, and news sentiment analysis to provide a comprehensive view of market conditions.

## Features
- **Stock Data Analysis**: Retrieves historical stock prices and key financial metrics.
- **Macroeconomic Indicators**: Incorporates Federal Reserve Economic Data (FRED) for economic insights.
- **SEC Filings Monitoring**: Extracts and processes regulatory filings.
- **News Sentiment Analysis**: Analyzes financial news sentiment.
- **Volatility Prediction Models**: Uses machine learning models (LSTM, GARCH, Random Forest) for forecasting.

## Installation
### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/jdelboccio/Volatility_Forecast.git
   cd Volatility_Forecast
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up API keys in a `.env` file:
   ```sh
   FMP_API_KEY=your_fmp_api_key
   FRED_API_KEY=your_fred_api_key
   NEWS_API_KEY=your_news_api_key
   SEC_API_KEY=your_sec_api_key
   ```
4. Run the Streamlit application:
   ```sh
   streamlit run src/app.py
   ```

## Usage
1. Enter a stock ticker (e.g., AAPL).
2. View historical price trends and volatility metrics.
3. Analyze macroeconomic data.
4. Check for SEC filings.
5. Examine news sentiment analysis.
6. Forecast future volatility using machine learning models.

## Data Sources
- **Financial Modeling Prep (FMP)**: Stock prices, financial statements, SEC filings.
- **Federal Reserve Economic Data (FRED)**: Macroeconomic indicators.
- **SEC EDGAR Database**: Regulatory filings.
- **News API**: Financial news sentiment analysis.

## Repository Structure
```
Volatility_Forecast/
│── data/                  # Processed datasets
│── notebooks/             # Jupyter notebooks for analysis
│── src/
│   ├── api/               # API handlers
│   ├── models/            # Machine learning models
│   ├── preprocess.py      # Data preprocessing
│   ├── app.py             # Streamlit application
│── README.md
│── requirements.txt
```

## Contact
Juan A. Del Boccio  
Email: juandelboccio@gmail.com
GitHub: [https://github.com/jdelboccio/Volatility_Forecast](https://github.com/jdelboccio/Volatility_Forecast)
