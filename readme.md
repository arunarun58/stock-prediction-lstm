````markdown
# Stock Price Prediction with LSTM

A deep learning-based stock price prediction system using LSTM neural networks and technical indicators.

## 📈 Overview

This project implements a sophisticated stock price prediction model that uses:

- **LSTM Neural Networks** for time series forecasting
- **Technical Indicators** (RSI, MACD, Moving Averages, etc.)
- **Historical OHLCV data** from Yahoo Finance
- **Early Stopping** and **Gradient Clipping** for stable training

The model achieves excellent performance with Mean Absolute Percentage Error (MAPE) typically around 1.5-2%.

## 🚀 Features

- **Multi-feature Analysis**: Uses 12 different technical indicators
- **Robust Preprocessing**: Handles missing data and outliers gracefully
- **Model Validation**: Comprehensive train/validation/test split
- **Real-time Prediction**: Predicts next day's closing price
- **Performance Metrics**: RMSE, MAE, and MAPE evaluation
- **Visualization**: Comparative plots of actual vs predicted prices

## 📊 Technical Indicators Used

- Open, High, Low, Close, Volume
- Daily Returns
- High-Low Spread
- Moving Averages (5, 10, 20 days)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install torch yfinance pandas numpy scikit-learn matplotlib
```
````

## 💻 Usage

### Basic Usage

1. Clone the repository:

```bash
git clone https://github.com/arunarun58/stock-prediction-lstm.git
cd stock-prediction-lstm
```

2. Run the main script:

```bash
python main.py
```

3. Follow the interactive prompts:

```
Enter stock ticker (e.g., AAPL, TSLA, MSFT): MSFT
Enter number of years of data (max 10): 5
```

### Example Output

```
Fetching data for MSFT from 2019-03-27 to 2024-03-27...
Training samples: 1717
Validation samples: 245
Test samples: 491

Epoch 135/200, Train Loss: 0.000188, Val Loss: 0.000142
Early stopping triggered!

Current closing price: 511.46 USD
Predicted closing price for tomorrow (MSFT): 504.63 USD
Predicted change: -6.83 USD (-1.33%)

Model Performance on Test Set:
RMSE: 8.01
MAE: 6.33
MAPE: 1.50%
```

## 🏗️ Model Architecture

```python
StockLSTM(
  (lstm): LSTM(12, 128, num_layers=2, batch_first=True, dropout=0.3)
  (fc): Linear(in_features=128, out_features=1)
  (dropout): Dropout(p=0.3, inplace=False)
)
```

### Training Parameters

- **Sequence Length**: 60 days
- **Hidden Size**: 128 units
- **LSTM Layers**: 2
- **Dropout**: 0.3
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Early Stopping Patience**: 15 epochs

## 📈 Performance Metrics

The model typically achieves:

- **MAPE**: 1.5-2.0% (Mean Absolute Percentage Error)
- **RMSE**: 1.5-2.0% of stock price
- **Training Time**: 2-5 minutes (depending on data size)

## 🎯 Supported Stocks

Any stock available on Yahoo Finance:

- **US Stocks**: AAPL, MSFT, GOOGL, TSLA, etc.
- **International Stocks**: Using appropriate tickers
- **ETFs and Indices**

## ⚠️ Limitations & Disclaimers

### Technical Limitations

- Maximum 10 years of historical data
- Daily frequency only (no intraday prediction)
- Market closures and holidays are not specifically handled

### Financial Disclaimers

> **Important**: This tool is for educational and research purposes only. Past performance is not indicative of future results. Never invest based solely on algorithmic predictions. Always consult with qualified financial advisors before making investment decisions.

## 🔧 Customization

### Adding New Features

```python
# Add to feature_cols list
feature_cols = ["Open", "High", "Low", "Close", "Volume", "Return",
                "HL_Spread", "MA5", "MA10", "MA20", "RSI", "MACD",
                "Your_New_Feature"]
```

### Modifying Model Parameters

```python
model = StockLSTM(
    input_size=len(feature_cols),
    hidden_size=256,  # Increase hidden units
    num_layers=3,     # Add more layers
    dropout=0.2       # Adjust dropout
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for providing free financial data
- PyTorch team for the excellent deep learning framework
- scikit-learn for preprocessing utilities

## 📞 Support

If you have any questions or issues, please open an issue on GitHub.

---

**Disclaimer**: This project is for educational purposes only. Use at your own risk.

```

This README.md provides:

1. **Clear overview** of the project
2. **Easy installation** instructions
3. **Simple usage** examples
4. **Technical details** for developers
5. **Important disclaimers**
6. **Customization options**
7. **Professional structure** suitable for GitHub

The README is comprehensive yet concise, making it easy for users to understand and use your project while covering all essential information.
```
