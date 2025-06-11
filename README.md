```markdown
# 📈 Rule-Based Forex Trading Bot

This is a modular, rule-based forex trading system designed to integrate with **MetaTrader 5** for live or historical trading. The bot uses a combination of popular indicators, pattern recognition, and trading logic to place trades, manage them intelligently, and log performance.

---

---

## 🔧 Functional Overview

### 1. **MT5 Integration**
- Pulls real-time data via MetaTrader5 API.
- Handles login, symbol selection, timeframe conversion, and tick streaming.

### 2. **Indicators Module (`indicators.py`)**
- Provides classes/functions for computing:
  - RSI
  - VWAP
  - SMA, EMA, WMA
  - Ichimoku Cloud
  - Fractals
  - Stochastic Oscillator
  - Custom indicator logic

### 3. **Helper Functions (`helper_functions.py`)**
Includes reusable utilities such as:
- SL/TP Calculation (based on ATR or fixed pip distances)
- Early exit logic based on volume, indicator crossovers, or price action
- Trailing Stop mechanisms
- Candlestick type determination (bullish/bearish engulfing, etc.)
- Position opening/closing logic using MT5 trading functions

### 4. **Strategies**
Each strategy file in the root directory encapsulates a unique trading logic:

| Strategy File       |                Description |
|-------------------------|-----------------------|
| `fractals_strategy.py` | Trades breakouts based on fractal highs/lows |
| `SMA_ichimoku_strategy.py` | Combines Ichimoku trend with SMA confirmation |
| `SMA_stochastic_strategy.py` | Trades stochastic oversold/overbought + SMA trend |
| `SMA_VWAP_RSI_strategy.py` | Uses VWAP for institutional levels, RSI for momentum |

---

## 📊 Logging and Debugging

All logs are stored under the `logs/` directory:
- Trade entries and exits
- SL/TP adjustments
- Errors from MT5 API
- Strategy-specific logs

Each strategy can create its own log file for clean separation.

---

## 🚀 How to Run

> Ensure MT5 is installed and the `MetaTrader5` Python package is installed.

```bash
pip install MetaTrader5
````

### ✅ 1. Set Up MetaTrader5

* Launch your MetaTrader5 terminal.
* Make sure you are logged in to a demo or live account.

### ✅ 2. Run Any Strategy

```bash
cd core
python SMA_VWAP_RSI_strategy.py
```

> You can swap this with any other strategy script.

---

## 📌 Customization

You can easily plug in or modify strategies by:

* Writing new logic using helpers in `helper_functions.py`
* Using indicator classes from `indicators.py`
* Creating new strategy scripts and importing the common components

---

## 🧠 Future Ideas (Suggested)

* ✅ Real-time dashboards using **Streamlit** or **Dash**
* ✅ Slack/Telegram alerts on trade actions
* ✅ Logging to a database for long-term analytics
* ✅ Backtesting framework (using historic `.csv` or `MT5` history)

---

## ✍️ Author

Cliff Ogola

Machine Learning and Computer Vision Engineer.

---

## 🛠 Requirements

```bash
pip install -r requirements.txt
```

You might need:

* `MetaTrader5`
* `pandas`
* `numpy`
* `matplotlib`

```

