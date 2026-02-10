# Market Risk Analysis - Data Pipeline & HMM Regime Detection

## Overview

This project implements a complete data pipeline for US market risk analysis, including:
1. **Data Fetching**: Daily equity prices and US Treasury yields (2006–present)
2. **Portfolio Construction**: Equal-weight, buy-and-hold multivariate portfolio
3. **Regime Detection**: 2-state Gaussian Hidden Markov Model (HMM) for market regime identification

All code follows best practices and is written in English with English comments.

---

## Data Sources & Coverage

**Period**: January 1, 2006 → Present (daily)  
**Market**: US only

### Equities & ETFs
- **Blue-chip stocks**: AAPL, MSFT, GOOGL, NVDA, JPM, BAC, JNJ, PG
- **Diversified holdings**: AMZN, BRK-B, XOM, CVX, GME, ENPH, GLD
- **Corporate credit proxy**: HYG (high-yield bond ETF)

### Fixed Income (FRED)
- US 10-year constant maturity yield (DGS10)
- US 2-year constant maturity yield (DGS2)

*Data sourced via Yahoo Finance (yfinance) and Federal Reserve Economic Data (pandas_datareader).*

---

## Scripts Overview

### 1. `src/fetch_market_data.py`
**Purpose**: Download and combine all market data.

**Outputs**:
- `data/raw/equities_adj_close.csv` – Adjusted close prices
- `data/raw/us_yields.csv` – Treasury yields
- `data/market_data_combined.csv` – Merged, forward-filled dataset

**Key Functions**:
- `fetch_equities()` – Yahoo Finance download
- `fetch_us_yields()` – FRED data pull
- `combine_and_fill()` – Align calendars & handle missing values

---

### 2. `src/compute_portfolio.py`
**Purpose**: Construct equal-weight, no-rebalancing portfolio.

**Outputs**:
- `data/portfolio_value_buy_and_hold.csv` – Portfolio value time series
- `data/portfolio_returns_buy_and_hold.csv` – Daily returns
- `data/portfolio_metrics_buy_and_hold.txt` – Summary stats (CAGR, volatility, max drawdown)

**Key Functions**:
- `prepare_price_like_series()` – Normalize prices; convert yields to price proxies
- `build_equal_weight_buy_and_hold()` – Constant-weight portfolio (no rebalancing)
- `compute_metrics()` – Annualized returns, volatility, max drawdown

---

### 3. `src/hmm_regime_detection.py`
**Purpose**: Identify market regimes using 2-state Gaussian HMM.

**Outputs**:
- `data/regime_visualization_sp500.png` – Chart of first equity with regime coloring
  - White background = CALM state
  - Blue background = CRISIS state
- `data/regime_timeseries.csv` – Daily regime labels + prices
- `data/hmm_parameters.txt` – Detailed HMM parameters & transition matrix

**Key Functions**:
- `load_and_prepare_returns()` – Log-return computation
- `standardize_returns()` – Feature scaling for HMM
- `fit_hmm()` – Gaussian HMM with 2 components
- `identify_regimes()` – Classify calm vs crisis by covariance norm
- `visualize_regimes()` – Regime-colored price chart

---

## HMM Results Summary

### State Identification
The model identified **2 market regimes**:

| State | Name | Volatility | Characterization |
|-------|------|-----------|-----------------|
| **State 0** | **CALM** | 7.22 | Normal market conditions; tight covariance; positive drift across assets |
| **State 1** | **CRISIS** | 14.45 | Stressed market; ~2× higher volatility; negative drift on equities; yield spikes |

### Transition Probabilities
```
From State \ To State     0         1
State 0 (Calm)         93.48%    6.52%
State 1 (Crisis)       27.76%   72.24%
```

**Interpretation**:
- Calm markets are *sticky*: 93.5% probability of remaining calm the next day.
- Crisis markets persist longer: 72.2% probability of staying in crisis.
- Recovery from crisis is fast when it occurs: 27.76% daily transition to calm.

### Economic Insights
1. **Equity Risk**: Asset means are highly negative in crisis (–7% to –13% per day vs +0.2% to +2.5% in calm).
2. **Volatility Clustering**: Multivariate covariance doubles in crisis regimes.
3. **Flight to Safety**: Yield volatility increases in crisis (bonds become unrelated to spot moves).
4. **HYG Resilience**: High-yield bond returns show distinct crisis sensitivity vs equity dynamics.

---

## Running the Pipeline

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Execute Scripts (in order)
```bash
# 1. Fetch data (takes ~30–60 seconds depending on internet)
python src/fetch_market_data.py

# 2. Build portfolio
python src/compute_portfolio.py

# 3. Fit HMM and detect regimes
python src/hmm_regime_detection.py
```

All outputs are auto-saved to `data/`.

---

## Data Files & Descriptions

| File | Size | Rows | Columns | Description |
|------|------|------|---------|-------------|
| `market_data_combined.csv` | ~7 MB | 5,247 | 82 | All prices + yields; dates as index |
| `portfolio_value_buy_and_hold.csv` | ~160 KB | 5,247 | 1 | Portfolio NAV time series |
| `portfolio_returns_buy_and_hold.csv` | ~169 KB | 5,246 | 1 | Daily log-returns |
| `regime_timeseries.csv` | ~187 KB | 5,245 | 3 | Dates, prices, regime labels |
| `regime_visualization_sp500.png` | ~221 KB | — | — | Price chart + regime background |
| `hmm_parameters.txt` | ~5.4 KB | — | — | Means, covariances, transition matrix |

---

## Code Quality & Standards

✓ **Type hints** throughout (Python 3.10+)  
✓ **Comprehensive logging** for debugging  
✓ **Modular functions** with clear docstrings  
✓ **Error handling** for missing data  
✓ **English comments & variable names**  
✓ **Efficient pandas/NumPy operations**  

---

## Next Steps for Analysis

1. **Macro indicators**: Add additional features (Fed funds rate, unemployment, inflation)
2. **Multi-regime models**: Extend to 3+ states (e.g., calm, moderate, severe crisis)
3. **Risk metrics**: Compute Value-at-Risk (VaR) and Expected Shortfall by regime
4. **Stress testing**: Simulate portfolio returns under different regime paths
5. **Backtesting**: Develop regime-aware trading or hedging strategies

---

*Project completed: February 10, 2026*  
*Author: Risk Analysis Team*
