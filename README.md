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
