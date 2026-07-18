# 📊 Hawkes‑Merton Credit Contagion Model

**Hybrid structural‑reduced form framework for systemic credit risk quantification**  
*Live interactive dashboard • Monte Carlo simulation • Any ticker, from 2000 to today*

---

## 🚀 Overview

This project implements a state‑of‑the‑art **Hawkes‑Merton Contagion Model** that combines classic structural credit risk theory with modern self‑exciting processes and regime‑switching mechanics. It is designed to quantify **systemic risk** in corporate portfolios, identify vulnerable firms, and stress‑test portfolios under various economic scenarios.

The model is built in **Python** and comes with a fully interactive **Streamlit** dashboard that lets you:

- Analyse **any listed company** (unlimited tickers)
- Pull live market data from **Yahoo Finance** and **FRED**
- Calibrate unobservable asset values and volatilities using the **KMV** approach
- Simulate **10,000+ Monte Carlo paths** for asset values, defaults, and contagion
- Compute **VaR (99%)**, **CVaR (99%)**, and **Distance‑to‑Default (DD)** for every firm
- Explore **sector‑level risk aggregation** and **correlation heatmaps**
- Visualise **asset paths** and **Hawkes intensity** trajectories under 5 custom scenarios

---

## 🧠 Key Features

| Feature | Description |
|---------|-------------|
| **Merton structural model** | Firm assets modelled as a call option on the firm’s value (Merton, 1974) |
| **KMV calibration** | Iterative extraction of asset value and volatility from equity market data |
| **Hawkes self‑exciting process** | Models default contagion and clustering (Hawkes, 1971) |
| **Jump‑diffusion** | Sudden downward jumps in asset values (Zhou, 2001) |
| **Dynamic default barrier** | Time‑varying, mean‑reverting debt barrier (Collin‑Dufresne & Goldstein, 2001) |
| **Incomplete information** | Noise on the observed default barrier (Duffie & Lando, 2001) |
| **Stochastic recovery** | Recovery rates negatively correlated with default intensity (Andersen & Sidenius, 2004) |
| **Regime‑switching** | Markov chains for normal vs. stress periods |
| **Interactive dashboard** | Streamlit app with real‑time parameter tuning and visualisations |
| **Unlimited tickers** | Analyse any stock from 2000‑01‑01 to today |
| **Alfa Pulse integration** | Proprietary early‑warning risk‑screening module (optional) |

---

## 📚 Academic Foundation

This model is built on decades of peer‑reviewed research. Key references include:

- **Merton, R. C. (1974)** – *On the Pricing of Corporate Debt: The Risk Structure of Interest Rates*
- **Zhou, C. (2001)** – *A Jump‑Diffusion Model for Asset Returns*
- **Duffie, D. & Lando, D. (2001)** – *Term Structures of Credit Spreads with Incomplete Accounting Information*
- **Collin‑Dufresne, P. & Goldstein, R. S. (2001)** – *Do Credit Spreads Reflect Stationary Leverage?*
- **Andersen, L. & Sidenius, J. (2004)** – *Extensions to the Gaussian Copula: Random Recovery and Random Factor Loadings*
- **Hawkes, A. G. (1971)** – *Spectra of Some Self‑Exciting and Mutually Exciting Point Processes*
- **Kealhofer, S., McQuown, J. & Vasicek, O.** – *The KMV Approach to Credit Risk*

For a full list, see the **Resources** section inside the app.

---

## 🛠️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hawkes-merton-contagion.git
cd hawkes-merton-contagion
```
### 2. Install dependencies

```bash
pip install -r requirements.txt
```

requirements.txt should contain:

```bash
streamlit>=1.28.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
plotly>=5.0.0
yfinance>=0.1.70
pandas-datareader>=0.10.0
requests>=2.26.0
tqdm>=4.62.0
```
### 3. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at http://localhost:8501.

### 💻 Usage

Sidebar controls

  1. Portfolio selection: Choose between the predefined Popular S&P 500 (50 stocks) or enter any custom tickers (comma‑separated).
  2. Date range: Defaults to 2000‑01‑01 to today (unlimited historical data).
  3. Simulation parameters: Adjust the number of Monte Carlo paths, jump intensity, contagion strength, recovery rate, and risk‑free rate.
  4. Run Simulation: Click to start the analysis.

Dashboard outputs

  1. Key metrics: VaR (99%), CVaR (99%), Default Probability, Average Loss.
  2. Loss distribution: Histogram with VaR/CVaR markers.
  3. Distance‑to‑Default (DD): Bar chart and histogram for all firms (colour‑coded by risk level).
  4. Correlation heatmap: Visualise asset return correlations.
  5. Top 10 riskiest firms: Table with DD, asset value, debt, and volatility.
  6. Sector analysis: Aggregated DD and asset values by sector (if sector mapping is available).
  7. Interactive asset & intensity paths: Select firms and scenarios to see simulated trajectories of asset values and Hawkes intensity (λ).

Scenarios

The model includes five pre‑defined scenarios that modify the baseline parameters:
  1. Baseline – moderate contagion and jumps
  2. High Contagion – strong contagion
  3. Severe Jumps – large downward jumps
  4. Compound Crisis – both high contagion and severe jumps
  5. Mild Stress – mild stress conditions

### 📂 Project Structure

```bash
hawkes-merton-contagion/
├── src/
│   ├── model.py          # Main HawkesMertonContagion class
│   ├── extractor.py      # MarketDataExtractor50 for Yahoo Finance & FRED
│   └── utils.py          # Monte Carlo runner and helpers
├── streamlit_app.py      # Streamlit dashboard (entry point)
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── LICENSE               # MIT License
```
