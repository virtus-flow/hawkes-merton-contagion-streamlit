import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ---------- IMPORTS ----------
from model import HawkesMertonContagion
from extractor import MarketDataExtractor50
from utils import run_monte_carlo_sequential

# ---------- Page config ----------
st.set_page_config(
    page_title="Hawkes-Merton Credit Contagion Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Hawkes-Merton Credit Contagion Model")
st.markdown("*Hybrid structural-reduced form model for systemic credit risk quantification*")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Simulation Parameters")

ticker_option = st.sidebar.radio(
    "Select portfolio:",
    ["Popular S&P 500 (50 stocks)", "Custom tickers"]
)

if ticker_option == "Popular S&P 500 (50 stocks)":
    tickers = MarketDataExtractor50.get_sp500_tickers(50)
else:
    st.sidebar.markdown("💡 Enter **any number** of tickers (comma-separated).")
    custom_tickers = st.sidebar.text_input(
        "Tickers (e.g., AAPL, MSFT, TSLA):",
        value="AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA, BRK-B, JPM, V, PG"
    )
    tickers = [t.strip() for t in custom_tickers.split(',') if t.strip()]

# Date range – default from 2000
end_date = datetime.today()
default_start = datetime(2000, 1, 1)
start_date = st.sidebar.date_input("Start date:", default_start)
end_date = st.sidebar.date_input("End date:", end_date)

n_sims = st.sidebar.slider(
    "Number of Monte Carlo simulations:",
    min_value=100,
    max_value=50000,
    value=5000,
    step=100
)

st.sidebar.subheader("📐 Model Parameters")
jump_intensity = st.sidebar.slider(
    "Jump intensity:",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01
)
gamma_multiplier = st.sidebar.slider(
    "Gamma multiplier (contagion):",
    min_value=0.0,
    max_value=3.0,
    value=0.3,
    step=0.1
)
recovery_base = st.sidebar.slider(
    "Base recovery rate:",
    min_value=0.3,
    max_value=0.8,
    value=0.65,
    step=0.05
)
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (%):",
    value=4.47,
    step=0.01
) / 100

run_button = st.sidebar.button(
    "🚀 Run Simulation",
    type="primary",
    use_container_width=True
)

# ---------- Scenario definitions ----------
scenario_modifiers = {
    'Baseline': {
        'gamma_multiplier': 1.0,
        'jump_intensity': 0.3,
        'recovery_sensitivity': -0.5,
        'description': 'Baseline (higher contagion and jumps)'
    },
    'High Contagion': {
        'gamma_multiplier': 2.5,
        'jump_intensity': 0.3,
        'recovery_sensitivity': -0.5,
        'description': 'High Contagion'
    },
    'Severe Jumps': {
        'gamma_multiplier': 1.0,
        'jump_intensity': 0.8,
        'jump_mean': -0.25,
        'jump_std': 0.15,
        'recovery_sensitivity': -0.5,
        'description': 'Severe Jumps'
    },
    'Compound Crisis': {
        'gamma_multiplier': 2.5,
        'jump_intensity': 0.8,
        'jump_mean': -0.25,
        'jump_std': 0.15,
        'recovery_sensitivity': -0.8,
        'description': 'Compound Crisis'
    },
    'Mild Stress': {
        'gamma_multiplier': 0.5,
        'jump_intensity': 0.1,
        'jump_mean': -0.05,
        'jump_std': 0.05,
        'recovery_sensitivity': -0.3,
        'description': 'Mild Stress'
    }
}

# ---------- Helper: generate paths ----------
def generate_paths_for_scenario(model_base, mods, tickers, selected_tickers):
    N = model_base.N
    corr = model_base.corr_assets
    params = {
        'n_companies': N,
        'T': model_base.T,
        'dt': model_base.dt,
        'use_heston': model_base.use_heston,
        'use_stochastic_rate': model_base.use_stochastic_rate,
        'barrier_growth_rate': model_base.barrier_growth_rate,
        'barrier_target': model_base.barrier_target,
        'barrier_mean_reversion': model_base.barrier_mean_reversion,
        'jump_intensity': mods.get('jump_intensity', model_base.jump_intensity),
        'jump_mean': mods.get('jump_mean', model_base.jump_mean),
        'jump_std': mods.get('jump_std', model_base.jump_std),
        'recovery_base': model_base.recovery_base,
        'recovery_sensitivity': mods.get('recovery_sensitivity', model_base.recovery_sensitivity),
        'regime_switching': model_base.regime_switching
    }
    model_new = HawkesMertonContagion(**params)
    model_new.V0 = model_base.V0.copy()
    model_new.vol = model_base.vol.copy()
    model_new.D0 = model_base.D0.copy()
    model_new.corr_assets = corr.copy()
    gamma_mult = mods.get('gamma_multiplier', 1.0)
    gamma = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                gamma[i, j] = 0.05 * gamma_mult * max(0, corr[i, j])
    model_new.set_contagion_network(gamma)
    model_new._prepare_cholesky()
    
    V, lam, default, r, v = model_new.simulate_single_path(return_paths=True)
    t_axis = np.linspace(0, model_new.T, model_new.steps)
    
    ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
    selected_indices = [ticker_to_idx[t] for t in selected_tickers if t in ticker_to_idx]
    
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Asset paths ($ billions)", "Hawkes intensity (λ)"),
                        vertical_spacing=0.15)
    
    for idx in selected_indices:
        ticker = tickers[idx]
        fig.add_trace(go.Scatter(x=t_axis, y=V[:, idx]/1e9, mode='lines',
                                 name=f'{ticker} (V)',
                                 line=dict(width=2)),
                      row=1, col=1)
        y = lam[:, idx]
        x = t_axis[~np.isnan(y)]
        y = y[~np.isnan(y)]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                 name=f'{ticker} (λ)',
                                 line=dict(dash='dot')),
                      row=2, col=1)
    
    if selected_indices:
        idx0 = selected_indices[0]
        D0_0 = model_new.D0[idx0]
        barrier_curve = D0_0 * np.exp(model_new.barrier_growth_rate * t_axis) / 1e9
        fig.add_trace(go.Scatter(x=t_axis, y=barrier_curve, mode='lines',
                                 name='Barrier (Debt)',
                                 line=dict(color='black', dash='dash')),
                      row=1, col=1)
    
    max_lambda = np.nanmax(lam[:, selected_indices]) if selected_indices else 0.01
    if max_lambda > 0:
        fig.update_yaxes(range=[0, max_lambda * 1.2], row=2, col=1)
    else:
        fig.update_yaxes(range=[0, 0.05], row=2, col=1)
    
    fig.update_layout(height=600, width=1000, showlegend=True)
    return fig

# ---------- SESSION STATE INIT ----------
if 'sim_done' not in st.session_state:
    st.session_state.sim_done = False
    st.session_state.model = None
    st.session_state.valid_tickers = None
    st.session_state.data = None
    st.session_state.losses = None
    st.session_state.VaR = None
    st.session_state.CVaR = None
    st.session_state.dd = None
    st.session_state.default_prob = None
    st.session_state.avg_loss = None
    st.session_state.max_loss = None
    st.session_state.dd_asc_idx = None

# ---------- RUN SIMULATION ----------
if run_button or st.session_state.sim_done:
    if run_button:
        with st.spinner("Fetching data and running simulation..."):
            try:
                data = MarketDataExtractor50.extract_all(
                    tickers,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                valid_tickers = data['tickers']
                st.success(f"✅ Retrieved {len(valid_tickers)} tickers")
            except Exception as e:
                st.error(f"❌ Error fetching data: {e}")
                st.stop()
            
            model = HawkesMertonContagion(
                n_companies=len(valid_tickers),
                T=1.0,
                dt=1/252,
                jump_intensity=jump_intensity,
                jump_mean=-0.15,
                jump_std=0.10,
                recovery_base=recovery_base,
                recovery_sensitivity=-0.3,
                regime_switching=False
            )
            model.set_correlation_assets(data['correlation_matrix'])
            
            with st.spinner("KMV calibration..."):
                V0_cal, vol_cal = model.calibrate_kmv(
                    equity_values=data['equity_values'],
                    equity_vols=data['equity_vols'],
                    debt_values=data['debt_values'],
                    risk_free_rate=risk_free_rate,
                    time_horizon=1.0
                )
            
            gamma = np.zeros((len(valid_tickers), len(valid_tickers)))
            for i in range(len(valid_tickers)):
                for j in range(len(valid_tickers)):
                    if i != j:
                        gamma[i, j] = 0.05 * gamma_multiplier * max(0, data['correlation_matrix'][i, j])
            model.set_contagion_network(gamma)
            
            exposures = np.ones(len(valid_tickers)) * 1_000_000
            
            with st.spinner(f"Monte Carlo simulation ({n_sims} paths)..."):
                losses, VaR, CVaR = run_monte_carlo_sequential(
                    model, n_sims, exposures, alpha=0.01, show_progress=False
                )
            
            dd = (model.V0 - model.D0) / (model.V0 * model.vol)
            default_prob = (losses > 0).mean() * 100
            avg_loss = losses.mean()
            max_loss = losses.max()
            dd_asc_idx = np.argsort(dd)
            
            # Store in session state
            st.session_state.sim_done = True
            st.session_state.model = model
            st.session_state.valid_tickers = valid_tickers
            st.session_state.data = data
            st.session_state.losses = losses
            st.session_state.VaR = VaR
            st.session_state.CVaR = CVaR
            st.session_state.dd = dd
            st.session_state.default_prob = default_prob
            st.session_state.avg_loss = avg_loss
            st.session_state.max_loss = max_loss
            st.session_state.dd_asc_idx = dd_asc_idx
            st.session_state.dd_sorted_idx = np.argsort(dd)[::-1]
            st.session_state.tickers = valid_tickers
            st.session_state.sector_map = {
                'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'GOOGL': 'Tech', 'AMZN': 'Tech',
                'META': 'Tech', 'BRK-B': 'Financials', 'LLY': 'Healthcare', 'AVGO': 'Tech', 'JPM': 'Financials',
                'V': 'Financials', 'TSLA': 'Consumer', 'XOM': 'Energy', 'UNH': 'Healthcare', 'PG': 'Consumer',
                'MA': 'Financials', 'JNJ': 'Healthcare', 'HD': 'Consumer', 'COST': 'Consumer', 'MRK': 'Healthcare',
                'ABBV': 'Healthcare', 'WMT': 'Consumer', 'BAC': 'Financials', 'CRM': 'Tech', 'CVX': 'Energy',
                'NFLX': 'Tech', 'ADBE': 'Tech', 'KO': 'Consumer', 'PEP': 'Consumer', 'TMO': 'Healthcare',
                'LIN': 'Materials', 'DIS': 'Consumer', 'ORCL': 'Tech', 'CSCO': 'Tech', 'MCD': 'Consumer',
                'ACN': 'Tech', 'IBM': 'Tech', 'ABT': 'Healthcare', 'CAT': 'Industrials', 'GE': 'Industrials',
                'DHR': 'Healthcare', 'VZ': 'Telecom', 'NOW': 'Tech', 'GS': 'Financials', 'PM': 'Consumer',
                'SPGI': 'Financials', 'QCOM': 'Tech', 'RTX': 'Industrials', 'TXN': 'Tech', 'NEE': 'Utilities'
            }

    # ---- Now display results from session state ----
    if st.session_state.sim_done:
        model = st.session_state.model
        valid_tickers = st.session_state.valid_tickers
        data = st.session_state.data
        losses = st.session_state.losses
        VaR = st.session_state.VaR
        CVaR = st.session_state.CVaR
        dd = st.session_state.dd
        default_prob = st.session_state.default_prob
        avg_loss = st.session_state.avg_loss
        max_loss = st.session_state.max_loss
        dd_asc_idx = st.session_state.dd_asc_idx
        dd_sorted_idx = st.session_state.dd_sorted_idx
        sector_map = st.session_state.sector_map

        # ---- Metrics ----
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VaR (99%)", f"${VaR:,.0f}")
        with col2:
            st.metric("CVaR (99%)", f"${CVaR:,.0f}")
        with col3:
            st.metric("Default Prob", f"{default_prob:.2f}%")
        with col4:
            st.metric("Average Loss", f"${avg_loss:,.0f}")

        # ---- Histogram of Losses ----
        st.subheader("📊 Loss Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=losses,
            nbinsx=50,
            name="Losses",
            marker_color='skyblue',
            opacity=0.7
        ))
        fig_hist.add_vline(x=VaR, line_dash="dash", line_color="red",
                           annotation_text=f"VaR 99% = ${VaR:,.0f}")
        fig_hist.add_vline(x=CVaR, line_dash="dash", line_color="darkred",
                           annotation_text=f"CVaR 99% = ${CVaR:,.0f}")
        fig_hist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        # ---- DD Bar Chart ----
        st.subheader("📊 Distance-to-Default (DD) – All firms")
        dd_sorted = dd[dd_sorted_idx]
        tickers_sorted = [valid_tickers[i] for i in dd_sorted_idx]
        colors = ['green' if d > 2.5 else 'orange' if d > 1.5 else 'red' for d in dd_sorted]
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Bar(
            x=tickers_sorted,
            y=dd_sorted,
            marker_color=colors,
            text=[f'{d:.2f}' for d in dd_sorted],
            textposition='outside'
        ))
        fig_dd.add_hline(y=2.5, line_dash="dash", line_color="orange",
                         annotation_text="Medium risk (DD=2.5)")
        fig_dd.add_hline(y=1.5, line_dash="dash", line_color="red",
                         annotation_text="High risk (DD=1.5)")
        fig_dd.update_layout(height=500, xaxis_title="Ticker", yaxis_title="Distance-to-Default", showlegend=False)
        st.plotly_chart(fig_dd, use_container_width=True)

        # ---- DD Histogram ----
        st.subheader("📊 DD Distribution")
        fig_dd_hist = go.Figure()
        fig_dd_hist.add_trace(go.Histogram(x=dd, nbinsx=20, marker_color='lightblue', opacity=0.7))
        fig_dd_hist.add_vline(x=2.5, line_dash="dash", line_color="orange", annotation_text="DD=2.5")
        fig_dd_hist.add_vline(x=1.5, line_dash="dash", line_color="red", annotation_text="DD=1.5")
        fig_dd_hist.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_dd_hist, use_container_width=True)

        # ---- Correlation Heatmap ----
        st.subheader("🔗 Asset Return Correlation Matrix")
        corr_matrix = data['correlation_matrix']
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=valid_tickers,
            y=valid_tickers,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation")
        ))
        fig_corr.update_layout(height=600, width=800)
        st.plotly_chart(fig_corr, use_container_width=True)

        # ---- Top 10 Riskiest ----
        st.subheader("🔥 Top 10 Riskiest Firms")
        top10_data = []
        for i in range(min(10, len(valid_tickers))):
            idx = dd_asc_idx[i]
            top10_data.append({
                'Ticker': valid_tickers[idx],
                'DD': dd[idx],
                'V0 (B$)': model.V0[idx] / 1e9,
                'Debt (B$)': model.D0[idx] / 1e9,
                'Vol (%)': model.vol[idx] * 100
            })
        df_top10 = pd.DataFrame(top10_data)
        st.dataframe(df_top10, use_container_width=True, hide_index=True)

        # ---- Sector Analysis ----
        sectors = [sector_map.get(t, 'Other') for t in valid_tickers]
        if len(set(sectors)) > 1:
            st.subheader("🏭 Sector Analysis")
            sector_df = pd.DataFrame({
                'Ticker': valid_tickers,
                'Sector': sectors,
                'DD': dd,
                'V0': model.V0
            })
            sector_agg = sector_df.groupby('Sector').agg({
                'DD': ['mean', 'min', 'max'],
                'V0': 'sum'
            }).round(3)
            sector_agg.columns = ['DD_avg', 'DD_min', 'DD_max', 'V0_sum']
            sector_agg['V0_sum_B'] = sector_agg['V0_sum'] / 1e9
            st.dataframe(sector_agg, use_container_width=True)
            fig_sector = go.Figure()
            fig_sector.add_trace(go.Bar(
                x=sector_agg.index,
                y=sector_agg['DD_avg'],
                marker_color='lightgreen',
                text=[f'{v:.2f}' for v in sector_agg['DD_avg']],
                textposition='outside'
            ))
            fig_sector.add_hline(y=2.5, line_dash="dash", line_color="orange")
            fig_sector.update_layout(height=400, xaxis_title="Sector", yaxis_title="Avg DD")
            st.plotly_chart(fig_sector, use_container_width=True)

        # ---- Interactive Paths ----
        st.subheader("🎯 Interactive Asset & Intensity Paths")
        selected_for_paths = st.multiselect(
            "Select firms to display:",
            options=valid_tickers,
            default=valid_tickers[:5] if len(valid_tickers) >= 5 else valid_tickers
        )
        scenario_names = list(scenario_modifiers.keys())
        selected_scenario = st.selectbox("Select scenario:", scenario_names, index=0)
        mods = scenario_modifiers[selected_scenario]

        if st.button("🔄 Generate paths for selected firms"):
            if selected_for_paths:
                with st.spinner(f"Generating paths for scenario: {selected_scenario}..."):
                    fig_paths = generate_paths_for_scenario(
                        model, mods, valid_tickers, selected_for_paths
                    )
                    st.plotly_chart(fig_paths, use_container_width=True)
            else:
                st.warning("Please select at least one firm.")

        if st.button("🔴 Generate paths for 5 riskiest firms"):
            top5_risky = [valid_tickers[i] for i in dd_asc_idx[:5]]
            with st.spinner("Generating paths for 5 riskiest firms..."):
                fig_paths_top5 = generate_paths_for_scenario(
                    model, mods, valid_tickers, top5_risky
                )
                st.plotly_chart(fig_paths_top5, use_container_width=True)

        # ---- Statistics ----
        st.subheader("📈 Portfolio Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average DD", f"{np.mean(dd):.3f}")
            st.metric("Min DD", f"{np.min(dd):.3f} ({valid_tickers[np.argmin(dd)]})")
            st.metric("Max DD", f"{np.max(dd):.3f} ({valid_tickers[np.argmax(dd)]})")
        with col2:
            st.metric("Avg V0", f"${np.mean(model.V0)/1e9:.1f}B")
            st.metric("Avg Volatility", f"{np.mean(model.vol)*100:.1f}%")
            st.metric("Firms with DD < 2.5", f"{np.sum(dd < 2.5)}")
            st.metric("Firms with DD < 1.5", f"{np.sum(dd < 1.5)}")

        # ---- Resources ----
        st.subheader("📚 Resources & Literature")
        with st.expander("📖 Key Books and Papers", expanded=False):
            st.markdown("""
            ### 📕 Books
            
            | Book | Author(s) | Year | Description |
            |------|-----------|------|-------------|
            | **Options, Futures, and Other Derivatives** | John C. Hull | 2021 | Foundational text on derivatives and valuation models |
            | **Dynamic Asset Pricing Theory** | Darrell Duffie | 2001 | Mathematical framework for asset pricing and risk |
            | **Credit Risk: Modeling, Valuation and Hedging** | Tomasz R. Bielecki, Marek Rutkowski | 2002 | Advanced credit risk models including structural approach |
            | **Fixed Income Securities** | Bruce Tuckman, Angel Serrat | 2011 | Comprehensive overview of fixed income and credit spreads |
            | **Risk Management and Financial Institutions** | John C. Hull | 2018 | Risk management in financial institutions |
            
            ### 📄 Key Academic Papers
            
            | Paper | Author(s) | Year | Key Contribution |
            |-------|-----------|------|------------------|
            | **On the Pricing of Corporate Debt: The Risk Structure of Interest Rates** | Robert C. Merton | 1974 | Foundational structural model – firm assets as an option |
            | **A Theory of the Term Structure of Interest Rates** | John C. Cox, Jonathan E. Ingersoll, Stephen A. Ross | 1985 | CIR interest rate model |
            | **A Jump-Diffusion Model for Asset Returns** | Chunsheng Zhou | 2001 | Jump-diffusion for credit spreads and default correlations |
            | **Do Credit Spreads Reflect Stationary Leverage?** | Pierre Collin-Dufresne, Robert S. Goldstein | 2001 | Dynamic barrier and stationary leverage |
            | **Term Structures of Credit Spreads with Incomplete Accounting Information** | Darrell Duffie, David Lando | 2001 | Incomplete information model for credit spreads |
            | **Extensions to the Gaussian Copula: Random Recovery and Random Factor Loadings** | Leif Andersen, Jakob Sidenius | 2004 | Stochastic recovery in credit risk models |
            | **The KMV Approach to Credit Risk** | Kealhofer, McQuown, Vasicek | 1990s | KMV calibration and Distance-to-Default (DD) |
            | **Hawkes Processes and Their Applications in Finance** | Emmanuel Bacry, Iacopo Mastromatteo, Jean-François Muzy | 2015 | Review of Hawkes processes in finance |
            | **Credit Default Swaps and the Credit Crisis** | David X. Li | 2006 | CDS and correlation models (Gaussian copula) |
            
            ### 🌐 Data Sources
            
            | Source | Link | Description |
            |--------|------|-------------|
            | **FRED (Federal Reserve Economic Data)** | [fred.stlouisfed.org](https://fred.stlouisfed.org) | Treasury yields, BAA corporate yields, macro data |
            | **Yahoo Finance** | [finance.yahoo.com](https://finance.yahoo.com) | Stock prices, market cap, volatility |
            | **Financial Modeling Prep** | [financialmodelingprep.com](https://financialmodelingprep.com) | Balance sheet and financial statement data (API) |
            | **SEC EDGAR** | [sec.gov/edgar](https://sec.gov/edgar) | 10-K and 10-Q reports |
            | **Moody's Default Rates** | [moodys.com](https://www.moodys.com) | Historical default rates by rating |
            | **S&P Global Default Studies** | [spglobal.com](https://www.spglobal.com) | Cumulative default and recovery rates |
            
            ### 💡 Conceptual Contributions
            
            | Concept | Description | Key Authors |
            |---------|-------------|-------------|
            | **KMV Approach** | Extracting unobservable asset value and volatility from market data | Kealhofer, McQuown, Vasicek |
            | **Distance-to-Default (DD)** | Measure of distance from assets to default barrier | Merton, KMV |
            | **Hawkes Process** | Self-exciting process for modeling default clustering | Hawkes (1971), Bacry et al. (2015) |
            | **Jump-Diffusion** | Combination of continuous GBM and discrete jumps | Zhou (2001) |
            | **Regime-Switching** | Markov chains for switching between normal and stress regimes | Hamilton (1989), Giesecke |
            | **Stochastic Recovery** | Recovery rate negatively correlated with default intensity | Andersen & Sidenius (2004) |
            | **Incomplete Information** | Noise on barrier observation due to accounting lags | Duffie & Lando (2001) |
            """)

        with st.sidebar.expander("📚 Literature", expanded=False):
            st.markdown("""
            - **Merton (1974)** – Structural model
            - **Zhou (2001)** – Jump-Diffusion
            - **Duffie & Lando (2001)** – Incomplete information
            - **Collin-Dufresne & Goldstein (2001)** – Dynamic barrier
            - **Andersen & Sidenius (2004)** – Stochastic recovery
            - **Hull (2021)** – Derivatives and risk management
            - **FRED / Yahoo Finance** – Data
            """)

else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Simulation'")
    st.markdown("""
    ### 📌 What does this model do?
    
    **Hawkes-Merton Contagion Model** is a hybrid framework for systemic credit risk quantification.
    
    - **Merton structural model** – firm asset valuation and Distance-to-Default (DD)
    - **Hawkes process** – modelling contagion between firms
    - **Jump-Diffusion** – sudden jumps in asset values
    - **KMV calibration** – extracting unobservable asset value from market data
    - **VaR / CVaR** – tail risk measures
    - **Interactive paths** – asset and intensity trajectories
    - **Sector analysis** – risk aggregation by sector
    - **Unlimited tickers** – analyse any listed company, from 2000 to today
    """)
