# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Dodaj src folder u path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import HawkesMertonContagion
from src.extractor import MarketDataExtractor50
from src.utils import run_monte_carlo_sequential

# ---------- Konfiguracija stranice ----------
st.set_page_config(
    page_title="Hawkes-Merton Credit Contagion Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Hawkes-Merton Credit Contagion Model")
st.markdown("*Hibridni strukturno-redukovani model za kvantifikaciju sistemskog kreditnog rizika*")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Parametri simulacije")

ticker_option = st.sidebar.radio(
    "Odaberi portfolio:",
    ["Top 50 S&P 500", "Custom tickeri"]
)

if ticker_option == "Top 50 S&P 500":
    tickers = MarketDataExtractor50.get_sp500_tickers(50)
else:
    custom_tickers = st.sidebar.text_input(
        "Unesi tickere (odvoji zarezom):",
        value="AAPL, MSFT, NVDA, GOOGL, AMZN"
    )
    tickers = [t.strip() for t in custom_tickers.split(',')]

end_date = datetime.today()
default_start = end_date - timedelta(days=2*365)
start_date = st.sidebar.date_input("Početni datum:", default_start)
end_date = st.sidebar.date_input("Završni datum:", end_date)

n_sims = st.sidebar.slider(
    "Broj Monte Carlo simulacija:",
    min_value=100,
    max_value=50000,
    value=5000,
    step=100
)

st.sidebar.subheader("📐 Parametri modela")
jump_intensity = st.sidebar.slider(
    "Jump intenzitet:",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01
)
gamma_multiplier = st.sidebar.slider(
    "Gamma multiplikator (zaraza):",
    min_value=0.0,
    max_value=3.0,
    value=0.3,
    step=0.1
)
recovery_base = st.sidebar.slider(
    "Osnovna stopa oporavka:",
    min_value=0.3,
    max_value=0.8,
    value=0.65,
    step=0.05
)
risk_free_rate = st.sidebar.number_input(
    "Bezrizična stopa (%):",
    value=4.47,
    step=0.01
) / 100

run_button = st.sidebar.button(
    "🚀 Pokreni simulaciju",
    type="primary",
    use_container_width=True
)

# ---------- Glavni dio ----------
if run_button:
    with st.spinner("Preuzimanje podataka i pokretanje simulacije..."):
        try:
            data = MarketDataExtractor50.extract_all(
                tickers,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            valid_tickers = data['tickers']
            st.success(f"✅ Preuzeto {len(valid_tickers)} tickera")
        except Exception as e:
            st.error(f"❌ Greška pri preuzimanju podataka: {e}")
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
        
        with st.spinner("KMV kalibracija..."):
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
        
        with st.spinner(f"Monte Carlo simulacija ({n_sims} putanja)..."):
            losses, VaR, CVaR = run_monte_carlo_sequential(
                model, n_sims, exposures, alpha=0.01, show_progress=False
            )
        
        dd = (model.V0 - model.D0) / (model.V0 * model.vol)
        default_prob = (losses > 0).mean() * 100
        avg_loss = losses.mean()
        max_loss = losses.max()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VaR (99%)", f"${VaR:,.0f}")
        with col2:
            st.metric("CVaR (99%)", f"${CVaR:,.0f}")
        with col3:
            st.metric("Default Prob", f"{default_prob:.2f}%")
        with col4:
            st.metric("Prosečan gubitak", f"${avg_loss:,.0f}")
        
        st.subheader("📊 Distribucija gubitaka")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=losses,
            nbinsx=50,
            name="Gubici",
            marker_color='skyblue',
            opacity=0.7
        ))
        fig_hist.add_vline(x=VaR, line_dash="dash", line_color="red",
                           annotation_text=f"VaR 99% = ${VaR:,.0f}")
        fig_hist.add_vline(x=CVaR, line_dash="dash", line_color="darkred",
                           annotation_text=f"CVaR 99% = ${CVaR:,.0f}")
        fig_hist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.subheader("🔥 Top 10 najrizičnijih firmi")
        dd_sorted_idx = np.argsort(dd)
        top10_data = []
        for i in range(10):
            idx = dd_sorted_idx[i]
            top10_data.append({
                'Ticker': valid_tickers[idx],
                'DD': dd[idx],
                'V0 (B$)': model.V0[idx] / 1e9,
                'Dug (B$)': model.D0[idx] / 1e9,
                'Vol (%)': model.vol[idx] * 100
            })
        df_top10 = pd.DataFrame(top10_data)
        st.dataframe(df_top10, use_container_width=True, hide_index=True)
        
        st.subheader("📈 Statistički pregled")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prosečan DD", f"{np.mean(dd):.3f}")
            st.metric("Min DD", f"{np.min(dd):.3f} ({valid_tickers[np.argmin(dd)]})")
            st.metric("Max DD", f"{np.max(dd):.3f} ({valid_tickers[np.argmax(dd)]})")
        with col2:
            st.metric("Prosečna V0", f"${np.mean(model.V0)/1e9:.1f}B")
            st.metric("Prosečna volatilnost", f"{np.mean(model.vol)*100:.1f}%")
            st.metric("Broj firmi sa DD < 2.5", f"{np.sum(dd < 2.5)}")

else:
    st.info("👈 Podesi parametre u bočnom meniju i klikni 'Pokreni simulaciju'")
    st.markdown("""
    ### 📌 Šta ovaj model radi?
    
    **Hawkes-Merton Contagion Model** je hibridni okvir za kvantifikaciju sistemskog kreditnog rizika.
    
    - **Mertonov strukturni model** – vrednovanje imovine i Distance-to-Default (DD)
    - **Hawkes proces** – modeliranje zaraze između firmi
    - **Jump-Diffusion** – nagli skokovi u vrednosti imovine
    - **KMV kalibracija** – izvlačenje nevidljive imovine iz tržišnih podataka
    - **VaR / CVaR** – izračunavanje repnih mjera rizika
    """)
