import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ---------- IMPORTI DIREKTNO IZ ROOT-A ----------
from model import HawkesMertonContagion
from extractor import MarketDataExtractor50
from utils import run_monte_carlo_sequential

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

# ---------- Definicija scenarija (isti kao u modelu) ----------
scenario_modifiers = {
    'Baseline': {
        'gamma_multiplier': 1.0,
        'jump_intensity': 0.3,
        'recovery_sensitivity': -0.5,
        'description': 'Osnovni model (veća zaraza i skokovi)'
    },
    'High Contagion': {
        'gamma_multiplier': 2.5,
        'jump_intensity': 0.3,
        'recovery_sensitivity': -0.5,
        'description': 'Pojačana zaraza'
    },
    'Severe Jumps': {
        'gamma_multiplier': 1.0,
        'jump_intensity': 0.8,
        'jump_mean': -0.25,
        'jump_std': 0.15,
        'recovery_sensitivity': -0.5,
        'description': 'Snažni skokovi'
    },
    'Compound Crisis': {
        'gamma_multiplier': 2.5,
        'jump_intensity': 0.8,
        'jump_mean': -0.25,
        'jump_std': 0.15,
        'recovery_sensitivity': -0.8,
        'description': 'Kombinovana kriza'
    },
    'Mild Stress': {
        'gamma_multiplier': 0.5,
        'jump_intensity': 0.1,
        'jump_mean': -0.05,
        'jump_std': 0.05,
        'recovery_sensitivity': -0.3,
        'description': 'Blagi stres'
    }
}

# ---------- Funkcija za generisanje putanja za dati scenario ----------
def generate_paths_for_scenario(model_base, mods, tickers, selected_tickers):
    """Generiše putanje za odabrane tickere u datom scenariju."""
    N = model_base.N
    corr = model_base.corr_assets
    # Kreiraj model sa modifikacijama
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
    
    # Simuliraj jednu putanju
    V, lam, default, r, v = model_new.simulate_single_path(return_paths=True)
    t_axis = np.linspace(0, model_new.T, model_new.steps)
    
    # Mapiramo ticker na indeks
    ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
    selected_indices = [ticker_to_idx[t] for t in selected_tickers if t in ticker_to_idx]
    
    # Kreiraj figuru
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Putanje imovine (milijarde $)", "Hawkes intenzitet (λ)"),
                        vertical_spacing=0.15)
    
    for idx in selected_indices:
        ticker = tickers[idx]
        # Asset values (u milijardama)
        fig.add_trace(go.Scatter(x=t_axis, y=V[:, idx]/1e9, mode='lines',
                                 name=f'{ticker} (V)',
                                 line=dict(width=2)),
                      row=1, col=1)
        # Lambda
        y = lam[:, idx]
        x = t_axis[~np.isnan(y)]
        y = y[~np.isnan(y)]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                 name=f'{ticker} (λ)',
                                 line=dict(dash='dot')),
                      row=2, col=1)
    
    # Dodaj barijeru za prvu odabranu firmu (ako postoji)
    if selected_indices:
        idx0 = selected_indices[0]
        D0_0 = model_new.D0[idx0]
        barrier_curve = D0_0 * np.exp(model_new.barrier_growth_rate * t_axis) / 1e9  # u milijardama
        fig.add_trace(go.Scatter(x=t_axis, y=barrier_curve, mode='lines',
                                 name='Barijera (Dug)',
                                 line=dict(color='black', dash='dash')),
                      row=1, col=1)
    
    # Podesi y-osu za lambda
    max_lambda = np.nanmax(lam[:, selected_indices]) if selected_indices else 0.01
    if max_lambda > 0:
        fig.update_yaxes(range=[0, max_lambda * 1.2], row=2, col=1)
    else:
        fig.update_yaxes(range=[0, 0.05], row=2, col=1)
    
    fig.update_layout(height=600, width=1000, showlegend=True)
    return fig

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
        
        # ======================
        # REZULTATI – METRIKE
        # ======================
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("VaR (99%)", f"${VaR:,.0f}")
        with col2:
            st.metric("CVaR (99%)", f"${CVaR:,.0f}")
        with col3:
            st.metric("Default Prob", f"{default_prob:.2f}%")
        with col4:
            st.metric("Prosečan gubitak", f"${avg_loss:,.0f}")
        
        # ======================
        # HISTOGRAM GUBITAKA
        # ======================
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
        
        # ======================
        # DD BAR CHART (sve firme)
        # ======================
        st.subheader("📊 Distance-to-Default (DD) – sve firme")
        dd_sorted_idx = np.argsort(dd)[::-1]  # opadajuće (od najsigurnijih ka najrizičnijim)
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
                         annotation_text="Srednji rizik (DD=2.5)")
        fig_dd.add_hline(y=1.5, line_dash="dash", line_color="red",
                         annotation_text="Visok rizik (DD=1.5)")
        fig_dd.update_layout(
            height=500,
            xaxis_title="Ticker",
            yaxis_title="Distance-to-Default",
            showlegend=False
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # ======================
        # DISTRIBUCIJA DD (histogram)
        # ======================
        st.subheader("📊 Distribucija Distance-to-Default")
        fig_dd_hist = go.Figure()
        fig_dd_hist.add_trace(go.Histogram(
            x=dd,
            nbinsx=20,
            marker_color='lightblue',
            opacity=0.7
        ))
        fig_dd_hist.add_vline(x=2.5, line_dash="dash", line_color="orange",
                              annotation_text="DD=2.5")
        fig_dd_hist.add_vline(x=1.5, line_dash="dash", line_color="red",
                              annotation_text="DD=1.5")
        fig_dd_hist.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_dd_hist, use_container_width=True)
        
        # ======================
        # KORELACIONA HEATMAP
        # ======================
        st.subheader("🔗 Korelaciona matrica povrata")
        corr_matrix = data['correlation_matrix']
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=valid_tickers,
            y=valid_tickers,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Korelacija")
        ))
        fig_corr.update_layout(height=600, width=800)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # ======================
        # TOP 10 NAJRIZIČNIJIH (tabela)
        # ======================
        st.subheader("🔥 Top 10 najrizičnijih firmi")
        dd_asc_idx = np.argsort(dd)  # rastuće (najmanji DD = najrizičniji)
        top10_data = []
        for i in range(min(10, len(valid_tickers))):
            idx = dd_asc_idx[i]
            top10_data.append({
                'Ticker': valid_tickers[idx],
                'DD': dd[idx],
                'V0 (B$)': model.V0[idx] / 1e9,
                'Dug (B$)': model.D0[idx] / 1e9,
                'Vol (%)': model.vol[idx] * 100
            })
        df_top10 = pd.DataFrame(top10_data)
        st.dataframe(df_top10, use_container_width=True, hide_index=True)
        
        # ======================
        # SEKTORSKA ANALIZA (ako su sektori dostupni)
        # ======================
        # Mapa sektora (može se proširiti)
        sector_map = {
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
        sectors = [sector_map.get(t, 'Other') for t in valid_tickers]
        if len(set(sectors)) > 1:
            st.subheader("🏭 Sektorska analiza")
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
            
            # Bar chart prosečnog DD po sektoru
            fig_sector = go.Figure()
            fig_sector.add_trace(go.Bar(
                x=sector_agg.index,
                y=sector_agg['DD_avg'],
                marker_color='lightgreen',
                text=[f'{v:.2f}' for v in sector_agg['DD_avg']],
                textposition='outside'
            ))
            fig_sector.add_hline(y=2.5, line_dash="dash", line_color="orange")
            fig_sector.update_layout(height=400, xaxis_title="Sektor", yaxis_title="Prosečan DD")
            st.plotly_chart(fig_sector, use_container_width=True)
        
        # ======================
        # INTERAKTIVNE PUTANJE
        # ======================
        st.subheader("🎯 Interaktivne putanje imovine i zaraze")
        
        # Izbor firmi za prikaz
        selected_for_paths = st.multiselect(
            "Odaberi firme za prikaz putanja:",
            options=valid_tickers,
            default=valid_tickers[:5] if len(valid_tickers) >= 5 else valid_tickers
        )
        
        # Izbor scenarija
        scenario_names = list(scenario_modifiers.keys())
        selected_scenario = st.selectbox("Odaberi scenario:", scenario_names, index=0)
        mods = scenario_modifiers[selected_scenario]
        
        if st.button("🔄 Generiši putanje za odabrane firme"):
            if selected_for_paths:
                with st.spinner(f"Generisanje putanja za scenario: {selected_scenario}..."):
                    fig_paths = generate_paths_for_scenario(
                        model, mods, valid_tickers, selected_for_paths
                    )
                    st.plotly_chart(fig_paths, use_container_width=True)
            else:
                st.warning("Odaberi barem jednu firmu.")
        
        # Dugme za top 5 rizičnih
        if st.button("🔴 Generiši putanje za 5 najrizičnijih firmi"):
            top5_risky = [valid_tickers[i] for i in dd_asc_idx[:5]]
            with st.spinner("Generisanje putanja za 5 najrizičnijih..."):
                fig_paths_top5 = generate_paths_for_scenario(
                    model, mods, valid_tickers, top5_risky
                )
                st.plotly_chart(fig_paths_top5, use_container_width=True)
        
        # ======================
        # STATISTIČKI PREGLED
        # ======================
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
            st.metric("Broj firmi sa DD < 1.5", f"{np.sum(dd < 1.5)}")
    # ================================================================
        # RESOURCES / BIBLIOTEKA
        # ================================================================
        st.subheader("📚 Resources & Literature")
        
        with st.expander("📖 Ključne knjige i radovi", expanded=False):
            st.markdown("""
            ### 📕 Knjige
            
            | Knjiga | Autor | Godina | Opis |
            |--------|-------|--------|------|
            | **Options, Futures, and Other Derivatives** | John C. Hull | 2021 | Osnovni udžbenik za derivativne instrumente i modele vrednovanja |
            | **Dynamic Asset Pricing Theory** | Darrell Duffie | 2001 | Matematički okvir za modeliranje cena imovine i rizika |
            | **Credit Risk: Modeling, Valuation and Hedging** | Tomasz R. Bielecki, Marek Rutkowski | 2002 | Napredni modeli kreditnog rizika, uključujući strukturni pristup |
            | **Fixed Income Securities** | Bruce Tuckman, Angel Serrat | 2011 | Detaljan pregled fiksnih prinosa i kreditnih spread-ova |
            | **Risk Management and Financial Institutions** | John C. Hull | 2018 | Upravljanje rizikom u finansijskim institucijama |
            
            ### 📄 Ključni akademski radovi
            
            | Rad | Autor(i) | Godina | Ključni doprinos |
            |-----|----------|--------|------------------|
            | **On the Pricing of Corporate Debt: The Risk Structure of Interest Rates** | Robert C. Merton | 1974 | Osnovni strukturni model – imovina firme kao opcija |
            | **A Theory of the Term Structure of Interest Rates** | John C. Cox, Jonathan E. Ingersoll, Stephen A. Ross | 1985 | CIR model kamatnih stopa |
            | **A Jump-Diffusion Model for Asset Returns** | Chunsheng Zhou | 2001 | Jump-Diffusion model za kreditne spread-ove i korelacije default-a |
            | **Do Credit Spreads Reflect Stationary Leverage?** | Pierre Collin-Dufresne, Robert S. Goldstein | 2001 | Dinamička barijera i stacionarni leveridž |
            | **Term Structures of Credit Spreads with Incomplete Accounting Information** | Darrell Duffie, David Lando | 2001 | Model nepotpunih informacija za kreditne spread-ove |
            | **Extensions to the Gaussian Copula: Random Recovery and Random Factor Loadings** | Leif Andersen, Jakob Sidenius | 2004 | Stohastički oporavak u modelima kreditnog rizika |
            | **The KMV Approach to Credit Risk** | Kealhofer, McQuown, Vasicek | 1990s | KMV kalibracija i Distance-to-Default (DD) |
            | **Hawkes Processes and Their Applications in Finance** | Emmanuel Bacry, Iacopo Mastromatteo, Jean-François Muzy | 2015 | Pregled primjene Hawkes procesa u finansijama |
            | **Credit Default Swaps and the Credit Crisis** | David X. Li | 2006 | CDS i korelacioni modeli (Gaussian copula) |
            
            ### 🌐 Izvori podataka
            
            | Izvor | Link | Opis |
            |-------|------|------|
            | **FRED (Federal Reserve Economic Data)** | [fred.stlouisfed.org](https://fred.stlouisfed.org) | Treasury stope, BAA korporativni prinosi, makro podaci |
            | **Yahoo Finance** | [finance.yahoo.com](https://finance.yahoo.com) | Cene akcija, tržišna kapitalizacija, volatilnost |
            | **Financial Modeling Prep** | [financialmodelingprep.com](https://financialmodelingprep.com) | Bilansni podaci i finansijski izveštaji (API) |
            | **SEC EDGAR** | [sec.gov/edgar](https://sec.gov/edgar) | 10-K i 10-Q izveštaji kompanija |
            | **Moody's Default Rates** | [moodys.com](https://www.moodys.com) | Istorijske stope defaulta po rejtingu |
            | **S&P Global Default Studies** | [spglobal.com](https://www.spglobal.com) | Kumulativne stope defaulta i recovery rates |
            
            ### 💡 Konceptualni doprinosi
            
            | Koncept | Opis | Ključni autori |
            |---------|------|----------------|
            | **KMV pristup** | Izvlačenje nevidljive imovine (V0) i volatilnosti iz tržišnih podataka | Kealhofer, McQuown, Vasicek |
            | **Distance-to-Default (DD)** | Mjera udaljenosti imovine od barijere bankrota | Merton, KMV |
            | **Hawkes proces** | Samopobudni proces za modeliranje klasterovanja default-a | Hawkes (1971), Bacry et al. (2015) |
            | **Jump-Diffusion** | Kombinacija kontinuiranog GBM i diskretnih skokova | Zhou (2001) |
            | **Regime-Switching** | Markovljevi lanci za prelazak između normalnog i stresnog režima | Hamilton (1989), Giesecke |
            | **Stohastički oporavak** | Stopa oporavka negativno korelisana sa intenzitetom bankrota | Andersen & Sidenius (2004) |
            | **Nepotpune informacije** | Šum na opservaciju barijere zbog kašnjenja u računovodstvenim izveštajima | Duffie & Lando (2001) |
            """)
            
        # Dodaj biblioteku i na dno sidebar-a (opciono)
        with st.sidebar.expander("📚 Literatura", expanded=False):
            st.markdown("""
            - **Merton (1974)** – Strukturni model
            - **Zhou (2001)** – Jump-Diffusion
            - **Duffie & Lando (2001)** – Nepotpune informacije
            - **Collin-Dufresne & Goldstein (2001)** – Dinamička barijera
            - **Andersen & Sidenius (2004)** – Stohastički oporavak
            - **Hull (2021)** – Derivati i upravljanje rizikom
            - **FRED / Yahoo Finance** – Podaci
            """)
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
    - **Interaktivne putanje** – praćenje imovine i intenziteta zaraze
    - **Sektorska analiza** – agregacija rizika po sektorima
    """)
