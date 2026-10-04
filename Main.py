import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ==============================================================================
# 1. KONFIGURATION & SETUP
# ==============================================================================
st.set_page_config(
    page_title="KPAX Screener V20.0",
    page_icon="📈",
    layout="wide"
)

st.title("KPAX Screener V20.0")

# Standard-Tickerliste
DEFAULT_TICKERS = [
    "SMO", "BMW.DE", "MAIR", "GOOGL", "IFX.DE", "1810.HK", "PEP", "KO",
    "NVDA", "AAPL", "MU", "INTC", "AMZN", "F8P", "NOK", "VST",
    "ASML", "NKE", "VRT", "TSM", "NVO", "MRVL", "000660.KS", "TSLA",
    "005930.KS", "AMD", "ADS.DE", "SU.PA", "ENR.DE", "SIE.DE", "MSFT",
    "AVGO", "SSUN.F", "MCD", "MUV2.DE", "ALV.DE"
]

# ==============================================================================
# 2. HELFERFUNKTIONEN & SCORE-BERECHNUNGEN
# ==============================================================================

def calc_rsi(prices, period=14):
    """Berechnet den Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return np.nan
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_individual_scores(info, hist):
    """Berechnet die 4 Einzelscores (0 bis 100 Punkte) basierend auf Fundamental- und Technikdaten."""
    
    # 1. VALUATION SCORE
    pe = info.get("trailingPE", None)
    peg = info.get("pegRatio", None)
    
    val_points = []
    if peg and peg > 0:
        if peg < 1.0: val_points.append(100)
        elif peg <= 1.5: val_points.append(60)
        elif peg <= 2.0: val_points.append(30)
        else: val_points.append(0)
        
    if pe and pe > 0:
        if pe < 15: val_points.append(100)
        elif pe <= 25: val_points.append(60)
        elif pe <= 35: val_points.append(30)
        else: val_points.append(0)
        
    valuation_score = np.mean(val_points) if val_points else np.nan

    # 2. GROWTH SCORE
    earnings_growth = info.get("earningsGrowth", None)
    revenue_growth = info.get("revenueGrowth", None)
    
    growth_points = []
    if earnings_growth is not None:
        if earnings_growth > 0.20: growth_points.append(100)
        elif earnings_growth > 0.10: growth_points.append(70)
        elif earnings_growth > 0: growth_points.append(40)
        else: growth_points.append(0)
        
    if revenue_growth is not None:
        if revenue_growth > 0.15: growth_points.append(100)
        elif revenue_growth > 0.05: growth_points.append(60)
        elif revenue_growth > 0: growth_points.append(30)
        else: growth_points.append(0)
        
    growth_score = np.mean(growth_points) if growth_points else np.nan

    # 3. FINANCIAL HEALTH SCORE
    debt_to_equity = info.get("debtToEquity", None)
    current_ratio = info.get("currentRatio", None)
    
    health_points = []
    if debt_to_equity is not None:
        # yfinance liefert debtToEquity meist als Prozentwert (z.B. 50 für 0.5)
        de_val = debt_to_equity / 100.0 if debt_to_equity > 10 else debt_to_equity
        if de_val < 0.5: health_points.append(100)
        elif de_val <= 1.5: health_points.append(60)
        else: health_points.append(10)
        
    if current_ratio is not None:
        if current_ratio > 1.5: health_points.append(100)
        elif current_ratio >= 1.0: health_points.append(50)
        else: health_points.append(0)
        
    health_score = np.mean(health_points) if health_points else np.nan

    # 4. MOMENTUM / TECHNIK SCORE
    tech_points = []
    if len(hist) >= 200:
        close_prices = hist["Close"]
        current_price = close_prices.iloc[-1]
        sma_200 = close_prices.rolling(window=200).mean().iloc[-1]
        rsi = calc_rsi(close_prices)
        
        if current_price > sma_200:
            tech_points.append(100)
        else:
            tech_points.append(20)
            
        if not np.isnan(rsi):
            if 40 <= rsi <= 60: tech_points.append(100)
            elif 30 <= rsi < 40 or 60 < rsi <= 70: tech_points.append(70)
            else: tech_points.append(20)
            
    momentum_score = np.mean(tech_points) if tech_points else np.nan

    return {
        "Valuation": valuation_score,
        "Growth": growth_score,
        "Health": health_score,
        "Momentum": momentum_score
    }

@st.cache_data(ttl=3600)
def fetch_screener_data(tickers):
    """Ruft Daten von yfinance ab und berechnet alle Kennzahlen sowie Scores."""
    data = []
    
    for symbol in tickers:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1y")
            
            if hist.empty:
                continue
                
            scores = calculate_individual_scores(info, hist)
            
            current_price = info.get("currentPrice", hist["Close"].iloc[-1])
            pe = info.get("trailingPE", np.nan)
            peg = info.get("pegRatio", np.nan)
            div_yield = info.get("dividendYield", 0)
            if div_yield:
                div_yield *= 100
                
            data.append({
                "Symbol": symbol,
                "Name": info.get("shortName", symbol),
                "Sektor": info.get("sector", "N/A"),
                "Kurs": current_price,
                "KGV (P/E)": pe,
                "PEG": peg,
                "Div. Rendite (%)": div_yield,
                "Bewertung": scores["Valuation"],
                "Wachstum": scores["Growth"],
                "Gesundheit": scores["Health"],
                "Momentum": scores["Momentum"]
            })
        except Exception:
            continue
            
    return pd.DataFrame(data)

# ==============================================================================
# 3. SIDEBAR STEUERUNG & GEWICHTUNG
# ==============================================================================
st.sidebar.header("⚙️ Einstellungen")

tickers_input = st.sidebar.text_area(
    "Ticker-Liste (kommagetrennt):",
    value=", ".join(DEFAULT_TICKERS),
    height=150
)
active_tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

st.sidebar.subheader("⚖️ Score-Gewichtung")
w_val = st.sidebar.slider("Gewicht Bewertung", 0.0, 1.0, 0.30, 0.05)
w_gro = st.sidebar.slider("Gewicht Wachstum", 0.0, 1.0, 0.30, 0.05)
w_hea = st.sidebar.slider("Gewicht Gesundheit", 0.0, 1.0, 0.20, 0.05)
w_mom = st.sidebar.slider("Gewicht Momentum", 0.0, 1.0, 0.20, 0.05)

# Normalisierung der Gewichte auf Summe = 1
total_w = w_val + w_gro + w_hea + w_mom
if total_w > 0:
    w_val_n, w_gro_n, w_hea_n, w_mom_n = w_val/total_w, w_gro/total_w, w_hea/total_w, w_mom/total_w
else:
    w_val_n = w_gro_n = w_hea_n = w_mom_n = 0.25

# ==============================================================================
# 4. TAZEN- & TABELLEN-DARSTELLUNG
# ==============================================================================
df = fetch_screener_data(active_tickers)

if not df.empty:
    # Gesamtscore dynamisch aus den vom Nutzer gewählten Gewichten berechnen
    def compute_total_score(row):
        scores = []
        weights = []
        if pd.notnull(row["Bewertung"]):
            scores.append(row["Bewertung"])
            weights.append(w_val_n)
        if pd.notnull(row["Wachstum"]):
            scores.append(row["Wachstum"])
            weights.append(w_gro_n)
        if pd.notnull(row["Gesundheit"]):
            scores.append(row["Gesundheit"])
            weights.append(w_hea_n)
        if pd.notnull(row["Momentum"]):
            scores.append(row["Momentum"])
            weights.append(w_mom_n)
            
        if not scores or sum(weights) == 0:
            return np.nan
        return np.average(scores, weights=weights)

    df["Gesamtscore"] = df.apply(compute_total_score, axis=1)
    
    # Sortierung standardmäßig nach Gesamtscore
    df = df.sort_values(by="Gesamtscore", ascending=False).reset_index(drop=True)
    
    # Tabelle anzeigen
    st.subheader("📋 Aktienübersicht & Scoring")
    
    st.dataframe(
        df.style.format({
            "Kurs": "{:.2f} €",
            "KGV (P/E)": "{:.2f}",
            "PEG": "{:.2f}",
            "Div. Rendite (%)": "{:.2f} %",
            "Bewertung": "{:.1f}",
            "Wachstum": "{:.1f}",
            "Gesundheit": "{:.1f}",
            "Momentum": "{:.1f}",
            "Gesamtscore": "{:.1f}"
        }).background_gradient(subset=["Gesamtscore"], cmap="RdYlGn", vmin=0, vmax=100),
        use_container_width=True,
        height=500
    )
else:
    st.warning("Keine Daten für die ausgewählten Ticker gefunden.")

# ==============================================================================
# 5. DOKUMENTATION & BERECHNUNGSGRUNDLAGEN (UNTEN IM SCREENER)
# ==============================================================================
st.markdown("---")
st.subheader("📐 Berechnungsgrundlagen der Scores")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Einzelscores (0 bis 100 Punkte)")
    
    st.markdown("""
    * **Bewertungs-Score (Valuation):**
      * **PEG-Ratio:** $< 1,0 \rightarrow 100\text{ Pkt}$, $1,0 - 1,5 \rightarrow 60\text{ Pkt}$, $> 2,0 \rightarrow 0\text{ Pkt}$
      * **KGV (P/E):** $< 15 \rightarrow 100\text{ Pkt}$, $15 - 25 \rightarrow 60\text{ Pkt}$, $> 35 \rightarrow 0\text{ Pkt}$
    
    * **Wachstums-Score (Growth):**
      * **EPS-Wachstum:** $> 20\% \rightarrow 100\text{ Pkt}$, $10\% - 20\% \rightarrow 70\text{ Pkt}$, $< 0\% \rightarrow 0\text{ Pkt}$
      * **Umsatzwachstum:** $> 15\% \rightarrow 100\text{ Pkt}$, $5\% - 15\% \rightarrow 60\text{ Pkt}$, $< 0\% \rightarrow 0\text{ Pkt}$
    
    * **Finanzielle Gesundheit (Financial Health):**
      * **Verschuldungsgrad (Debt/Equity):** $< 0,5 \rightarrow 100\text{ Pkt}$, $0,5 - 1,5 \rightarrow 60\text{ Pkt}$, $> 1,5 \rightarrow 10\text{ Pkt}$
      * **Current Ratio:** $> 1,5 \rightarrow 100\text{ Pkt}$, $1,0 - 1,5 \rightarrow 50\text{ Pkt}$, $< 1,0 \rightarrow 0\text{ Pkt}$
    
    * **Momentum / Technik-Score:**
      * **RSI (14 Tage):** $40 - 60 \rightarrow 100\text{ Pkt}$, neutraler Bereich; Überkauft/Überverkauft Abzüge.
      * **Abstand GD200:** Kurs über GD200 $\rightarrow 100\text{ Pkt}$, unter GD200 $\rightarrow 20\text{ Pkt}$.
    """)

with col2:
    st.markdown("### 🏆 Gesamtscore-Berechnung")
    
    st.markdown("""
    Der **Gesamtscore** berechnet sich aus dem gewichteten Mittelwert der vier Einzelscores:
    """)
    
    st.latex(r"""
    \text{Gesamtscore} = \frac{(w_1 \cdot \text{Bewertung}) + (w_2 \cdot \text{Wachstum}) + (w_3 \cdot \text{Gesundheit}) + (w_4 \cdot \text{Momentum})}{w_1 + w_2 + w_3 + w_4}
    """)
    
    st.markdown(f"""
    **Aktuell eingestellte Gewichtung (Sidebar):**
    * ⚖️ **Bewertung:** {w_val_n*100:.1f} % ($w_1 = {w_val_n:.2f}$)
    * 🚀 **Wachstum:** {w_gro_n*100:.1f} % ($w_2 = {w_gro_n:.2f}$)
    * 🛡️ **Finanzielle Gesundheit:** {w_hea_n*100:.1f} % ($w_3 = {w_hea_n:.2f}$)
    * 📈 **Momentum:** {w_mom_n*100:.1f} % ($w_4 = {w_mom_n:.2f}$)

    ---
    *Hinweis: Fehlen bei einzelnen Titeln Kennzahlen (z.B. fehlendes PEG bei negativen Gewinnen), werden die verbleibenden Einzelscores für die Gesamtberechnung automatisch re-normiert.*
    """)
