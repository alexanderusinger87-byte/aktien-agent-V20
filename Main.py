import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import numbers

# ==============================================================================
# 1. KONFIGURATION & SETUP
# ==============================================================================

st.set_page_config(
    page_title="KPAX Screener V20.1",
    page_icon="📈",
    layout="wide"
)

st.title("KPAX Screener V20.1")

DEFAULT_TICKERS = [
    "SMO", "BMW.DE", "MAIR", "GOOGL", "IFX.DE", "1810.HK", "PEP", "KO",
    "NVDA", "AAPL", "MU", "INTC", "AMZN", "F8P", "NOK", "VST",
    "ASML", "NKE", "VRT", "TSM", "NVO", "MRVL", "000660.KS", "TSLA",
    "005930.KS", "AMD", "ADS.DE", "SU.PA", "ENR.DE", "SIE.DE", "MSFT",
    "AVGO", "SSUN.F", "MCD", "MUV2.DE", "ALV.DE"
]


# ==============================================================================
# 2. HILFSFUNKTIONEN
# ==============================================================================

def is_valid_number(value):
    """Prüft zuverlässig, ob ein Wert eine verwendbare Zahl ist."""
    return (
        isinstance(value, numbers.Number)
        and not isinstance(value, bool)
        and np.isfinite(value)
    )


def score_with_minimum(score):
    """
    Real berechnete Scores werden auf mindestens 5 Punkte begrenzt.
    NaN bleibt NaN = Daten fehlen.
    """
    if pd.isna(score):
        return np.nan

    return max(5.0, float(score))


def calc_rsi(prices, period=14):
    """Berechnet RSI(14)."""

    if prices is None or len(prices) < period + 1:
        return np.nan

    prices = pd.to_numeric(prices, errors="coerce").dropna()

    if len(prices) < period + 1:
        return np.nan

    delta = prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    if avg_loss.iloc[-1] == 0:
        return 100.0

    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]

    return 100 - (100 / (1 + rs))


# ==============================================================================
# 3. EINZELSCORES
# ==============================================================================

def calculate_individual_scores(info, hist):

    if not isinstance(info, dict):
        info = {}

    # --------------------------------------------------------------------------
    # 1. VALUATION
    # --------------------------------------------------------------------------

    pe = info.get("trailingPE")
    peg = info.get("pegRatio")

    val_points = []

    # PEG
    if is_valid_number(peg) and peg > 0:

        if peg < 1.0:
            val_points.append(100)
        elif peg <= 1.5:
            val_points.append(60)
        elif peg <= 2.0:
            val_points.append(30)
        else:
            val_points.append(0)

    # KGV
    if is_valid_number(pe) and pe > 0:

        if pe < 15:
            val_points.append(100)
        elif pe <= 25:
            val_points.append(60)
        elif pe <= 35:
            val_points.append(30)
        else:
            val_points.append(0)

    if val_points:
        valuation_score = score_with_minimum(np.mean(val_points))
    else:
        valuation_score = np.nan


    # --------------------------------------------------------------------------
    # 2. GROWTH
    # --------------------------------------------------------------------------

    earnings_growth = info.get("earningsGrowth")
    revenue_growth = info.get("revenueGrowth")

    growth_points = []

    if is_valid_number(earnings_growth):

        if earnings_growth > 0.20:
            growth_points.append(100)
        elif earnings_growth > 0.10:
            growth_points.append(70)
        elif earnings_growth > 0:
            growth_points.append(40)
        else:
            growth_points.append(0)

    if is_valid_number(revenue_growth):

        if revenue_growth > 0.15:
            growth_points.append(100)
        elif revenue_growth > 0.05:
            growth_points.append(60)
        elif revenue_growth > 0:
            growth_points.append(30)
        else:
            growth_points.append(0)

    if growth_points:
        growth_score = score_with_minimum(np.mean(growth_points))
    else:
        growth_score = np.nan


    # --------------------------------------------------------------------------
    # 3. FINANCIAL HEALTH
    # --------------------------------------------------------------------------

    debt_to_equity = info.get("debtToEquity")
    current_ratio = info.get("currentRatio")

    health_points = []

    if is_valid_number(debt_to_equity):

        # Yahoo liefert D/E meistens als Prozentwert.
        # Beispiel: 80 = 80% = 0.8
        de_ratio = (
            debt_to_equity / 100
            if debt_to_equity > 10
            else debt_to_equity
        )

        if de_ratio < 0.5:
            health_points.append(100)
        elif de_ratio <= 1.5:
            health_points.append(60)
        else:
            health_points.append(10)

    if is_valid_number(current_ratio):

        if current_ratio > 1.5:
            health_points.append(100)
        elif current_ratio >= 1.0:
            health_points.append(50)
        else:
            health_points.append(0)

    if health_points:
        health_score = score_with_minimum(np.mean(health_points))
    else:
        health_score = np.nan


    # --------------------------------------------------------------------------
    # 4. MOMENTUM / TECHNIK
    # --------------------------------------------------------------------------

    tech_points = []

    if hist is not None and not hist.empty:

        close_prices = pd.to_numeric(
            hist["Close"],
            errors="coerce"
        ).dropna()

        if len(close_prices) >= 50:

            current_price = close_prices.iloc[-1]

            # GD200, falls weniger Daten vorhanden sind automatisch
            # über vorhandenen Zeitraum berechnen.
            window = min(200, len(close_prices))

            sma = (
                close_prices
                .rolling(window=window)
                .mean()
                .iloc[-1]
            )

            rsi = calc_rsi(close_prices)

            # Kurs vs. GD
            if is_valid_number(sma):

                if current_price > sma:
                    tech_points.append(100)
                else:
                    tech_points.append(20)

            # RSI
            if is_valid_number(rsi):

                if 40 <= rsi <= 60:
                    tech_points.append(100)

                elif 30 <= rsi < 40 or 60 < rsi <= 70:
                    tech_points.append(70)

                else:
                    tech_points.append(20)

    if tech_points:
        momentum_score = score_with_minimum(np.mean(tech_points))
    else:
        momentum_score = np.nan


    return {
        "Valuation": valuation_score,
        "Growth": growth_score,
        "Health": health_score,
        "Momentum": momentum_score
    }


# ==============================================================================
# 4. YAHOO DATEN ABRUFEN
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screener_data(tickers):

    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        return pd.DataFrame(), []

    data = []
    errors = []

    # --------------------------------------------------------------------------
    # KURSDATEN IN EINEM EINZIGEN DOWNLOAD
    # --------------------------------------------------------------------------

    try:

        hist_all = yf.download(
            tickers=tickers,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True
        )

    except Exception as e:

        return pd.DataFrame(), [
            f"Fehler beim Yahoo-Kursabruf: {str(e)}"
        ]


    # --------------------------------------------------------------------------
    # EINZELNE AKTIEN VERARBEITEN
    # --------------------------------------------------------------------------

    for idx, symbol in enumerate(tickers):

        try:

            # --------------------------------------------------------------
            # HISTORIE EXTRAHIEREN
            # --------------------------------------------------------------

            if len(tickers) == 1:

                hist = hist_all.copy()

            else:

                try:
                    hist = hist_all[symbol].copy()
                except Exception:
                    hist = pd.DataFrame()

            if hist.empty or "Close" not in hist.columns:

                errors.append(f"{symbol}: keine Kursdaten")
                continue


            hist = hist.dropna(subset=["Close"])

            if hist.empty:

                errors.append(f"{symbol}: keine gültigen Schlusskurse")
                continue


            # --------------------------------------------------------------
            # FUNDAMENTALDATEN
            # --------------------------------------------------------------

            ticker = yf.Ticker(symbol)

            info = {}

            try:
                info = ticker.get_info()

                if not isinstance(info, dict):
                    info = {}

            except Exception as e:

                errors.append(
                    f"{symbol}: Fundamentaldaten nicht verfügbar"
                )

                info = {}


            # --------------------------------------------------------------
            # KURS
            # --------------------------------------------------------------

            current_price = np.nan

            try:

                current_price = hist["Close"].iloc[-1]

                if isinstance(current_price, pd.Series):
                    current_price = current_price.iloc[-1]

            except Exception:
                pass


            if not is_valid_number(current_price):

                try:
                    current_price = info.get("currentPrice")
                except Exception:
                    current_price = np.nan


            # --------------------------------------------------------------
            # SCORES
            # --------------------------------------------------------------

            scores = calculate_individual_scores(
                info,
                hist
            )


            # --------------------------------------------------------------
            # KENNZAHLEN
            # --------------------------------------------------------------

            pe = info.get("trailingPE", np.nan)
            peg = info.get("pegRatio", np.nan)

            div_yield = info.get(
                "dividendYield",
                np.nan
            )

            if is_valid_number(div_yield):

                # Yahoo liefert Dividend Yield normalerweise als Dezimalzahl
                if div_yield < 1:
                    div_yield *= 100

            else:
                div_yield = np.nan


            # --------------------------------------------------------------
            # DATENSATZ
            # --------------------------------------------------------------

            data.append({

                "Symbol": symbol,

                "Name": info.get(
                    "shortName",
                    symbol
                ),

                "Sektor": info.get(
                    "sector",
                    "N/A"
                ),

                "Kurs": current_price,

                "KGV (P/E)": pe,

                "PEG": peg,

                "Div. Rendite (%)": div_yield,

                "Bewertung": scores["Valuation"],

                "Wachstum": scores["Growth"],

                "Gesundheit": scores["Health"],

                "Momentum": scores["Momentum"]
            })


        except Exception as e:

            errors.append(
                f"{symbol}: {type(e).__name__}: {str(e)}"
            )

            continue


    return pd.DataFrame(data), errors


# ==============================================================================
# 5. SIDEBAR
# ==============================================================================

st.sidebar.header("⚙️ Einstellungen")

tickers_input = st.sidebar.text_area(
    "Ticker-Liste (kommagetrennt):",
    value=", ".join(DEFAULT_TICKERS),
    height=150
)

active_tickers = [
    t.strip()
    for t in tickers_input.split(",")
    if t.strip()
]


# ==============================================================================
# 6. GEWICHTUNG
# ==============================================================================

st.sidebar.subheader("⚖️ Score-Gewichtung")

w_val = st.sidebar.slider(
    "Gewicht Bewertung",
    0.0,
    1.0,
    0.30,
    0.05
)

w_gro = st.sidebar.slider(
    "Gewicht Wachstum",
    0.0,
    1.0,
    0.30,
    0.05
)

w_hea = st.sidebar.slider(
    "Gewicht Gesundheit",
    0.0,
    1.0,
    0.20,
    0.05
)

w_mom = st.sidebar.slider(
    "Gewicht Momentum",
    0.0,
    1.0,
    0.20,
    0.05
)


total_w = w_val + w_gro + w_hea + w_mom

if total_w > 0:

    w_val_n = w_val / total_w
    w_gro_n = w_gro / total_w
    w_hea_n = w_hea / total_w
    w_mom_n = w_mom / total_w

else:

    w_val_n = 0.25
    w_gro_n = 0.25
    w_hea_n = 0.25
    w_mom_n = 0.25


# ==============================================================================
# 7. DATEN LADEN
# ==============================================================================

with st.spinner("📡 Lade Yahoo-Finance-Daten..."):

    df, errors = fetch_screener_data(
        tuple(active_tickers)
    )


# ==============================================================================
# 8. FEHLER / DIAGNOSE
# ==============================================================================

if errors:

    with st.expander(
        f"⚠️ Datenhinweise ({len(errors)})",
        expanded=False
    ):

        for error in errors:
            st.write("•", error)


# ==============================================================================
# 9. GESAMTSCORE
# ==============================================================================

if not df.empty:

    def compute_total_score(row):

        score_weight_pairs = [

            (
                row["Bewertung"],
                w_val_n
            ),

            (
                row["Wachstum"],
                w_gro_n
            ),

            (
                row["Gesundheit"],
                w_hea_n
            ),

            (
                row["Momentum"],
                w_mom_n
            )
        ]

        valid_scores = [
            (score, weight)
            for score, weight in score_weight_pairs
            if pd.notna(score) and weight > 0
        ]

        if not valid_scores:
            return np.nan

        scores = [
            x[0]
            for x in valid_scores
        ]

        weights = [
            x[1]
            for x in valid_scores
        ]

        return np.average(
            scores,
            weights=weights
        )


    df["Gesamtscore"] = df.apply(
        compute_total_score,
        axis=1
    )


    # --------------------------------------------------------------------------
    # SORTIEREN
    # --------------------------------------------------------------------------

    df = (
        df
        .sort_values(
            by="Gesamtscore",
            ascending=False,
            na_position="last"
        )
        .reset_index(drop=True)
    )


    # ==============================================================================
    # 10. TABELLE
    # ==============================================================================

    st.subheader("📋 Aktienübersicht & Scoring")

    st.dataframe(

        df.style.format({

            "Kurs": "{:.2f}",

            "KGV (P/E)": "{:.2f}",

            "PEG": "{:.2f}",

            "Div. Rendite (%)": "{:.2f} %",

            "Bewertung": "{:.1f}",

            "Wachstum": "{:.1f}",

            "Gesundheit": "{:.1f}",

            "Momentum": "{:.1f}",

            "Gesamtscore": "{:.1f}"
        },

        na_rep="—")

        .background_gradient(
            subset=["Gesamtscore"],
            cmap="RdYlGn",
            vmin=0,
            vmax=100
        ),

        use_container_width=True,

        height=500
    )


    # ==============================================================================
    # 11. DATENQUALITÄT
    # ==============================================================================

    st.subheader("🔎 Datenqualität")

    quality_cols = [
        "Bewertung",
        "Wachstum",
        "Gesundheit",
        "Momentum"
    ]

    df_quality = pd.DataFrame({

        "Score": quality_cols,

        "Vorhanden": [
            int(df[c].notna().sum())
            for c in quality_cols
        ],

        "Fehlend": [
            int(df[c].isna().sum())
            for c in quality_cols
        ]
    })

    st.dataframe(
        df_quality,
        hide_index=True,
        use_container_width=True
    )


else:

    st.error(
        "❌ Es konnten keine verwertbaren Kursdaten geladen werden."
    )

    if errors:

        st.write("Letzte Fehlermeldungen:")

        for error in errors[:20]:
            st.write("•", error)


# ==============================================================================
# 12. BERECHNUNGSGRUNDLAGEN
# ==============================================================================

st.markdown("---")

st.subheader(
    "📐 Berechnungsgrundlagen der Scores"
)

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        "### 📊 Einzelscores (5 bis 100 Punkte)"
    )

    st.markdown("""
    **Bewertungs-Score (Valuation)**

    * PEG < 1,0 → 100 Punkte
    * PEG 1,0–1,5 → 60 Punkte
    * PEG 1,5–2,0 → 30 Punkte
    * PEG > 2,0 → 0 → **5 Punkte Mindestwert**
    * KGV < 15 → 100 Punkte
    * KGV 15–25 → 60 Punkte
    * KGV 25–35 → 30 Punkte
    * KGV > 35 → 0 → **5 Punkte Mindestwert**

    **Wachstums-Score (Growth)**

    * EPS-Wachstum > 20 % → 100
    * EPS-Wachstum 10–20 % → 70
    * EPS-Wachstum 0–10 % → 40
    * EPS-Wachstum < 0 % → 0 → **5**

    * Umsatzwachstum > 15 % → 100
    * Umsatzwachstum 5–15 % → 60
    * Umsatzwachstum 0–5 % → 30
    * Umsatzwachstum < 0 % → 0 → **5**

    **Finanzielle Gesundheit**

    * D/E < 0,5 → 100
    * D/E 0,5–1,5 → 60
    * D/E > 1,5 → 10

    * Current Ratio > 1,5 → 100
    * Current Ratio 1,0–1,5 → 50
    * Current Ratio < 1,0 → 0 → **5**

    **Momentum / Technik**

    * Kurs über GD → 100
    * Kurs unter GD → 20
    * RSI 40–60 → 100
    * RSI 30–40 bzw. 60–70 → 70
    * RSI außerhalb → 20
    """)


with col2:

    st.markdown(
        "### 🏆 Gesamtscore"
    )

    st.markdown(
        "Der Gesamtscore ist ein gewichteter Mittelwert "
        "der tatsächlich verfügbaren Einzelscores."
    )

    st.latex(
        r"""
        Gesamtscore =
        \frac{
        w_1 V +
        w_2 G +
        w_3 H +
        w_4 M
        }{
        \sum w_{\mathrm{vorhanden}}
        }
        """
    )

    st.markdown(f"""
    **Aktuelle Gewichtung**

    ⚖️ Bewertung: **{w_val_n*100:.1f} %**

    🚀 Wachstum: **{w_gro_n*100:.1f} %**

    🛡️ Gesundheit: **{w_hea_n*100:.1f} %**

    📈 Momentum: **{w_mom_n*100:.1f} %**

    ---

    **Wichtig:**

    **— = Daten fehlen**

    **5 = Daten vorhanden, aber schlechtester Score**

    Dadurch kann ein echter schlechter Score nicht mehr
    mit einer Datenlücke verwechselt werden.

    Fehlt beispielsweise das PEG, wird die Bewertung
    ausschließlich aus dem vorhandenen KGV berechnet.

    Fehlen sämtliche Bewertungsdaten, bleibt der
    Bewertungs-Score **NaN** und wird beim Gesamtscore
    automatisch nicht berücksichtigt.
    """)
