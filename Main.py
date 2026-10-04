import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import numbers

# ==============================================================================
# 1. KONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="KPAX Screener V20.2",
    page_icon="📈",
    layout="wide"
)

st.title("KPAX Screener V20.2")

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
    """Prüft, ob ein Wert eine gültige Zahl ist."""

    return (
        isinstance(value, numbers.Number)
        and not isinstance(value, bool)
        and np.isfinite(value)
    )


def minimum_score(score):
    """
    Echter Score:
        0 -> 5
        10 -> 10
        ...
        100 -> 100

    NaN bleibt NaN = Daten fehlen.
    """

    if pd.isna(score):
        return np.nan

    return max(5.0, float(score))


def safe_float(value):
    """Konvertiert Zahlen robust nach float."""

    if value is None:
        return np.nan

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


# ==============================================================================
# 3. RSI
# ==============================================================================

def calc_rsi(prices, period=14):

    if prices is None or len(prices) < period + 1:
        return np.nan

    prices = pd.to_numeric(
        prices,
        errors="coerce"
    ).dropna()

    if len(prices) < period + 1:
        return np.nan

    delta = prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]

    if pd.isna(last_gain) or pd.isna(last_loss):
        return np.nan

    if last_loss == 0:
        return 100.0

    rs = last_gain / last_loss

    return 100 - (100 / (1 + rs))


# ==============================================================================
# 4. EINZELSCORES
# ==============================================================================

def calculate_individual_scores(info, hist):

    if not isinstance(info, dict):
        info = {}

    # ==========================================================================
    # VALUATION
    # ==========================================================================

    pe = safe_float(info.get("trailingPE"))
    peg = safe_float(info.get("pegRatio"))

    valuation_points = []

    # PEG
    if not pd.isna(peg) and peg > 0:

        if peg < 1.0:
            valuation_points.append(100)

        elif peg <= 1.5:
            valuation_points.append(60)

        elif peg <= 2.0:
            valuation_points.append(30)

        else:
            valuation_points.append(0)

    # KGV
    if not pd.isna(pe) and pe > 0:

        if pe < 15:
            valuation_points.append(100)

        elif pe <= 25:
            valuation_points.append(60)

        elif pe <= 35:
            valuation_points.append(30)

        else:
            valuation_points.append(0)

    if valuation_points:
        valuation_score = minimum_score(
            np.mean(valuation_points)
        )
    else:
        valuation_score = np.nan


    # ==========================================================================
    # GROWTH
    # ==========================================================================

    earnings_growth = safe_float(
        info.get("earningsGrowth")
    )

    revenue_growth = safe_float(
        info.get("revenueGrowth")
    )

    growth_points = []

    # EPS-Wachstum
    if not pd.isna(earnings_growth):

        if earnings_growth > 0.20:
            growth_points.append(100)

        elif earnings_growth > 0.10:
            growth_points.append(70)

        elif earnings_growth > 0:
            growth_points.append(40)

        else:
            growth_points.append(0)

    # Umsatzwachstum
    if not pd.isna(revenue_growth):

        if revenue_growth > 0.15:
            growth_points.append(100)

        elif revenue_growth > 0.05:
            growth_points.append(60)

        elif revenue_growth > 0:
            growth_points.append(30)

        else:
            growth_points.append(0)

    if growth_points:
        growth_score = minimum_score(
            np.mean(growth_points)
        )
    else:
        growth_score = np.nan


    # ==========================================================================
    # FINANCIAL HEALTH
    # ==========================================================================

    debt_to_equity = safe_float(
        info.get("debtToEquity")
    )

    current_ratio = safe_float(
        info.get("currentRatio")
    )

    health_points = []

    # Debt / Equity
    if not pd.isna(debt_to_equity):

        # Yahoo liefert D/E meistens als Prozentwert.
        # 80 = 80 % = 0.8
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

    # Current Ratio
    if not pd.isna(current_ratio):

        if current_ratio > 1.5:
            health_points.append(100)

        elif current_ratio >= 1.0:
            health_points.append(50)

        else:
            health_points.append(0)

    if health_points:
        health_score = minimum_score(
            np.mean(health_points)
        )
    else:
        health_score = np.nan


    # ==========================================================================
    # MOMENTUM / TECHNIK
    # ==========================================================================

    tech_points = []

    if hist is not None and not hist.empty:

        close_prices = pd.to_numeric(
            hist["Close"],
            errors="coerce"
        ).dropna()

        if len(close_prices) >= 50:

            current_price = close_prices.iloc[-1]

            # GD200 bzw. längstmöglicher Durchschnitt
            window = min(
                200,
                len(close_prices)
            )

            sma = (
                close_prices
                .rolling(window)
                .mean()
                .iloc[-1]
            )

            rsi = calc_rsi(
                close_prices,
                14
            )

            # Kurs vs. GD
            if not pd.isna(sma):

                if current_price > sma:
                    tech_points.append(100)
                else:
                    tech_points.append(20)

            # RSI
            if not pd.isna(rsi):

                if 40 <= rsi <= 60:
                    tech_points.append(100)

                elif 30 <= rsi < 40 or 60 < rsi <= 70:
                    tech_points.append(70)

                else:
                    tech_points.append(20)

    if tech_points:
        momentum_score = minimum_score(
            np.mean(tech_points)
        )
    else:
        momentum_score = np.nan


    return {
        "Valuation": valuation_score,
        "Growth": growth_score,
        "Health": health_score,
        "Momentum": momentum_score
    }


# ==============================================================================
# 5. DATENABRUF
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screener_data(tickers):

    tickers = list(dict.fromkeys(tickers))

    data = []
    errors = []

    if not tickers:
        return pd.DataFrame(), []

    # ==========================================================================
    # KURSDATEN
    # ==========================================================================

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
            f"Kursabruf fehlgeschlagen: {e}"
        ]


    # ==========================================================================
    # EINZELNE TICKER
    # ==========================================================================

    for symbol in tickers:

        try:

            # ------------------------------------------------------------------
            # HISTORIE
            # ------------------------------------------------------------------

            if len(tickers) == 1:

                hist = hist_all.copy()

            else:

                try:
                    hist = hist_all[symbol].copy()

                except Exception:
                    hist = pd.DataFrame()

            if hist.empty or "Close" not in hist.columns:

                errors.append(
                    f"{symbol}: keine Kursdaten"
                )

                continue

            hist = hist.dropna(
                subset=["Close"]
            )

            if hist.empty:

                errors.append(
                    f"{symbol}: leere Kursdaten"
                )

                continue


            # ------------------------------------------------------------------
            # KURS
            # ------------------------------------------------------------------

            current_price = safe_float(
                hist["Close"].iloc[-1]
            )


            # ------------------------------------------------------------------
            # YAHOO TICKER
            # ------------------------------------------------------------------

            ticker = yf.Ticker(symbol)


            # ------------------------------------------------------------------
            # INFO
            # ------------------------------------------------------------------

            info = {}

            try:

                info = ticker.get_info()

                if not isinstance(info, dict):
                    info = {}

            except Exception:

                info = {}


            # ------------------------------------------------------------------
            # FALLBACK: FAST INFO
            # ------------------------------------------------------------------

            fast_info = {}

            try:

                fast_info = dict(
                    ticker.fast_info
                )

            except Exception:

                fast_info = {}


            # ------------------------------------------------------------------
            # FUNDAMENTALDATEN
            # ------------------------------------------------------------------

            pe = safe_float(
                info.get("trailingPE")
            )

            peg = safe_float(
                info.get("pegRatio")
            )

            earnings_growth = safe_float(
                info.get("earningsGrowth")
            )

            revenue_growth = safe_float(
                info.get("revenueGrowth")
            )

            debt_to_equity = safe_float(
                info.get("debtToEquity")
            )

            current_ratio = safe_float(
                info.get("currentRatio")
            )

            div_yield = safe_float(
                info.get("dividendYield")
            )


            # ------------------------------------------------------------------
            # DIVIDENDENRENDITE
            # ------------------------------------------------------------------

            if not pd.isna(div_yield):

                if div_yield < 1:
                    div_yield *= 100


            # ------------------------------------------------------------------
            # SCORES
            # ------------------------------------------------------------------

            scores = calculate_individual_scores(

                {
                    "trailingPE": pe,
                    "pegRatio": peg,
                    "earningsGrowth": earnings_growth,
                    "revenueGrowth": revenue_growth,
                    "debtToEquity": debt_to_equity,
                    "currentRatio": current_ratio
                },

                hist
            )


            # ------------------------------------------------------------------
            # NAME
            # ------------------------------------------------------------------

            name = info.get(
                "shortName",
                info.get(
                    "longName",
                    symbol
                )
            )


            # ------------------------------------------------------------------
            # SEKTOR
            # ------------------------------------------------------------------

            sector = info.get(
                "sector",
                "N/A"
            )


            # ------------------------------------------------------------------
            # DATENZEILE
            # ------------------------------------------------------------------

            data.append({

                "Symbol": symbol,

                "Name": name,

                "Sektor": sector,

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
                f"{symbol}: {type(e).__name__}: {e}"
            )


    return pd.DataFrame(data), errors


# ==============================================================================
# 6. SIDEBAR
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
# 7. GEWICHTUNG
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


# ==============================================================================
# 8. GEWICHTE NORMALISIEREN
# ==============================================================================

total_w = (
    w_val +
    w_gro +
    w_hea +
    w_mom
)

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
# 9. DATEN LADEN
# ==============================================================================

with st.spinner("📡 Lade Yahoo-Finance-Daten..."):

    df, errors = fetch_screener_data(
        tuple(active_tickers)
    )


# ==============================================================================
# 10. DIAGNOSE
# ==============================================================================

if errors:

    with st.expander(
        f"⚠️ Datenhinweise ({len(errors)})"
    ):

        for error in errors:
            st.write("•", error)


# ==============================================================================
# 11. SCORE BERECHNEN
# ==============================================================================

if not df.empty:

    def compute_total_score(row):

        values = [
            (row["Bewertung"], w_val_n),
            (row["Wachstum"], w_gro_n),
            (row["Gesundheit"], w_hea_n),
            (row["Momentum"], w_mom_n)
        ]

        valid = [
            (score, weight)
            for score, weight in values
            if pd.notna(score) and weight > 0
        ]

        if not valid:
            return np.nan

        scores = [
            item[0]
            for item in valid
        ]

        weights = [
            item[1]
            for item in valid
        ]

        return np.average(
            scores,
            weights=weights
        )


    df["Gesamtscore"] = df.apply(
        compute_total_score,
        axis=1
    )


    # ==========================================================================
    # SORTIEREN
    # ==========================================================================

    df = (
        df
        .sort_values(
            "Gesamtscore",
            ascending=False,
            na_position="last"
        )
        .reset_index(drop=True)
    )


    # ==========================================================================
    # TABELLE
    # ==========================================================================

    st.subheader(
        "📋 Aktienübersicht & Scoring"
    )

    st.dataframe(

        df.style.format(

            {
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

            na_rep="—"

        ).background_gradient(

            subset=["Gesamtscore"],

            cmap="RdYlGn",

            vmin=0,

            vmax=100

        ),

        use_container_width=True,

        height=500
    )


    # ==========================================================================
    # DATENQUALITÄT
    # ==========================================================================

    st.subheader("🔎 Datenqualität")

    quality = []

    for col in [
        "KGV (P/E)",
        "PEG",
        "Bewertung",
        "Wachstum",
        "Gesundheit",
        "Momentum"
    ]:

        available = int(
            df[col].notna().sum()
        )

        missing = int(
            df[col].isna().sum()
        )

        quality.append({

            "Kennzahl": col,

            "Daten vorhanden": available,

            "Daten fehlen": missing,

            "Abdeckung": (
                f"{available / len(df) * 100:.0f}%"
                if len(df) > 0
                else "0%"
            )
        })


    st.dataframe(
        pd.DataFrame(quality),
        hide_index=True,
        use_container_width=True
    )


else:

    st.error(
        "❌ Es konnten keine Aktienkursdaten geladen werden."
    )


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
        "### 📊 Einzelscores (5–100 Punkte)"
    )

    st.markdown("""
**Bewertung (Valuation)**

* PEG < 1,0 → 100 Punkte
* PEG 1,0–1,5 → 60 Punkte
* PEG 1,5–2,0 → 30 Punkte
* PEG > 2,0 → 5 Punkte
* KGV < 15 → 100 Punkte
* KGV 15–25 → 60 Punkte
* KGV 25–35 → 30 Punkte
* KGV > 35 → 5 Punkte

**Wachstum**

* EPS-Wachstum > 20 % → 100
* EPS-Wachstum 10–20 % → 70
* EPS-Wachstum 0–10 % → 40
* EPS-Wachstum < 0 % → 5

* Umsatzwachstum > 15 % → 100
* Umsatzwachstum 5–15 % → 60
* Umsatzwachstum 0–5 % → 30
* Umsatzwachstum < 0 % → 5

**Finanzielle Gesundheit**

* D/E < 0,5 → 100
* D/E 0,5–1,5 → 60
* D/E > 1,5 → 10

* Current Ratio > 1,5 → 100
* Current Ratio 1,0–1,5 → 50
* Current Ratio < 1,0 → 5

**Momentum**

* Kurs über GD → 100
* Kurs unter GD → 20
* RSI 40–60 → 100
* RSI 30–40 bzw. 60–70 → 70
* RSI außerhalb → 20
""")


with col2:

    st.markdown(
        "### 🏆 Gesamtscore-Berechnung"
    )

    st.markdown(
        "Der Gesamtscore ist ein gewichteter Mittelwert "
        "der verfügbaren Einzelscores."
    )

    st.latex(
        r"""
        Gesamtscore =
        \frac{
        w_1V+w_2G+w_3H+w_4M
        }{
        \sum w_{\mathrm{vorhanden}}
        }
        """
    )

    st.markdown(
        f"""
**Aktuelle Gewichtung**

⚖️ Bewertung: **{w_val_n*100:.1f} %**

🚀 Wachstum: **{w_gro_n*100:.1f} %**

🛡️ Gesundheit: **{w_hea_n*100:.1f} %**

📈 Momentum: **{w_mom_n*100:.1f} %**

---

### Bedeutung der Anzeige

**— = Daten fehlen**

**5 = Daten vorhanden, aber schlechtester Score**

Damit ist eindeutig erkennbar, ob ein Titel tatsächlich
schlecht bewertet wird oder ob Yahoo die notwendige
Kennzahl nicht liefert.

Fehlt beispielsweise das PEG, wird der Valuation-Score
nur aus dem vorhandenen KGV berechnet.

Fehlen sämtliche Bewertungsdaten, bleibt der
Valuation-Score **—** und wird beim Gesamtscore
nicht berücksichtigt.
"""
    )
