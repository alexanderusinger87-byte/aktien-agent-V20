import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import numbers

# ==============================================================================
# 1. KONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="KPAX Screener V21.0",
    page_icon="📈",
    layout="wide"
)

st.title("KPAX Screener V21.0")

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

def safe_float(value):

    if value is None:
        return np.nan

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def minimum_score(score):

    if pd.isna(score):
        return np.nan

    return max(5.0, float(score))


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

    g = avg_gain.iloc[-1]
    l = avg_loss.iloc[-1]

    if pd.isna(g) or pd.isna(l):
        return np.nan

    if l == 0:
        return 100.0

    rs = g / l

    return 100 - (100 / (1 + rs))


# ==============================================================================
# 3. SCORE-BERECHNUNG
# ==============================================================================

def calculate_individual_scores(info, hist):

    # --------------------------------------------------------------------------
    # VALUATION
    # --------------------------------------------------------------------------

    pe = safe_float(info.get("trailingPE"))
    peg = safe_float(info.get("pegRatio"))

    valuation_points = []

    if not pd.isna(peg) and peg > 0:

        if peg < 1.0:
            valuation_points.append(100)

        elif peg <= 1.5:
            valuation_points.append(60)

        elif peg <= 2.0:
            valuation_points.append(30)

        else:
            valuation_points.append(0)

    if not pd.isna(pe) and pe > 0:

        if pe < 15:
            valuation_points.append(100)

        elif pe <= 25:
            valuation_points.append(60)

        elif pe <= 35:
            valuation_points.append(30)

        else:
            valuation_points.append(0)

    valuation_score = (
        minimum_score(np.mean(valuation_points))
        if valuation_points
        else np.nan
    )


    # --------------------------------------------------------------------------
    # GROWTH
    # --------------------------------------------------------------------------

    earnings_growth = safe_float(
        info.get("earningsGrowth")
    )

    revenue_growth = safe_float(
        info.get("revenueGrowth")
    )

    growth_points = []

    if not pd.isna(earnings_growth):

        if earnings_growth > 0.20:
            growth_points.append(100)

        elif earnings_growth > 0.10:
            growth_points.append(70)

        elif earnings_growth > 0:
            growth_points.append(40)

        else:
            growth_points.append(0)

    if not pd.isna(revenue_growth):

        if revenue_growth > 0.15:
            growth_points.append(100)

        elif revenue_growth > 0.05:
            growth_points.append(60)

        elif revenue_growth > 0:
            growth_points.append(30)

        else:
            growth_points.append(0)

    growth_score = (
        minimum_score(np.mean(growth_points))
        if growth_points
        else np.nan
    )


    # --------------------------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------------------------

    debt_to_equity = safe_float(
        info.get("debtToEquity")
    )

    current_ratio = safe_float(
        info.get("currentRatio")
    )

    health_points = []

    if not pd.isna(debt_to_equity):

        de = (
            debt_to_equity / 100
            if debt_to_equity > 10
            else debt_to_equity
        )

        if de < 0.5:
            health_points.append(100)

        elif de <= 1.5:
            health_points.append(60)

        else:
            health_points.append(10)

    if not pd.isna(current_ratio):

        if current_ratio > 1.5:
            health_points.append(100)

        elif current_ratio >= 1.0:
            health_points.append(50)

        else:
            health_points.append(0)

    health_score = (
        minimum_score(np.mean(health_points))
        if health_points
        else np.nan
    )


    # --------------------------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------------------------

    tech_points = []

    if hist is not None and not hist.empty:

        close = pd.to_numeric(
            hist["Close"],
            errors="coerce"
        ).dropna()

        if len(close) >= 50:

            current_price = close.iloc[-1]

            window = min(200, len(close))

            sma = (
                close
                .rolling(window)
                .mean()
                .iloc[-1]
            )

            rsi = calc_rsi(close)

            if not pd.isna(sma):

                if current_price > sma:
                    tech_points.append(100)
                else:
                    tech_points.append(20)

            if not pd.isna(rsi):

                if 40 <= rsi <= 60:
                    tech_points.append(100)

                elif 30 <= rsi < 40 or 60 < rsi <= 70:
                    tech_points.append(70)

                else:
                    tech_points.append(20)

    momentum_score = (
        minimum_score(np.mean(tech_points))
        if tech_points
        else np.nan
    )


    return {
        "Valuation": valuation_score,
        "Growth": growth_score,
        "Health": health_score,
        "Momentum": momentum_score
    }


# ==============================================================================
# 4. FUNDAMENTALDATEN HOLEN
# ==============================================================================

def get_fundamental_data(symbol):

    ticker = yf.Ticker(symbol)

    info = {}
    valuation = pd.DataFrame()
    income = pd.DataFrame()
    balance = pd.DataFrame()

    diagnostics = {
        "Info": "FEHLT",
        "Valuation": "FEHLT",
        "Income": "FEHLT",
        "Balance": "FEHLT"
    }

    # --------------------------------------------------------------------------
    # INFO
    # --------------------------------------------------------------------------

    try:

        info = ticker.get_info()

        if isinstance(info, dict) and len(info) > 0:
            diagnostics["Info"] = "OK"

        else:
            info = {}

    except Exception:
        info = {}


    # --------------------------------------------------------------------------
    # VALUATION
    # --------------------------------------------------------------------------

    try:

        valuation = ticker.get_valuation_measures(
            freq="trailing",
            periods=1
        )

        if (
            isinstance(valuation, pd.DataFrame)
            and not valuation.empty
        ):
            diagnostics["Valuation"] = "OK"

    except Exception:
        valuation = pd.DataFrame()


    # --------------------------------------------------------------------------
    # INCOME STATEMENT
    # --------------------------------------------------------------------------

    try:

        income = ticker.get_income_stmt(
            freq="yearly"
        )

        if (
            isinstance(income, pd.DataFrame)
            and not income.empty
        ):
            diagnostics["Income"] = "OK"

    except Exception:
        income = pd.DataFrame()


    # --------------------------------------------------------------------------
    # BALANCE SHEET
    # --------------------------------------------------------------------------

    try:

        balance = ticker.get_balance_sheet(
            freq="yearly"
        )

        if (
            isinstance(balance, pd.DataFrame)
            and not balance.empty
        ):
            diagnostics["Balance"] = "OK"

    except Exception:
        balance = pd.DataFrame()


    # ==========================================================================
    # HILFSFUNKTION: VALUE AUS INFO
    # ==========================================================================

    def info_value(*keys):

        for key in keys:

            if key in info:

                value = safe_float(
                    info.get(key)
                )

                if not pd.isna(value):
                    return value

        return np.nan


    # ==========================================================================
    # VALUATION VALUE
    # ==========================================================================

    def valuation_value(*keys):

        if valuation is None or valuation.empty:
            return np.nan

        for key in keys:

            # Direkter Index
            if key in valuation.index:

                try:

                    value = valuation.loc[key]

                    if isinstance(value, pd.Series):
                        value = value.iloc[-1]

                    value = safe_float(value)

                    if not pd.isna(value):
                        return value

                except Exception:
                    pass

            # Spalten
            if key in valuation.columns:

                try:

                    value = valuation[key]

                    if isinstance(value, pd.Series):
                        value = value.iloc[-1]

                    value = safe_float(value)

                    if not pd.isna(value):
                        return value

                except Exception:
                    pass

        return np.nan


    # ==========================================================================
    # KGV
    # ==========================================================================

    pe = info_value(
        "trailingPE"
    )

    if pd.isna(pe):

        pe = valuation_value(
            "PeRatio",
            "Trailing P/E"
        )


    # ==========================================================================
    # PEG
    # ==========================================================================

    peg = info_value(
        "pegRatio"
    )

    if pd.isna(peg):

        peg = valuation_value(
            "PegRatio",
            "PEG Ratio (5yr expected)"
        )


    # ==========================================================================
    # GROWTH
    # ==========================================================================

    earnings_growth = info_value(
        "earningsGrowth"
    )

    revenue_growth = info_value(
        "revenueGrowth"
    )


    # ==========================================================================
    # FALLBACK: GROWTH AUS INCOME STATEMENT
    # ==========================================================================

    if income is not None and not income.empty:

        try:

            # EPS
            eps_row = None

            for candidate in [
                "DilutedEPS",
                "BasicEPS",
                "Diluted EPS",
                "Basic EPS"
            ]:

                if candidate in income.index:
                    eps_row = candidate
                    break

            if eps_row is not None:

                eps_series = pd.to_numeric(
                    income.loc[eps_row],
                    errors="coerce"
                ).dropna()

                if len(eps_series) >= 2:

                    latest = eps_series.iloc[0]
                    previous = eps_series.iloc[1]

                    if (
                        previous != 0
                        and pd.isna(earnings_growth)
                    ):

                        earnings_growth = (
                            latest / previous
                        ) - 1


            # Revenue
            revenue_row = None

            for candidate in [
                "TotalRevenue",
                "OperatingRevenue",
                "Total Revenue"
            ]:

                if candidate in income.index:
                    revenue_row = candidate
                    break

            if revenue_row is not None:

                revenue_series = pd.to_numeric(
                    income.loc[revenue_row],
                    errors="coerce"
                ).dropna()

                if len(revenue_series) >= 2:

                    latest = revenue_series.iloc[0]
                    previous = revenue_series.iloc[1]

                    if (
                        previous != 0
                        and pd.isna(revenue_growth)
                    ):

                        revenue_growth = (
                            latest / previous
                        ) - 1

        except Exception:
            pass


    # ==========================================================================
    # HEALTH
    # ==========================================================================

    debt_to_equity = info_value(
        "debtToEquity"
    )

    current_ratio = info_value(
        "currentRatio"
    )


    # ==========================================================================
    # FALLBACK BALANCE SHEET
    # ==========================================================================

    if balance is not None and not balance.empty:

        try:

            def bs_value(*names):

                for name in names:

                    if name in balance.index:

                        series = pd.to_numeric(
                            balance.loc[name],
                            errors="coerce"
                        ).dropna()

                        if not series.empty:
                            return float(series.iloc[0])

                return np.nan


            total_debt = bs_value(
                "TotalDebt",
                "LongTermDebtAndCapitalLeaseObligations",
                "LongTermDebt"
            )

            equity = bs_value(
                "StockholdersEquity",
                "CommonStockEquity",
                "TotalEquityGrossMinorityInterest"
            )

            current_assets = bs_value(
                "CurrentAssets"
            )

            current_liabilities = bs_value(
                "CurrentLiabilities"
            )


            if (
                pd.isna(debt_to_equity)
                and not pd.isna(total_debt)
                and not pd.isna(equity)
                and equity != 0
            ):

                debt_to_equity = (
                    total_debt / equity
                ) * 100


            if (
                pd.isna(current_ratio)
                and not pd.isna(current_assets)
                and not pd.isna(current_liabilities)
                and current_liabilities != 0
            ):

                current_ratio = (
                    current_assets /
                    current_liabilities
                )

        except Exception:
            pass


    # ==========================================================================
    # DIVIDEND
    # ==========================================================================

    div_yield = info_value(
        "dividendYield"
    )

    if not pd.isna(div_yield):

        if div_yield < 1:
            div_yield *= 100


    # ==========================================================================
    # RESULTAT
    # ==========================================================================

    fundamental_info = {

        "trailingPE": pe,

        "pegRatio": peg,

        "earningsGrowth": earnings_growth,

        "revenueGrowth": revenue_growth,

        "debtToEquity": debt_to_equity,

        "currentRatio": current_ratio
    }


    return (
        ticker,
        info,
        fundamental_info,
        div_yield,
        diagnostics
    )


# ==============================================================================
# 5. KOMPLETTER DATENABRUF
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screener_data(tickers):

    data = []
    errors = []
    diagnostics = []

    # --------------------------------------------------------------------------
    # KURSE
    # --------------------------------------------------------------------------

    try:

        hist_all = yf.download(
            tickers=list(tickers),
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True
        )

    except Exception as e:

        return (
            pd.DataFrame(),
            [f"Kursabruf: {e}"],
            pd.DataFrame()
        )


    # --------------------------------------------------------------------------
    # TICKER
    # --------------------------------------------------------------------------

    for symbol in tickers:

        try:

            # ==============================================================
            # HISTORIE
            # ==============================================================

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
                continue


            current_price = safe_float(
                hist["Close"].iloc[-1]
            )


            # ==============================================================
            # FUNDAMENTALDATEN
            # ==============================================================

            (
                ticker,
                raw_info,
                info,
                div_yield,
                diag
            ) = get_fundamental_data(symbol)


            # ==============================================================
            # SCORES
            # ==============================================================

            scores = calculate_individual_scores(
                info,
                hist
            )


            # ==============================================================
            # NAME
            # ==============================================================

            name = raw_info.get(
                "shortName",
                raw_info.get(
                    "longName",
                    symbol
                )
            )


            sector = raw_info.get(
                "sector",
                "N/A"
            )


            # ==============================================================
            # DATAFRAME
            # ==============================================================

            data.append({

                "Symbol": symbol,

                "Name": name,

                "Sektor": sector,

                "Kurs": current_price,

                "KGV (P/E)": info["trailingPE"],

                "PEG": info["pegRatio"],

                "Div. Rendite (%)": div_yield,

                "EPS Wachstum": info["earningsGrowth"],

                "Umsatz Wachstum": info["revenueGrowth"],

                "Debt/Equity": info["debtToEquity"],

                "Current Ratio": info["currentRatio"],

                "Bewertung": scores["Valuation"],

                "Wachstum": scores["Growth"],

                "Gesundheit": scores["Health"],

                "Momentum": scores["Momentum"]
            })


            # ==============================================================
            # DIAGNOSE
            # ==============================================================

            diagnostics.append({

                "Symbol": symbol,

                "Info": diag["Info"],

                "Valuation API": diag["Valuation"],

                "Income Statement": diag["Income"],

                "Balance Sheet": diag["Balance"],

                "KGV": (
                    "OK"
                    if not pd.isna(info["trailingPE"])
                    else "—"
                ),

                "PEG": (
                    "OK"
                    if not pd.isna(info["pegRatio"])
                    else "—"
                ),

                "EPS Growth": (
                    "OK"
                    if not pd.isna(info["earningsGrowth"])
                    else "—"
                ),

                "Revenue Growth": (
                    "OK"
                    if not pd.isna(info["revenueGrowth"])
                    else "—"
                ),

                "D/E": (
                    "OK"
                    if not pd.isna(info["debtToEquity"])
                    else "—"
                ),

                "Current Ratio": (
                    "OK"
                    if not pd.isna(info["currentRatio"])
                    else "—"
                )
            })


        except Exception as e:

            errors.append(
                f"{symbol}: {type(e).__name__}: {e}"
            )


    return (
        pd.DataFrame(data),
        errors,
        pd.DataFrame(diagnostics)
    )


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
    0.0, 1.0, 0.30, 0.05
)

w_gro = st.sidebar.slider(
    "Gewicht Wachstum",
    0.0, 1.0, 0.30, 0.05
)

w_hea = st.sidebar.slider(
    "Gewicht Gesundheit",
    0.0, 1.0, 0.20, 0.05
)

w_mom = st.sidebar.slider(
    "Gewicht Momentum",
    0.0, 1.0, 0.20, 0.05
)

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
# 8. DATEN LADEN
# ==============================================================================

with st.spinner("📡 Lade Yahoo-Finance-Daten..."):

    df, errors, diagnostics = fetch_screener_data(
        tuple(active_tickers)
    )


# ==============================================================================
# 9. FEHLER
# ==============================================================================

if errors:

    with st.expander(
        f"⚠️ Fehler / Hinweise ({len(errors)})"
    ):

        for error in errors:
            st.write("•", error)


# ==============================================================================
# 10. GESAMTSCORE
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
            x[0]
            for x in valid
        ]

        weights = [
            x[1]
            for x in valid
        ]

        return np.average(
            scores,
            weights=weights
        )


    df["Gesamtscore"] = df.apply(
        compute_total_score,
        axis=1
    )


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
    # HAUPTTABELLE
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
                "EPS Wachstum": "{:.1%}",
                "Umsatz Wachstum": "{:.1%}",
                "Debt/Equity": "{:.1f}",
                "Current Ratio": "{:.2f}",
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

        height=550
    )


# ==============================================================================
# 11. DIAGNOSE
# ==============================================================================

st.markdown("---")

st.subheader(
    "🧪 Yahoo-/yfinance-Diagnose"
)

st.write(
    "Diese Tabelle zeigt, ob die verschiedenen Yahoo-Datenquellen "
    "tatsächlich Daten liefern. Damit lässt sich das None-Problem "
    "eindeutig lokalisieren."
)

if not diagnostics.empty:

    st.dataframe(
        diagnostics,
        hide_index=True,
        use_container_width=True
    )


# ==============================================================================
# 12. ROHDATEN-TEST
# ==============================================================================

st.subheader(
    "🔬 Rohdaten-Test"
)

st.caption(
    "Zum Gegencheck wird hier beispielhaft NVDA direkt über yfinance abgefragt."
)

if "NVDA" in active_tickers:

    try:

        test = yf.Ticker("NVDA")

        col1, col2 = st.columns(2)

        with col1:

            st.write("**get_info()**")

            test_info = test.get_info()

            st.write({
                "trailingPE":
                    test_info.get("trailingPE"),

                "pegRatio":
                    test_info.get("pegRatio"),

                "earningsGrowth":
                    test_info.get("earningsGrowth"),

                "revenueGrowth":
                    test_info.get("revenueGrowth"),

                "debtToEquity":
                    test_info.get("debtToEquity"),

                "currentRatio":
                    test_info.get("currentRatio")
            })


        with col2:

            st.write("**Valuation Measures**")

            try:

                test_val = test.get_valuation_measures(
                    freq="trailing",
                    periods=1
                )

                st.dataframe(
                    test_val,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Valuation-Test fehlgeschlagen: {e}"
                )


    except Exception as e:

        st.error(
            f"NVDA-Test fehlgeschlagen: {e}"
        )


# ==============================================================================
# 13. BERECHNUNGSGRUNDLAGEN
# ==============================================================================

st.markdown("---")

st.subheader(
    "📐 Berechnungsgrundlagen"
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
### 📊 Einzelscores

**Bewertung**

* PEG < 1,0 → 100
* PEG 1,0–1,5 → 60
* PEG 1,5–2,0 → 30
* PEG > 2,0 → 5
* KGV < 15 → 100
* KGV 15–25 → 60
* KGV 25–35 → 30
* KGV > 35 → 5

**Wachstum**

* EPS > 20 % → 100
* EPS 10–20 % → 70
* EPS 0–10 % → 40
* EPS < 0 % → 5

* Umsatz > 15 % → 100
* Umsatz 5–15 % → 60
* Umsatz 0–5 % → 30
* Umsatz < 0 % → 5

**Gesundheit**

* D/E < 0,5 → 100
* D/E 0,5–1,5 → 60
* D/E > 1,5 → 10

* Current Ratio > 1,5 → 100
* Current Ratio 1,0–1,5 → 50
* Current Ratio < 1,0 → 5

**Momentum**

* Kurs > GD → 100
* Kurs < GD → 20
* RSI 40–60 → 100
* RSI 30–40 / 60–70 → 70
* sonst → 20
""")


with col2:

    st.markdown("### 🏆 Gesamtscore")

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

Bewertung: **{w_val_n*100:.1f} %**

Wachstum: **{w_gro_n*100:.1f} %**

Gesundheit: **{w_hea_n*100:.1f} %**

Momentum: **{w_mom_n*100:.1f} %**

---

**Anzeige:**

`—` = Daten fehlen

`5` = Daten vorhanden, schlechtester Score

`100` = maximaler Score
"""
    )
