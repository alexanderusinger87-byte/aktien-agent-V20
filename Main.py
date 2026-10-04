import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aktien-Screener V20.4",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

MIN_VALID_SCORE = 5.0
MAX_SCORE = 100.0

INITIAL_SCORE_WEIGHTS = {
    "kpax": 0.45,
    "kpax_fv": 0.45,
    "risk": 0.10
}

INITIAL_KPAX_FV_WEIGHTS = {
    "fair_value": 0.55,
    "relative_valuation": 0.45
}


# ============================================================
# TICKER UNIVERSE
# ============================================================

TICKERS = [
    "SMO",
    "BMW.DE",
    "MAIR",
    "GOOGL",
    "IFX.DE",
    "1810.HK",
    "PEP",
    "KO",
    "NVDA",
    "AAPL",
    "MU",
    "INTC",
    "AMZN",
    "F8P",
    "NOK",
    "VST",
    "ASML",
    "NKE",
    "VRT",
    "TSM",
    "NVO",
    "MRVL",
    "000660.KS",
    "TSLA",
    "005930.KS",
    "AMD",
    "ADS.DE",
    "SU.PA",
    "ENR.DE",
    "SIE.DE",
    "MSFT",
    "AVGO",
    "SSUN.F",
    "MCD",
    "MUV2.DE",
    "ALV.DE"
]


# ============================================================
# FALLBACK SECTORS
# ============================================================

FALLBACK_SECTORS = {
    "Technology": [
        "GOOGL", "NVDA", "AAPL", "MU", "INTC",
        "ASML", "TSM", "MRVL", "000660.KS",
        "005930.KS", "AMD", "MSFT", "AVGO",
        "IFX.DE", "NOK", "VRT", "1810.HK",
        "F8P", "SSUN.F"
    ],
    "Consumer Cyclical": [
        "AMZN", "BMW.DE", "NKE", "TSLA", "MCD", "ADS.DE"
    ],
    "Consumer Defensive": [
        "PEP", "KO"
    ],
    "Healthcare": [
        "NVO"
    ],
    "Energy": [
        "SU.PA"
    ],
    "Utilities": [
        "VST"
    ],
    "Industrials": [
        "SMO", "ENR.DE", "SIE.DE"
    ],
    "Financial Services": [
        "MUV2.DE", "ALV.DE", "MAIR"
    ]
}


# ============================================================
# SESSION STATE
# ============================================================

if "score_weights" not in st.session_state:
    st.session_state.score_weights = INITIAL_SCORE_WEIGHTS.copy()

if "kpax_fv_weights" not in st.session_state:
    st.session_state.kpax_fv_weights = INITIAL_KPAX_FV_WEIGHTS.copy()


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    """
    Converts scalar-like values to float.
    Returns None for invalid / missing data.
    """
    try:
        if value is None:
            return None

        if isinstance(value, (pd.Series, pd.DataFrame)):
            if len(value) == 0:
                return None
            value = value.iloc[0]

        if pd.isna(value):
            return None

        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except Exception:
        return None


def clamp_score(value):
    """
    Valid computed score: 5..100.
    None remains None.
    """
    value = safe_float(value)

    if value is None:
        return None

    return float(np.clip(value, MIN_VALID_SCORE, MAX_SCORE))


def normalize_weights(weights):
    """
    Normalize positive weights to 100%.
    """
    clean = {}

    for key, value in weights.items():
        try:
            v = float(value)
        except Exception:
            v = 0.0

        clean[key] = max(0.0, v)

    total = sum(clean.values())

    if total <= 0:
        return {k: 1.0 / len(clean) for k in clean}

    return {k: v / total for k, v in clean.items()}


def weighted_score(components):
    """
    components = [(score, weight), ...]
    Missing scores are excluded and weights renormalized.
    """
    valid = []

    for score, weight in components:
        score = safe_float(score)
        weight = safe_float(weight)

        if score is not None and weight is not None and weight > 0:
            valid.append((score, weight))

    if not valid:
        return None

    total_weight = sum(w for _, w in valid)

    if total_weight <= 0:
        return None

    value = sum(score * weight for score, weight in valid) / total_weight

    return clamp_score(value)


def get_latest(df, labels):
    """
    Get latest value from a financial statement.
    """
    if df is None or df.empty:
        return None

    for label in labels:
        if label in df.index:
            try:
                series = df.loc[label].dropna()

                if len(series) > 0:
                    return safe_float(series.iloc[0])
            except Exception:
                pass

    return None


def get_series(df, labels):
    """
    Returns a cleaned numerical series from financial statement.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    for label in labels:
        if label in df.index:
            try:
                series = pd.to_numeric(df.loc[label], errors="coerce")
                series = series.replace([np.inf, -np.inf], np.nan).dropna()

                if len(series) > 0:
                    return series
            except Exception:
                pass

    return pd.Series(dtype=float)


def safe_pct_change(current, previous):
    if current is None or previous is None:
        return None

    if previous == 0:
        return None

    return (current / previous - 1.0) * 100.0


def score_positive(value, low, high):
    """
    Linear score where low = 5 and high = 100.
    """
    value = safe_float(value)

    if value is None:
        return None

    if high == low:
        return 50.0

    score = 5 + 95 * ((value - low) / (high - low))

    return clamp_score(score)


def sigmoid_score(value, midpoint, scale, inverse=False):
    """
    Stable sigmoid.

    Important:
    exponent is clipped to avoid math range errors,
    especially for 1810.HK and extreme valuation ratios.
    """
    value = safe_float(value)
    midpoint = safe_float(midpoint)
    scale = safe_float(scale)

    if value is None or midpoint is None or scale is None or scale <= 0:
        return None

    z = (value - midpoint) / scale
    z = float(np.clip(z, -60.0, 60.0))

    try:
        if inverse:
            score = 5 + 95 / (1 + math.exp(z))
        else:
            score = 5 + 95 / (1 + math.exp(-z))
    except Exception:
        return None

    return clamp_score(score)


def median_or_none(values):
    clean = []

    for value in values:
        value = safe_float(value)

        if value is not None:
            clean.append(value)

    if not clean:
        return None

    return float(np.median(clean))


# ============================================================
# DATA EXTRACTION
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def download_ticker(ticker):
    """
    Downloads ticker data.
    """
    try:
        obj = yf.Ticker(ticker)

        info = obj.info

        try:
            income = obj.income_stmt
        except Exception:
            income = pd.DataFrame()

        try:
            cashflow = obj.cashflow
        except Exception:
            cashflow = pd.DataFrame()

        try:
            balance = obj.balance_sheet
        except Exception:
            balance = pd.DataFrame()

        try:
            history = obj.history(
                period="5y",
                interval="1d",
                auto_adjust=False
            )
        except Exception:
            history = pd.DataFrame()

        return {
            "info": info if isinstance(info, dict) else {},
            "income": income,
            "cashflow": cashflow,
            "balance": balance,
            "history": history
        }

    except Exception as e:
        return {
            "info": {},
            "income": pd.DataFrame(),
            "cashflow": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "history": pd.DataFrame(),
            "error": str(e)
        }


# ============================================================
# COMPANY / BASIC DATA
# ============================================================

def get_company_name(data, ticker):
    info = data.get("info", {})

    name = (
        info.get("longName")
        or info.get("shortName")
        or ticker
    )

    return str(name)


def get_sector(data, ticker):
    info = data.get("info", {})

    sector = info.get("sector")

    if sector:
        return sector

    for sector_name, tickers in FALLBACK_SECTORS.items():
        if ticker in tickers:
            return sector_name

    return "Unknown"


def get_price(data):
    info = data.get("info", {})

    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )

    if price is not None:
        return safe_float(price)

    history = data.get("history")

    if history is not None and not history.empty:
        try:
            return safe_float(history["Close"].dropna().iloc[-1])
        except Exception:
            pass

    return None


def get_currency(data):
    info = data.get("info", {})
    return info.get("currency")


def get_market_cap(data):
    info = data.get("info", {})
    return safe_float(info.get("marketCap"))


def get_beta(data):
    info = data.get("info", {})
    return safe_float(info.get("beta"))


# ============================================================
# SHARES
# ============================================================

def get_shares(data):
    info = data.get("info", {})

    candidates = [
        info.get("sharesOutstanding"),
        info.get("impliedSharesOutstanding")
    ]

    for value in candidates:
        value = safe_float(value)

        if value is not None and value > 0:
            return value

    return None


# ============================================================
# EARNINGS / GROWTH
# ============================================================

def get_eps_history(data):
    income = data.get("income")

    series = get_series(
        income,
        [
            "Diluted EPS",
            "Basic EPS"
        ]
    )

    return series


def get_earnings_growth(data):
    info = data.get("info", {})

    candidates = [
        info.get("earningsGrowth"),
        info.get("earningsQuarterlyGrowth")
    ]

    for value in candidates:
        value = safe_float(value)

        if value is not None:
            return value * 100

    eps = get_eps_history(data)

    if len(eps) >= 2:
        current = eps.iloc[0]
        previous = eps.iloc[-1]

        growth = safe_pct_change(current, previous)

        if growth is not None:
            return growth

    return None


def get_revenue_growth(data):
    info = data.get("info", {})

    value = safe_float(info.get("revenueGrowth"))

    if value is not None:
        return value * 100

    income = data.get("income")

    revenue = get_series(
        income,
        [
            "Total Revenue",
            "Operating Revenue"
        ]
    )

    if len(revenue) >= 2:
        return safe_pct_change(
            revenue.iloc[0],
            revenue.iloc[-1]
        )

    return None


def get_forward_eps(data):
    info = data.get("info", {})

    return safe_float(
        info.get("forwardEps")
    )


def get_long_term_growth(data):
    info = data.get("info", {})

    value = safe_float(
        info.get("earningsGrowth")
    )

    if value is not None:
        return value * 100

    return None


def get_forward_earnings_growth(data):
    info = data.get("info", {})

    forward = safe_float(info.get("earningsQuarterlyGrowth"))

    if forward is not None:
        return forward * 100

    return get_earnings_growth(data)


# ============================================================
# FCF
# ============================================================

def get_fcf_series(data):
    """
    Robust annual FCF extraction.

    Priority:
    1. Free Cash Flow
    2. Operating Cash Flow + Capital Expenditure

    Yahoo often does NOT expose a direct "Free Cash Flow"
    row in cashflow. Therefore we reconstruct it.
    """

    cashflow = data.get("cashflow")

    if cashflow is None or cashflow.empty:
        return pd.Series(dtype=float)

    # Direct FCF
    direct = get_series(
        cashflow,
        [
            "Free Cash Flow"
        ]
    )

    if len(direct) > 0:
        return direct

    # Operating Cash Flow
    ocf = get_series(
        cashflow,
        [
            "Operating Cash Flow",
            "Total Cash From Operating Activities"
        ]
    )

    # CapEx
    capex = get_series(
        cashflow,
        [
            "Capital Expenditure",
            "Capital Expenditure Reported"
        ]
    )

    if len(ocf) == 0 or len(capex) == 0:
        return pd.Series(dtype=float)

    # Align by date
    try:
        df = pd.concat(
            [
                ocf.rename("ocf"),
                capex.rename("capex")
            ],
            axis=1
        ).dropna()

        if df.empty:
            return pd.Series(dtype=float)

        # Yahoo usually reports CapEx as negative.
        # If positive, convert to negative.
        df["capex"] = np.where(
            df["capex"] > 0,
            -df["capex"],
            df["capex"]
        )

        fcf = df["ocf"] + df["capex"]

        return fcf.dropna()

    except Exception:
        return pd.Series(dtype=float)


def get_current_fcf(data):
    info = data.get("info", {})

    value = safe_float(
        info.get("freeCashflow")
    )

    if value is not None:
        return value

    series = get_fcf_series(data)

    if len(series) > 0:
        return safe_float(series.iloc[0])

    return None


def get_normalized_fcf(data):
    """
    Uses the median of the latest positive annual FCF values.

    This is deliberately robust against one-off FCF peaks,
    especially for cyclical semiconductor companies.

    Returns:
        normalized_fcf,
        number_of_years,
        all_positive_values
    """

    series = get_fcf_series(data)

    if len(series) == 0:
        current = get_current_fcf(data)

        if current is not None and current > 0:
            return current, 1, [current]

        return None, 0, []

    # Latest three annual observations
    values = []

    for value in series.iloc[:3]:
        value = safe_float(value)

        if value is not None:
            values.append(value)

    if not values:
        return None, 0, []

    # For valuation we need positive sustainable FCF.
    positive = [
        value for value in values
        if value > 0
    ]

    if len(positive) >= 2:
        normalized = float(np.median(positive))
        return normalized, len(positive), positive

    if len(positive) == 1:
        return positive[0], 1, positive

    return None, 0, []


def get_fcf_margin(data):
    info = data.get("info", {})

    direct = safe_float(
        info.get("freeCashflow")
    )

    revenue = get_latest(
        data.get("income"),
        [
            "Total Revenue",
            "Operating Revenue"
        ]
    )

    if direct is not None and revenue is not None and revenue > 0:
        return direct / revenue * 100

    fcf_series = get_fcf_series(data)

    if len(fcf_series) > 0 and revenue is not None and revenue > 0:
        return fcf_series.iloc[0] / revenue * 100

    return None


# ============================================================
# PROFITABILITY
# ============================================================

def get_gross_margin(data):
    info = data.get("info", {})

    value = safe_float(
        info.get("grossMargins")
    )

    if value is not None:
        return value * 100

    income = data.get("income")

    revenue = get_latest(
        income,
        [
            "Total Revenue",
            "Operating Revenue"
        ]
    )

    gross_profit = get_latest(
        income,
        [
            "Gross Profit"
        ]
    )

    if revenue is not None and revenue > 0 and gross_profit is not None:
        return gross_profit / revenue * 100

    return None


def get_net_margin(data):
    info = data.get("info", {})

    value = safe_float(
        info.get("profitMargins")
    )

    if value is not None:
        return value * 100

    income = data.get("income")

    revenue = get_latest(
        income,
        [
            "Total Revenue",
            "Operating Revenue"
        ]
    )

    net_income = get_latest(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders"
        ]
    )

    if revenue is not None and revenue > 0 and net_income is not None:
        return net_income / revenue * 100

    return None


def get_roe(data):
    info = data.get("info", {})

    value = safe_float(
        info.get("returnOnEquity")
    )

    if value is not None:
        return value * 100

    return None


def calculate_roic(data):
    """
    Approximate ROIC:

    NOPAT / Invested Capital

    Invested Capital ≈ Total Debt + Equity - Cash

    Falls back to ROA when required inputs are unavailable.
    """

    income = data.get("income")
    balance = data.get("balance")

    operating_income = get_latest(
        income,
        [
            "Operating Income"
        ]
    )

    total_debt = get_latest(
        balance,
        [
            "Total Debt"
        ]
    )

    equity = get_latest(
        balance,
        [
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
            "Common Stock Equity"
        ]
    )

    cash = get_latest(
        balance,
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents"
        ]
    )

    if (
        operating_income is not None
        and equity is not None
    ):
        if total_debt is None:
            total_debt = 0.0

        if cash is None:
            cash = 0.0

        invested_capital = (
            total_debt
            + equity
            - cash
        )

        if invested_capital > 0:
            tax_expense = get_latest(
                income,
                [
                    "Tax Provision"
                ]
            )

            pretax_income = get_latest(
                income,
                [
                    "Pretax Income"
                ]
            )

            if (
                tax_expense is not None
                and pretax_income is not None
                and pretax_income > 0
            ):
                tax_rate = abs(tax_expense) / pretax_income
                tax_rate = float(
                    np.clip(tax_rate, 0, 0.35)
                )
            else:
                tax_rate = 0.21

            nopat = operating_income * (1 - tax_rate)

            return nopat / invested_capital * 100

    # ROA fallback
    roa = get_latest(
        income,
        [
            "Net Income"
        ]
    )

    assets = get_latest(
        balance,
        [
            "Total Assets"
        ]
    )

    if roa is not None and assets is not None and assets > 0:
        return roa / assets * 100

    return None


# ============================================================
# DEBT / BALANCE SHEET
# ============================================================

def get_total_debt(data):
    return get_latest(
        data.get("balance"),
        [
            "Total Debt"
        ]
    )


def get_cash(data):
    return get_latest(
        data.get("balance"),
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents"
        ]
    )


def get_equity(data):
    return get_latest(
        data.get("balance"),
        [
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
            "Common Stock Equity"
        ]
    )


def get_debt_equity(data):
    info = data.get("info", {})

    value = safe_float(
        info.get("debtToEquity")
    )

    if value is not None:
        return value / 100

    debt = get_total_debt(data)
    equity = get_equity(data)

    if debt is not None and equity is not None and equity > 0:
        return debt / equity

    return None


def get_cash_debt_ratio(data):
    debt = get_total_debt(data)
    cash = get_cash(data)

    if debt is not None and debt > 0 and cash is not None:
        return cash / debt

    return None


def get_current_ratio(data):
    info = data.get("info", {})

    return safe_float(
        info.get("currentRatio")
    )


# ============================================================
# VALUATION DATA
# ============================================================

def get_forward_pe(data):
    info = data.get("info", {})

    value = safe_float(
        info.get("forwardPE")
    )

    if value is not None and value > 0:
        return value

    price = get_price(data)
    eps = get_forward_eps(data)

    if (
        price is not None
        and eps is not None
        and eps > 0
    ):
        return price / eps

    return None


def get_trailing_pe(data):
    info = data.get("info", {})

    value = safe_float(
        info.get("trailingPE")
    )

    if value is not None and value > 0:
        return value

    return None


def get_peg(data):
    info = data.get("info", {})

    value = safe_float(
        info.get("pegRatio")
    )

    if value is not None and value > 0:
        return value

    return None


def get_price_sales(data):
    info = data.get("info", {})

    value = safe_float(
        info.get("priceToSalesTrailing12Months")
    )

    if value is not None and value > 0:
        return value

    return None


def get_price_book(data):
    info = data.get("info", {})

    value = safe_float(
        info.get("priceToBook")
    )

    if value is not None and value > 0:
        return value

    return None


# ============================================================
# ANALYST TARGET
# ============================================================

def get_analyst_target(data):
    info = data.get("info", {})

    candidates = [
        info.get("targetMeanPrice"),
        info.get("targetMedianPrice")
    ]

    values = []

    for value in candidates:
        value = safe_float(value)

        if value is not None and value > 0:
            values.append(value)

    if not values:
        return None

    return float(np.median(values))


# ============================================================
# FAIR VALUE
# ============================================================

def fair_value_model(data):
    """
    V20.4 robust Fair Value model.

    Earnings FV:
        Forward EPS × Fair P/E

    Fair P/E:
        12 + 0.25 × growth
        bounded 12..24

    FCF FV:
        normalized FCF/share / target yield

    Analyst FV:
        analyst consensus target

    Important V20.4 changes:
        - normalized 3Y FCF
        - model-level price bounds
        - robust median reference
        - final FV cap 0.40x..2.50x price
    """

    price = get_price(data)

    if price is None or price <= 0:
        return {
            "fair_value": None,
            "fair_value_score": None,
            "earnings_fv": None,
            "fcf_fv": None,
            "analyst_fv": None,
            "normalized_fcf": None,
            "fcf_years": 0,
            "model_count": 0,
            "model_confidence": 0
        }

    growth = get_long_term_growth(data)

    if growth is None:
        growth = get_earnings_growth(data)

    if growth is None:
        growth = 8.0

    growth = float(np.clip(growth, -10, 50))

    # --------------------------------------------------------
    # 1. Earnings FV
    # --------------------------------------------------------

    forward_eps = get_forward_eps(data)

    earnings_fv = None

    if (
        forward_eps is not None
        and forward_eps > 0
    ):
        fair_pe = 12 + 0.25 * growth
        fair_pe = float(np.clip(fair_pe, 12, 24))

        earnings_fv = forward_eps * fair_pe

    # --------------------------------------------------------
    # 2. Normalized FCF FV
    # --------------------------------------------------------

    normalized_fcf, fcf_years, _ = get_normalized_fcf(data)

    shares = get_shares(data)

    fcf_fv = None

    if (
        normalized_fcf is not None
        and normalized_fcf > 0
        and shares is not None
        and shares > 0
    ):
        fcf_per_share = normalized_fcf / shares

        target_yield = (
            0.070
            - 0.0008 * growth
        )

        target_yield = float(
            np.clip(
                target_yield,
                0.045,
                0.090
            )
        )

        beta = get_beta(data)

        if beta is not None and beta > 1:
            target_yield += min(
                (beta - 1) * 0.005,
                0.02
            )

        target_yield = float(
            np.clip(
                target_yield,
                0.045,
                0.110
            )
        )

        fcf_fv = fcf_per_share / target_yield

    # --------------------------------------------------------
    # 3. Analyst FV
    # --------------------------------------------------------

    analyst_fv = get_analyst_target(data)

    # --------------------------------------------------------
    # Model-level bounds
    # --------------------------------------------------------

    raw_models = [
        earnings_fv,
        fcf_fv,
        analyst_fv
    ]

    bounded_models = []

    for model in raw_models:
        if model is None:
            bounded_models.append(None)
            continue

        model = float(
            np.clip(
                model,
                price * 0.25,
                price * 2.50
            )
        )

        bounded_models.append(model)

    earnings_fv = bounded_models[0]
    fcf_fv = bounded_models[1]
    analyst_fv = bounded_models[2]

    # --------------------------------------------------------
    # Robust median reference
    # --------------------------------------------------------

    valid_models = [
        x for x in bounded_models
        if x is not None and x > 0
    ]

    if not valid_models:
        return {
            "fair_value": None,
            "fair_value_score": None,
            "earnings_fv": None,
            "fcf_fv": None,
            "analyst_fv": None,
            "normalized_fcf": normalized_fcf,
            "fcf_years": fcf_years,
            "model_count": 0,
            "model_confidence": 0
        }

    median_model = float(
        np.median(valid_models)
    )

    # --------------------------------------------------------
    # Robust outlier bounding around median
    # --------------------------------------------------------

    robust_models = []

    for model in valid_models:
        bounded = float(
            np.clip(
                model,
                median_model * 0.50,
                median_model * 1.80
            )
        )

        robust_models.append(bounded)

    # --------------------------------------------------------
    # Financial vs non-financial weighting
    # --------------------------------------------------------

    sector = get_sector(
        data,
        ""
    )

    is_financial = (
        sector == "Financial Services"
    )

    if is_financial:
        weights = {
            "earnings": 0.35,
            "fcf": 0.15,
            "analyst": 0.50
        }
    else:
        weights = {
            "earnings": 0.45,
            "fcf": 0.25,
            "analyst": 0.30
        }

    weighted_parts = []

    if earnings_fv is not None:
        value = float(
            np.clip(
                earnings_fv,
                median_model * 0.50,
                median_model * 1.80
            )
        )
        weighted_parts.append(
            (value, weights["earnings"])
        )

    if fcf_fv is not None:
        value = float(
            np.clip(
                fcf_fv,
                median_model * 0.50,
                median_model * 1.80
            )
        )
        weighted_parts.append(
            (value, weights["fcf"])
        )

    if analyst_fv is not None:
        value = float(
            np.clip(
                analyst_fv,
                median_model * 0.50,
                median_model * 1.80
            )
        )
        weighted_parts.append(
            (value, weights["analyst"])
        )

    if not weighted_parts:
        return {
            "fair_value": None,
            "fair_value_score": None,
            "earnings_fv": earnings_fv,
            "fcf_fv": fcf_fv,
            "analyst_fv": analyst_fv,
            "normalized_fcf": normalized_fcf,
            "fcf_years": fcf_years,
            "model_count": 0,
            "model_confidence": 0
        }

    total_weight = sum(
        weight for _, weight in weighted_parts
    )

    fair_value = (
        sum(
            value * weight
            for value, weight in weighted_parts
        )
        / total_weight
    )

    # --------------------------------------------------------
    # Final hard price bound
    # --------------------------------------------------------

    fair_value = float(
        np.clip(
            fair_value,
            price * 0.40,
            price * 2.50
        )
    )

    # --------------------------------------------------------
    # Fair Value Score
    # --------------------------------------------------------

    upside_pct = (
        fair_value / price - 1
    ) * 100

    # Softer V20.4 slope
    fair_value_score = clamp_score(
        55 + 0.60 * upside_pct
    )

    # --------------------------------------------------------
    # Model confidence
    # --------------------------------------------------------

    model_count = len(valid_models)

    confidence = {
        1: 45,
        2: 75,
        3: 100
    }.get(model_count, 0)

    if fcf_years >= 2:
        confidence += 5

    confidence = int(
        np.clip(confidence, 0, 100)
    )

    return {
        "fair_value": fair_value,
        "fair_value_score": fair_value_score,
        "earnings_fv": earnings_fv,
        "fcf_fv": fcf_fv,
        "analyst_fv": analyst_fv,
        "normalized_fcf": normalized_fcf,
        "fcf_years": fcf_years,
        "model_count": model_count,
        "model_confidence": confidence
    }


# ============================================================
# RELATIVE VALUATION
# ============================================================

def relative_valuation(data):
    """
    V20.4 Relative Valuation.

    Non-financial:
        Forward PE 35%
        Trailing PE 10%
        PEG 30%
        P/S 25%

    Financial:
        Forward PE 35%
        Trailing PE 15%
        PEG 15%
        P/B 35%

    Smooth sigmoid scoring avoids hard thresholds.
    """

    sector = get_sector(
        data,
        ""
    )

    is_financial = (
        sector == "Financial Services"
    )

    forward_pe = get_forward_pe(data)
    trailing_pe = get_trailing_pe(data)
    peg = get_peg(data)
    ps = get_price_sales(data)
    pb = get_price_book(data)

    components = []

    # Forward PE
    if forward_pe is not None and forward_pe > 0:
        components.append(
            (
                sigmoid_score(
                    forward_pe,
                    22,
                    8,
                    inverse=True
                ),
                0.35
            )
        )

    # Trailing PE
    if trailing_pe is not None and trailing_pe > 0:
        components.append(
            (
                sigmoid_score(
                    trailing_pe,
                    25,
                    10,
                    inverse=True
                ),
                0.15 if is_financial else 0.10
            )
        )

    # PEG
    if peg is not None and peg > 0:
        components.append(
            (
                sigmoid_score(
                    peg,
                    1.8,
                    0.8,
                    inverse=True
                ),
                0.15 if is_financial else 0.30
            )
        )

    if is_financial:
        if pb is not None and pb > 0:
            components.append(
                (
                    sigmoid_score(
                        pb,
                        1.8,
                        1.0,
                        inverse=True
                    ),
                    0.35
                )
            )
    else:
        if ps is not None and ps > 0:
            components.append(
                (
                    sigmoid_score(
                        ps,
                        4.0,
                        2.5,
                        inverse=True
                    ),
                    0.25
                )
            )

    scores = [
        (score, weight)
        for score, weight in components
        if score is not None
    ]

    if not scores:
        return None

    return weighted_score(scores)


# ============================================================
# QUALITY SCORE
# ============================================================

def quality_score(data):
    sector = get_sector(
        data,
        ""
    )

    is_financial = (
        sector == "Financial Services"
    )

    earnings_growth = get_earnings_growth(data)
    revenue_growth = get_revenue_growth(data)
    gross_margin = get_gross_margin(data)
    fcf_margin = get_fcf_margin(data)
    roic = calculate_roic(data)
    roe = get_roe(data)
    net_margin = get_net_margin(data)

    components = []

    if is_financial:

        if roe is not None:
            components.append(
                (
                    score_positive(
                        roe,
                        0,
                        25
                    ),
                    0.40
                )
            )

        if net_margin is not None:
            components.append(
                (
                    score_positive(
                        net_margin,
                        0,
                        30
                    ),
                    0.30
                )
            )

        if earnings_growth is not None:
            components.append(
                (
                    score_positive(
                        earnings_growth,
                        -10,
                        30
                    ),
                    0.20
                )
            )

        if revenue_growth is not None:
            components.append(
                (
                    score_positive(
                        revenue_growth,
                        -5,
                        20
                    ),
                    0.10
                )
            )

    else:

        if roic is not None:
            components.append(
                (
                    score_positive(
                        roic,
                        0,
                        25
                    ),
                    0.30
                )
            )

        if gross_margin is not None:
            components.append(
                (
                    score_positive(
                        gross_margin,
                        10,
                        70
                    ),
                    0.15
                )
            )

        if fcf_margin is not None:
            components.append(
                (
                    score_positive(
                        fcf_margin,
                        -5,
                        30
                    ),
                    0.20
                )
            )

        if earnings_growth is not None:
            components.append(
                (
                    score_positive(
                        earnings_growth,
                        -10,
                        35
                    ),
                    0.20
                )
            )

        if revenue_growth is not None:
            components.append(
                (
                    score_positive(
                        revenue_growth,
                        -5,
                        25
                    ),
                    0.15
                )
            )

    return weighted_score(components)


# ============================================================
# FUTURE SCORE
# ============================================================

def future_score(data):
    earnings_growth = get_earnings_growth(data)
    revenue_growth = get_revenue_growth(data)
    forward_growth = get_forward_earnings_growth(data)
    long_term_growth = get_long_term_growth(data)
    fcf_margin = get_fcf_margin(data)

    acceleration = None

    if (
        forward_growth is not None
        and earnings_growth is not None
    ):
        acceleration = (
            forward_growth
            - earnings_growth
        )

    components = []

    if earnings_growth is not None:
        components.append(
            (
                score_positive(
                    earnings_growth,
                    -10,
                    35
                ),
                0.25
            )
        )

    if revenue_growth is not None:
        components.append(
            (
                score_positive(
                    revenue_growth,
                    -5,
                    25
                ),
                0.20
            )
        )

    if forward_growth is not None:
        components.append(
            (
                score_positive(
                    forward_growth,
                    -10,
                    35
                ),
                0.20
            )
        )

    if long_term_growth is not None:
        components.append(
            (
                score_positive(
                    long_term_growth,
                    -5,
                    30
                ),
                0.15
            )
        )

    if fcf_margin is not None:
        components.append(
            (
                score_positive(
                    fcf_margin,
                    -5,
                    30
                ),
                0.10
            )
        )

    if acceleration is not None:
        components.append(
            (
                score_positive(
                    acceleration,
                    -20,
                    20
                ),
                0.10
            )
        )

    score = weighted_score(components)

    return score


# ============================================================
# KPAX
# ============================================================

def kpax_score(data):
    quality = quality_score(data)
    future = future_score(data)

    return weighted_score(
        [
            (quality, 0.40),
            (future, 0.60)
        ]
    )


# ============================================================
# KPAX-FV
# ============================================================

def kpax_fv_score(fair_value_score, relative_score):
    weights = normalize_weights(
        st.session_state.kpax_fv_weights
    )

    return weighted_score(
        [
            (
                fair_value_score,
                weights["fair_value"]
            ),
            (
                relative_score,
                weights["relative_valuation"]
            )
        ]
    )


# ============================================================
# RISK
# ============================================================

def calculate_volatility(history):
    if history is None or history.empty:
        return None

    try:
        returns = (
            history["Close"]
            .pct_change()
            .dropna()
        )

        if len(returns) < 30:
            return None

        volatility = (
            returns.std()
            * math.sqrt(252)
            * 100
        )

        return volatility

    except Exception:
        return None


def calculate_max_drawdown(history):
    if history is None or history.empty:
        return None

    try:
        close = history["Close"].dropna()

        if len(close) < 30:
            return None

        running_max = close.cummax()

        drawdown = (
            close / running_max - 1
        )

        return abs(drawdown.min()) * 100

    except Exception:
        return None


def risk_score(data):
    history = data.get("history")

    volatility = calculate_volatility(history)
    drawdown = calculate_max_drawdown(history)

    beta = get_beta(data)

    market_components = []

    if volatility is not None:
        market_components.append(
            (
                sigmoid_score(
                    volatility,
                    30,
                    15,
                    inverse=True
                ),
                0.55
            )
        )

    if drawdown is not None:
        market_components.append(
            (
                sigmoid_score(
                    drawdown,
                    45,
                    20,
                    inverse=True
                ),
                0.45
            )
        )

    market_score = weighted_score(
        market_components
    )

    if market_score is None and beta is not None:
        market_score = sigmoid_score(
            beta,
            1.2,
            0.5,
            inverse=True
        )

    debt_equity = get_debt_equity(data)
    cash_debt = get_cash_debt_ratio(data)

    stability_components = []

    if debt_equity is not None:
        stability_components.append(
            (
                sigmoid_score(
                    debt_equity,
                    1.0,
                    0.8,
                    inverse=True
                ),
                0.65
            )
        )

    if cash_debt is not None:
        stability_components.append(
            (
                sigmoid_score(
                    cash_debt,
                    0.5,
                    0.5,
                    inverse=False
                ),
                0.35
            )
        )

    stability_score = weighted_score(
        stability_components
    )

    financial_score = stability_score

    if financial_score is None:
        financial_score = 50

    if market_score is None:
        market_score = 50

    if stability_score is None:
        stability_score = 50

    return weighted_score(
        [
            (financial_score, 0.45),
            (market_score, 0.35),
            (stability_score, 0.20)
        ]
    )


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_score(data):
    history = data.get("history")

    if history is None or history.empty:
        return None

    try:
        close = history["Close"].dropna()

        if len(close) < 200:
            return None

        price = close.iloc[-1]

        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1]

        score = 50

        if price > sma50:
            score += 15
        else:
            score -= 15

        if price > sma200:
            score += 20
        else:
            score -= 20

        if sma50 > sma200:
            score += 15
        else:
            score -= 15

        return clamp_score(score)

    except Exception:
        return None


# ============================================================
# INVESTMENT SCORE
# ============================================================

def investment_score(kpax, kpax_fv, risk):
    weights = normalize_weights(
        st.session_state.score_weights
    )

    return weighted_score(
        [
            (
                kpax,
                weights["kpax"]
            ),
            (
                kpax_fv,
                weights["kpax_fv"]
            ),
            (
                risk,
                weights["risk"]
            )
        ]
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def base_recommendation(score):
    if score is None:
        return "—"

    if score >= 85:
        return "Strong Buy"

    if score >= 80:
        return "Buy"

    if score >= 75:
        return "Accumulate"

    if score >= 68:
        return "Hold"

    if score >= 55:
        return "Reduce / Watch"

    return "Avoid"


def apply_technical_filter(recommendation, technical):
    """
    Technical is NOT part of Investment Score.

    It only limits the maximum recommendation.
    """

    if recommendation == "—" or technical is None:
        return recommendation

    levels = [
        "Avoid",
        "Reduce / Watch",
        "Hold",
        "Accumulate",
        "Buy",
        "Strong Buy"
    ]

    current_index = levels.index(
        recommendation
    )

    if technical < 30:
        max_level = "Hold"

    elif technical < 45:
        max_level = "Accumulate"

    elif technical < 60:
        max_level = "Buy"

    else:
        max_level = "Strong Buy"

    max_index = levels.index(
        max_level
    )

    return levels[
        min(current_index, max_index)
    ]


# ============================================================
# VERIFY
# ============================================================

def verify_score(data):
    """
    Independent cross-check.

    Growth 25%
    Profitability 25%
    Debt 15%
    Valuation 35%
    """

    sector = get_sector(
        data,
        ""
    )

    is_financial = (
        sector == "Financial Services"
    )

    # --------------------------------------------------------
    # Growth
    # --------------------------------------------------------

    earnings_growth = get_earnings_growth(data)
    revenue_growth = get_revenue_growth(data)

    growth_components = []

    if earnings_growth is not None:
        growth_components.append(
            (
                score_positive(
                    earnings_growth,
                    -10,
                    35
                ),
                0.60
            )
        )

    if revenue_growth is not None:
        growth_components.append(
            (
                score_positive(
                    revenue_growth,
                    -5,
                    25
                ),
                0.40
            )
        )

    growth = weighted_score(
        growth_components
    )

    # --------------------------------------------------------
    # Profitability
    # --------------------------------------------------------

    profitability_components = []

    if is_financial:

        roe = get_roe(data)
        net_margin = get_net_margin(data)

        if roe is not None:
            profitability_components.append(
                (
                    score_positive(
                        roe,
                        0,
                        25
                    ),
                    0.60
                )
            )

        if net_margin is not None:
            profitability_components.append(
                (
                    score_positive(
                        net_margin,
                        0,
                        30
                    ),
                    0.40
                )
            )

    else:

        roic = calculate_roic(data)
        fcf_margin = get_fcf_margin(data)
        gross_margin = get_gross_margin(data)

        if roic is not None:
            profitability_components.append(
                (
                    score_positive(
                        roic,
                        0,
                        25
                    ),
                    0.50
                )
            )

        if fcf_margin is not None:
            profitability_components.append(
                (
                    score_positive(
                        fcf_margin,
                        -5,
                        30
                    ),
                    0.30
                )
            )

        if gross_margin is not None:
            profitability_components.append(
                (
                    score_positive(
                        gross_margin,
                        10,
                        70
                    ),
                    0.20
                )
            )

    profitability = weighted_score(
        profitability_components
    )

    # --------------------------------------------------------
    # Debt
    # --------------------------------------------------------

    debt_components = []

    debt_equity = get_debt_equity(data)
    cash_debt = get_cash_debt_ratio(data)
    current_ratio = get_current_ratio(data)

    if debt_equity is not None:
        debt_components.append(
            (
                sigmoid_score(
                    debt_equity,
                    1.0,
                    0.8,
                    inverse=True
                ),
                0.60
            )
        )

    if cash_debt is not None:
        debt_components.append(
            (
                sigmoid_score(
                    cash_debt,
                    0.5,
                    0.5,
                    inverse=False
                ),
                0.25 if not is_financial else 0.40
            )
        )

    if (
        not is_financial
        and current_ratio is not None
    ):
        debt_components.append(
            (
                sigmoid_score(
                    current_ratio,
                    1.5,
                    0.8,
                    inverse=False
                ),
                0.15
            )
        )

    debt = weighted_score(
        debt_components
    )

    # --------------------------------------------------------
    # Valuation
    # --------------------------------------------------------

    valuation_components = []

    forward_pe = get_forward_pe(data)
    peg = get_peg(data)
    ps = get_price_sales(data)
    pb = get_price_book(data)

    if forward_pe is not None:
        valuation_components.append(
            (
                sigmoid_score(
                    forward_pe,
                    22,
                    8,
                    inverse=True
                ),
                0.60 if is_financial else 0.50
            )
        )

    if is_financial:

        if pb is not None:
            valuation_components.append(
                (
                    sigmoid_score(
                        pb,
                        1.8,
                        1.0,
                        inverse=True
                    ),
                    0.40
                )
            )

    else:

        if peg is not None:
            valuation_components.append(
                (
                    sigmoid_score(
                        peg,
                        1.8,
                        0.8,
                        inverse=True
                    ),
                    0.30
                )
            )

        if ps is not None:
            valuation_components.append(
                (
                    sigmoid_score(
                        ps,
                        4.0,
                        2.5,
                        inverse=True
                    ),
                    0.20
                )
            )

    valuation = weighted_score(
        valuation_components
    )

    # --------------------------------------------------------
    # Final Verify
    # --------------------------------------------------------

    components = [
        (growth, 0.25),
        (profitability, 0.25),
        (debt, 0.15),
        (valuation, 0.35)
    ]

    final = weighted_score(
        components
    )

    available = sum(
        1
        for score, _ in components
        if score is not None
    )

    data_completeness = (
        available / 4 * 100
    )

    return {
        "verify": final,
        "verify_data": data_completeness,
        "verify_growth": growth,
        "verify_profitability": profitability,
        "verify_debt": debt,
        "verify_valuation": valuation
    }


# ============================================================
# MODEL CONFIDENCE
# ============================================================

def model_confidence(
    kpax,
    kpax_fv,
    risk,
    fair_value_data,
    verify_data,
    history
):
    component_count = sum(
        x is not None
        for x in [
            kpax,
            kpax_fv,
            risk
        ]
    )

    component_score = (
        component_count / 3 * 100
    )

    fv_confidence = fair_value_data.get(
        "model_confidence",
        0
    )

    history_score = 0

    if history is not None:
        try:
            days = len(history)

            if days >= 700:
                history_score = 100
            elif days >= 400:
                history_score = 80
            elif days >= 200:
                history_score = 60
            elif days >= 100:
                history_score = 40
        except Exception:
            history_score = 0

    return round(
        0.30 * component_score
        + 0.35 * fv_confidence
        + 0.20 * verify_data
        + 0.15 * history_score,
        1
    )


# ============================================================
# VERIFY STATUS
# ============================================================

def verify_status(investment, verify, verify_data):
    if (
        investment is None
        or verify is None
        or verify_data is None
    ):
        return "weak"

    if verify_data < 60:
        return "weak"

    gap = investment - verify

    if abs(gap) <= 10:
        return "green"

    if abs(gap) <= 20:
        return "yellow"

    return "red"


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_ticker(ticker):
    data = download_ticker(ticker)

    info = data.get("info", {})

    if not info:
        return {
            "Ticker": ticker,
            "Company": ticker,
            "Price": None,
            "Investment": None,
            "Recommendation": "—",
            "error": data.get("error", "No data")
        }

    price = get_price(data)

    if price is None:
        return {
            "Ticker": ticker,
            "Company": get_company_name(data, ticker),
            "Price": None,
            "Investment": None,
            "Recommendation": "—",
            "error": "No price"
        }

    sector = get_sector(
        data,
        ticker
    )

    # --------------------------------------------------------
    # Main scores
    # --------------------------------------------------------

    quality = quality_score(data)
    future = future_score(data)

    kpax = weighted_score(
        [
            (quality, 0.40),
            (future, 0.60)
        ]
    )

    fv = fair_value_model(data)

    fair_value_score = fv[
        "fair_value_score"
    ]

    relative = relative_valuation(data)

    kpax_fv = kpax_fv_score(
        fair_value_score,
        relative
    )

    risk = risk_score(data)

    investment = investment_score(
        kpax,
        kpax_fv,
        risk
    )

    technical = technical_score(data)

    recommendation = base_recommendation(
        investment
    )

    recommendation = apply_technical_filter(
        recommendation,
        technical
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify = verify_score(data)

    verify_value = verify[
        "verify"
    ]

    gap = None

    if (
        investment is not None
        and verify_value is not None
    ):
        gap = investment - verify_value

    status = verify_status(
        investment,
        verify_value,
        verify["verify_data"]
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = model_confidence(
        kpax,
        kpax_fv,
        risk,
        fv,
        verify["verify_data"],
        data.get("history")
    )

    # --------------------------------------------------------
    # Fair Value upside
    # --------------------------------------------------------

    fair_value = fv["fair_value"]

    upside = None

    if fair_value is not None:
        upside = (
            fair_value / price - 1
        ) * 100

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "Ticker": ticker,
        "Company": get_company_name(
            data,
            ticker
        ),
        "Sector": sector,
        "Currency": get_currency(data),

        "Price": price,
        "Fair Value": fair_value,
        "FV Upside %": upside,

        "Fair Value Score": fair_value_score,
        "Relative Valuation": relative,

        "KPAX": kpax,
        "KPAX-FV": kpax_fv,
        "Risk": risk,
        "Technical": technical,
        "Investment": investment,

        "Recommendation": recommendation,

        "Quality": quality,
        "Future": future,

        "Earnings FV": fv["earnings_fv"],
        "FCF FV": fv["fcf_fv"],
        "Analyst FV": fv["analyst_fv"],

        "Normalized FCF": fv[
            "normalized_fcf"
        ],
        "FCF Years": fv[
            "fcf_years"
        ],
        "FV Models": fv[
            "model_count"
        ],
        "FV Confidence": fv[
            "model_confidence"
        ],

        "Model Confidence": confidence,

        "Verify": verify_value,
        "Verify Gap": gap,
        "Verify Status": status,
        "Verify Data": verify[
            "verify_data"
        ],
        "Verify Growth": verify[
            "verify_growth"
        ],
        "Verify Profitability": verify[
            "verify_profitability"
        ],
        "Verify Debt": verify[
            "verify_debt"
        ],
        "Verify Valuation": verify[
            "verify_valuation"
        ]
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_score(value):
    if value is None:
        return "—"

    return f"{value:.1f}"


def format_price(value):
    if value is None:
        return "—"

    return f"{value:,.2f}"


def format_pct(value):
    if value is None:
        return "—"

    return f"{value:+.1f}%"


def style_recommendation(value):
    if value == "Strong Buy":
        return "🟢 Strong Buy"

    if value == "Buy":
        return "🟢 Buy"

    if value == "Accumulate":
        return "🟢 Accumulate"

    if value == "Hold":
        return "🟡 Hold"

    if value == "Reduce / Watch":
        return "🟠 Reduce / Watch"

    if value == "Avoid":
        return "🔴 Avoid"

    return "—"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Modell")

st.sidebar.subheader(
    "Investment Score Gewichtung"
)

col1, col2, col3 = st.sidebar.columns(3)

with col1:
    kpax_weight = st.number_input(
        "KPAX %",
        min_value=0.0,
        max_value=100.0,
        value=45.0,
        step=5.0
    )

with col2:
    fv_weight = st.number_input(
        "KPAX-FV %",
        min_value=0.0,
        max_value=100.0,
        value=45.0,
        step=5.0
    )

with col3:
    risk_weight = st.number_input(
        "Risk %",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=5.0
    )

st.session_state.score_weights = {
    "kpax": kpax_weight,
    "kpax_fv": fv_weight,
    "risk": risk_weight
}

st.sidebar.caption(
    f"Summe: "
    f"{kpax_weight + fv_weight + risk_weight:.0f}% "
    "(wird automatisch normalisiert)"
)


st.sidebar.subheader(
    "KPAX-FV Gewichtung"
)

fv_part = st.sidebar.number_input(
    "Fair Value %",
    min_value=0.0,
    max_value=100.0,
    value=55.0,
    step=5.0
)

rel_part = st.sidebar.number_input(
    "Relative Valuation %",
    min_value=0.0,
    max_value=100.0,
    value=45.0,
    step=5.0
)

st.session_state.kpax_fv_weights = {
    "fair_value": fv_part,
    "relative_valuation": rel_part
}


if st.sidebar.button(
    "🔄 Reset Gewichte",
    use_container_width=True
):
    st.session_state.score_weights = (
        INITIAL_SCORE_WEIGHTS.copy()
    )

    st.session_state.kpax_fv_weights = (
        INITIAL_KPAX_FV_WEIGHTS.copy()
    )

    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Aktien-Screener V20.4"
)

st.caption(
    "KPAX + robuste Fair-Value-Bewertung + "
    "Relative Valuation + unabhängiger Verify-Check"
)


# ============================================================
# RUN
# ============================================================

run = st.button(
    "🚀 Screener starten",
    type="primary",
    use_container_width=True
)


if run:

    progress = st.progress(0)

    results = []

    total = len(TICKERS)

    for i, ticker in enumerate(TICKERS):

        try:
            result = analyze_ticker(
                ticker
            )

            results.append(result)

        except Exception as e:

            results.append(
                {
                    "Ticker": ticker,
                    "Company": ticker,
                    "Price": None,
                    "Investment": None,
                    "Recommendation": "—",
                    "error": str(e)
                }
            )

        progress.progress(
            int(
                (i + 1)
                / total
                * 100
            )
        )

        time.sleep(0.03)

    progress.empty()

    df = pd.DataFrame(results)

    if not df.empty:

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        df = df.sort_values(
            by="Investment",
            ascending=False,
            na_position="last"
        )

        # ----------------------------------------------------
        # Main table
        # ----------------------------------------------------

        st.subheader(
            "📈 Investment Ranking"
        )

        display = df.copy()

        display["Price"] = display[
            "Price"
        ].apply(format_price)

        display["Fair Value"] = display[
            "Fair Value"
        ].apply(format_price)

        display["FV Upside %"] = display[
            "FV Upside %"
        ].apply(format_pct)

        score_columns = [
            "Fair Value Score",
            "Relative Valuation",
            "KPAX",
            "KPAX-FV",
            "Risk",
            "Technical",
            "Investment",
            "Verify",
            "Verify Gap",
            "Verify Data",
            "Model Confidence"
        ]

        for col in score_columns:
            if col in display.columns:
                display[col] = display[
                    col
                ].apply(format_score)

        display[
            "Recommendation"
        ] = display[
            "Recommendation"
        ].apply(
            style_recommendation
        )

        main_columns = [
            "Ticker",
            "Company",
            "Price",
            "Fair Value",
            "FV Upside %",
            "Fair Value Score",
            "Relative Valuation",
            "KPAX",
            "KPAX-FV",
            "Risk",
            "Technical",
            "Investment",
            "Recommendation",
            "Verify",
            "Verify Gap",
            "Verify Status",
            "Model Confidence"
        ]

        main_columns = [
            c for c in main_columns
            if c in display.columns
        ]

        st.dataframe(
            display[main_columns],
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Fair Value diagnostics
        # ----------------------------------------------------

        st.subheader(
            "💰 Fair Value Detail"
        )

        fv_display = df.copy()

        fv_columns = [
            "Ticker",
            "Company",
            "Price",
            "Fair Value",
            "FV Upside %",
            "Earnings FV",
            "FCF FV",
            "Analyst FV",
            "Normalized FCF",
            "FCF Years",
            "FV Models",
            "FV Confidence"
        ]

        fv_columns = [
            c for c in fv_columns
            if c in fv_display.columns
        ]

        st.dataframe(
            fv_display[fv_columns],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "V20.4: FCF wird aus bis zu drei "
            "Geschäftsjahren normalisiert. Einzelne "
            "extreme FCF-Spitzen dominieren den Fair Value "
            "damit nicht mehr."
        )

        # ----------------------------------------------------
        # Verify
        # ----------------------------------------------------

        st.subheader(
            "🔎 Independent Verify"
        )

        verify_display = df.copy()

        verify_columns = [
            "Ticker",
            "Investment",
            "Verify",
            "Verify Gap",
            "Verify Status",
            "Verify Data",
            "Verify Growth",
            "Verify Profitability",
            "Verify Debt",
            "Verify Valuation"
        ]

        verify_columns = [
            c for c in verify_columns
            if c in verify_display.columns
        ]

        for col in verify_columns:
            if col not in [
                "Ticker",
                "Verify Status"
            ]:
                verify_display[col] = (
                    verify_display[col]
                    .apply(format_score)
                )

        st.dataframe(
            verify_display[
                verify_columns
            ],
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Formula section
        # ----------------------------------------------------

        st.subheader(
            "🧮 Mathematik des Modells"
        )

        st.markdown(
            """
### 1. KPAX

**KPAX = 40 % Quality + 60 % Future**

---

### 2. Quality

Nicht-finanzielle Unternehmen:

**Quality = 30 % ROIC + 15 % Gross Margin + 20 % FCF Margin + 20 % Earnings Growth + 15 % Revenue Growth**

Finanzunternehmen:

**Quality = 40 % ROE + 30 % Net Margin + 20 % Earnings Growth + 10 % Revenue Growth**

Fehlende Komponenten werden automatisch herausgerechnet und die vorhandenen Gewichte normalisiert.

---

### 3. Future

**Future = 25 % historische Earnings Growth + 20 % Revenue Growth + 20 % Forward Earnings Growth + 15 % Long-Term Growth + 10 % FCF Margin + 10 % Growth Acceleration**

Dabei:

**Growth Acceleration = Forward Earnings Growth − historische Earnings Growth**

---

### 4. Fair Value

#### Earnings Fair Value

**Fair P/E = 12 + 0,25 × Growth**

begrenzt auf:

**12 ≤ Fair P/E ≤ 24**

Dann:

**Earnings FV = Forward EPS × Fair P/E**

---

#### Normalized FCF Fair Value

Der FCF wird nicht mehr aus einem einzelnen Spitzenjahr verwendet.

Es werden bis zu drei Jahre betrachtet:

**Normalized FCF = Median der positiven Jahres-FCFs**

Dann:

**FCF/share = Normalized FCF / Shares Outstanding**

und:

**FCF FV = FCF/share / Target Yield**

Die Target Yield wird abhängig vom Wachstum angepasst und zusätzlich bei hohem Beta erhöht.

---

#### Analyst Fair Value

**Analyst FV = Median aus verfügbarem Analyst Mean/Median Target**

---

#### Gesamt-Fair-Value

Nicht-finanziell:

**FV = 45 % Earnings FV + 25 % FCF FV + 30 % Analyst FV**

Finanzunternehmen:

**FV = 35 % Earnings FV + 15 % FCF FV + 50 % Analyst FV**

Extreme Einzelmodelle werden vorher begrenzt.

Zusätzlich:

**0,40 × Kurs ≤ Fair Value ≤ 2,50 × Kurs**

---

### 5. Fair Value Score

**Fair Value Score = 55 + 0,60 × Upside %**

anschließend:

**5 ≤ Score ≤ 100**

---

### 6. Relative Valuation

Nicht-finanzielle Unternehmen:

**35 % Forward P/E + 10 % Trailing P/E + 30 % PEG + 25 % P/S**

Finanzunternehmen:

**35 % Forward P/E + 15 % Trailing P/E + 15 % PEG + 35 % P/B**

Die einzelnen Bewertungskennzahlen werden über eine stabile Sigmoid-Funktion in Scores von 5–100 umgerechnet.

---

### 7. KPAX-FV

**KPAX-FV = 55 % Fair Value Score + 45 % Relative Valuation**

---

### 8. Risk

**Risk = 45 % Financial/Stability + 35 % Market Risk + 20 % Stability**

Market Risk berücksichtigt insbesondere:

- annualisierte Volatilität
- Maximum Drawdown
- Beta als Fallback

---

### 9. Investment Score

Standard:

**Investment = 45 % KPAX + 45 % KPAX-FV + 10 % Risk**

Die Gewichte können oben links verändert werden.

---

### 10. Technical

Technical ist **nicht Bestandteil des Investment Scores**.

Er dient ausschließlich als Timing-/Recommendation-Filter.

Dadurch kann beispielsweise ein fundamental starker Titel mit schwachem Chart maximal auf „Hold“ reduziert werden.

---

### 11. Recommendation

- **≥ 85:** Strong Buy
- **≥ 80:** Buy
- **≥ 75:** Accumulate
- **≥ 68:** Hold
- **≥ 55:** Reduce / Watch
- **< 55:** Avoid

Der Technical Score kann die maximale Empfehlung begrenzen.
            """
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        st.subheader(
            "🛠️ Daten- & Modell-Diagnose"
        )

        diagnostic_columns = [
            "Ticker",
            "Company",
            "Currency",
            "Sector",
            "FCF Years",
            "FV Models",
            "FV Confidence",
            "Model Confidence"
        ]

        diagnostic_columns = [
            c for c in diagnostic_columns
            if c in df.columns
        ]

        st.dataframe(
            df[diagnostic_columns],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Interpretation: „—“ bedeutet weiterhin "
            "Daten fehlen. Ein Score von 5 ist dagegen "
            "ein tatsächlich berechneter sehr schlechter Score."
        )

        # ----------------------------------------------------
        # Sanity warnings
        # ----------------------------------------------------

        warnings = []

        for _, row in df.iterrows():

            ticker = row.get(
                "Ticker",
                ""
            )

            fair_value = safe_float(
                row.get("Fair Value")
            )

            price = safe_float(
                row.get("Price")
            )

            confidence = safe_float(
                row.get("Model Confidence")
            )

            if (
                fair_value is not None
                and price is not None
                and price > 0
            ):
                ratio = fair_value / price

                if ratio >= 2.45:
                    warnings.append(
                        f"{ticker}: Fair Value nahe am "
                        f"2,5×-Cap ({ratio:.2f}×)"
                    )

            if (
                confidence is not None
                and confidence < 50
            ):
                warnings.append(
                    f"{ticker}: niedrige Model Confidence "
                    f"({confidence:.0f})"
                )

        if warnings:

            st.subheader(
                "⚠️ Modell-Hinweise"
            )

            for warning in warnings:
                st.warning(warning)

        # ----------------------------------------------------
        # Summary metrics
        # ----------------------------------------------------

        st.subheader(
            "📊 Zusammenfassung"
        )

        valid_investments = (
            df["Investment"]
            .dropna()
        )

        if len(valid_investments) > 0:

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Analysierte Titel",
                    len(df)
                )

            with c2:
                st.metric(
                    "Valide Scores",
                    len(valid_investments)
                )

            with c3:
                st.metric(
                    "Ø Investment Score",
                    f"{valid_investments.mean():.1f}"
                )

            with c4:
                strong = sum(
                    df["Recommendation"]
                    == "Strong Buy"
                )

                st.metric(
                    "Strong Buy",
                    strong
                )
