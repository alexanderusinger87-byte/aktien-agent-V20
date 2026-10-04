import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aktien-Screener V20.1",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# STANDARD TICKER
# ============================================================

DEFAULT_TICKERS = (
    "SMO, BMW.DE, MAIR, GOOGL, IFX.DE, 1810.HK, PEP, KO, "
    "NVDA, AAPL, MU, INTC, AMZN, F8P, NOK, VST, ASML, NKE, "
    "VRT, TSM, NVO, MRVL, 000660.KS, TSLA, 005930.KS, AMD, "
    "ADS.DE, SU.PA, ENR.DE, SIE.DE, MSFT, AVGO, SSUN.F, MCD, "
    "MUV2.DE, ALV.DE"
)


# ============================================================
# INITIAL SCORE WEIGHTS
# ============================================================

INITIAL_SCORE_WEIGHTS = {
    "kpax": 0.40,
    "kpax_fv": 0.50,
    "risk": 0.10
}

MIN_VALID_SCORE = 5


# ============================================================
# FALLBACK SECTORS
# ============================================================

FALLBACK_SECTORS = {

    # Technology
    "GOOGL": "Technology",
    "NVDA": "Technology",
    "AAPL": "Technology",
    "MU": "Technology",
    "INTC": "Technology",
    "ASML": "Technology",
    "TSM": "Technology",
    "MRVL": "Technology",
    "000660.KS": "Technology",
    "005930.KS": "Technology",
    "AMD": "Technology",
    "MSFT": "Technology",
    "AVGO": "Technology",
    "IFX.DE": "Technology",
    "NOK": "Technology",
    "VRT": "Technology",
    "1810.HK": "Technology",
    "F8P": "Technology",
    "SSUN.F": "Technology",

    # Consumer
    "AMZN": "Consumer Cyclical",
    "BMW.DE": "Consumer Cyclical",
    "NKE": "Consumer Cyclical",
    "TSLA": "Consumer Cyclical",
    "MCD": "Consumer Cyclical",
    "SMO": "Industrials",
    "MAIR": "Technology",

    # Consumer Defensive
    "PEP": "Consumer Defensive",
    "KO": "Consumer Defensive",

    # Healthcare
    "NVO": "Healthcare",

    # Energy
    "SU.PA": "Energy",

    # Utilities
    "VST": "Utilities",

    # Industrials
    "ADS.DE": "Consumer Cyclical",
    "ENR.DE": "Industrials",
    "SIE.DE": "Industrials",

    # Financials
    "MUV2.DE": "Financial Services",
    "ALV.DE": "Financial Services",

    # Existing
    "PFE": "Healthcare",
    "TTE.PA": "Energy",
    "ING": "Financial Services",
}


# ============================================================
# FINANCIAL STATEMENT ALIASES
# ============================================================

ALIASES = {

    "revenue": [
        "Total Revenue",
        "Operating Revenue",
        "TotalRevenue"
    ],

    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
        "NetIncome"
    ],

    "ebit": [
        "EBIT",
        "Operating Income",
        "OperatingIncome"
    ],

    "ebitda": [
        "EBITDA",
        "Normalized EBITDA"
    ],

    "pretax": [
        "Pretax Income",
        "PretaxIncome"
    ],

    "tax": [
        "Tax Provision",
        "TaxProvision"
    ],

    "total_assets": [
        "Total Assets",
        "TotalAssets"
    ],

    "current_liabilities": [
        "Current Liabilities",
        "CurrentLiabilities"
    ],

    "cash": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "CashAndCashEquivalents"
    ],

    "debt": [
        "Total Debt",
        "TotalDebt"
    ],

    "operating_cashflow": [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "OperatingCashFlow"
    ],

    "capex": [
        "Capital Expenditure",
        "Capital Expenditure Reported",
        "CapitalExpenditure"
    ],
}


# ============================================================
# HELPERS
# ============================================================

def safe_get(obj, key, default=np.nan):
    try:
        if obj is None:
            return default

        if isinstance(obj, dict):
            value = obj.get(key, default)
        else:
            value = getattr(obj, key, default)

        return value
    except Exception:
        return default


def to_number(value):
    try:
        if value is None:
            return np.nan

        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "")

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:
        return np.nan


def clean_percentage(value):
    value = to_number(value)

    if np.isnan(value):
        return np.nan

    if abs(value) < 1:
        return value * 100

    return value


def clean_positive(value):
    value = to_number(value)

    if np.isnan(value) or value <= 0:
        return np.nan

    return value


def safe_mean(values):
    values = [
        x for x in values
        if x is not None and np.isfinite(x)
    ]

    if not values:
        return np.nan

    return float(np.mean(values))


def get_row(df, aliases):
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.Series(dtype=float)

    for alias in aliases:
        if alias in df.index:
            return df.loc[alias]

    return pd.Series(dtype=float)


def get_first_valid_row(df, aliases):
    row = get_row(df, aliases)

    if row.empty:
        return row

    return row.dropna()


def get_numeric_series(row):
    if row is None or len(row) == 0:
        return pd.Series(dtype=float)

    values = pd.to_numeric(row, errors="coerce")
    values = values.dropna()

    return values


def calculate_growth_volatility(series):
    values = get_numeric_series(series)

    if len(values) < 3:
        return np.nan

    values = values.sort_index()

    growth = values.pct_change().replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(growth) < 2:
        return np.nan

    return float(growth.std() * 100)


def get_estimate_value(df, names):
    if df is None or not isinstance(df, pd.DataFrame):
        return np.nan

    for name in names:

        if name in df.index:

            row = pd.to_numeric(
                df.loc[name],
                errors="coerce"
            ).dropna()

            if len(row) > 0:
                return float(row.iloc[0])

    return np.nan


def weighted_available(values, weights):
    """
    Berechnet einen gewichteten Score nur aus verfügbaren
    Komponenten und normalisiert die Gewichte automatisch.
    """

    valid = []

    for key, weight in weights.items():

        value = values.get(key, np.nan)

        if (
            value is not None
            and np.isfinite(value)
            and weight > 0
        ):
            valid.append((value, weight))

    if not valid:
        return np.nan

    total_weight = sum(weight for _, weight in valid)

    if total_weight <= 0:
        return np.nan

    result = sum(
        value * weight
        for value, weight in valid
    ) / total_weight

    return clip_score(result)


def clip_score(value):
    if value is None or not np.isfinite(value):
        return np.nan

    return float(np.clip(value, MIN_VALID_SCORE, 100))


# ============================================================
# SCORE FUNCTIONS
# ============================================================

def score_growth(value):
    value = to_number(value)

    if np.isnan(value):
        return np.nan

    return np.interp(
        value,
        [-20, -10, 0, 5, 10, 20, 35, 50],
        [5, 15, 30, 45, 58, 75, 90, 100]
    )


def score_margin(value):
    value = to_number(value)

    if np.isnan(value):
        return np.nan

    return np.interp(
        value,
        [-20, 0, 5, 10, 15, 20, 30, 40],
        [5, 20, 35, 50, 65, 78, 90, 100]
    )


def score_roic(value):
    value = to_number(value)

    if np.isnan(value):
        return np.nan

    return np.interp(
        value,
        [-10, 0, 5, 10, 15, 20, 30, 40],
        [5, 20, 35, 50, 65, 78, 90, 100]
    )


def score_roe(value):
    value = to_number(value)

    if np.isnan(value):
        return np.nan

    return np.interp(
        value,
        [-10, 0, 5, 10, 15, 20, 30, 40],
        [5, 20, 35, 50, 65, 78, 90, 100]
    )


def score_forward_pe(value):
    value = to_number(value)

    if np.isnan(value) or value <= 0:
        return np.nan

    return np.interp(
        value,
        [5, 10, 15, 20, 25, 35, 50, 80],
        [100, 90, 78, 65, 50, 30, 15, 5]
    )


def score_trailing_pe(value):
    value = to_number(value)

    if np.isnan(value) or value <= 0:
        return np.nan

    return np.interp(
        value,
        [5, 10, 15, 20, 25, 35, 50, 80],
        [100, 90, 78, 65, 50, 30, 15, 5]
    )


def score_peg(value):
    value = to_number(value)

    if np.isnan(value) or value <= 0:
        return np.nan

    return np.interp(
        value,
        [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0],
        [100, 90, 80, 65, 50, 25, 5]
    )


def score_pb(value):
    value = to_number(value)

    if np.isnan(value) or value <= 0:
        return np.nan

    return np.interp(
        value,
        [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
        [100, 90, 80, 65, 45, 20, 5]
    )


def score_ps(value):
    value = to_number(value)

    if np.isnan(value) or value <= 0:
        return np.nan

    return np.interp(
        value,
        [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0],
        [100, 90, 75, 60, 40, 20, 5]
    )


def score_debt_to_ebitda(value):
    value = to_number(value)

    if np.isnan(value):
        return np.nan

    return np.interp(
        value,
        [-2, 0, 1, 2, 3, 4, 6, 10],
        [100, 95, 85, 75, 65, 50, 25, 5]
    )


def score_debt_equity(value):
    value = to_number(value)

    if np.isnan(value):
        return np.nan

    return np.interp(
        value,
        [0, 20, 40, 60, 100, 150, 250, 400],
        [100, 95, 85, 75, 60, 40, 20, 5]
    )


def score_current_ratio(value):
    value = to_number(value)

    if np.isnan(value):
        return np.nan

    return np.interp(
        value,
        [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
        [5, 15, 35, 50, 70, 85, 100]
    )


def score_volatility(value):
    value = to_number(value)

    if np.isnan(value):
        return np.nan

    return np.interp(
        value,
        [10, 15, 20, 25, 30, 40, 60],
        [100, 90, 78, 65, 50, 25, 5]
    )


def score_drawdown(value):
    value = to_number(value)

    if np.isnan(value):
        return np.nan

    # value wird als negative Drawdown-Prozentzahl erwartet
    return np.interp(
        value,
        [-80, -60, -40, -25, -15, -5, 0],
        [5, 15, 30, 50, 70, 90, 100]
    )


# ============================================================
# DATA FETCH
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_data(ticker):

    result = {
        "ticker": ticker,
        "info": {},
        "fast_info": {},
        "history": pd.DataFrame(),
        "financials": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "ttm_income": pd.DataFrame(),
        "ttm_cashflow": pd.DataFrame(),
        "quarterly_balance": pd.DataFrame(),
        "analyst_targets": pd.DataFrame(),
        "earnings_estimate": pd.DataFrame(),
        "revenue_estimate": pd.DataFrame(),
        "growth_estimate": pd.DataFrame(),
    }

    try:
        stock = yf.Ticker(ticker)

        try:
            result["info"] = stock.info or {}
        except Exception:
            result["info"] = {}

        try:
            result["fast_info"] = dict(stock.fast_info)
        except Exception:
            result["fast_info"] = {}

        try:
            result["history"] = stock.history(
                period="2y",
                auto_adjust=False
            )
        except Exception:
            pass

        try:
            result["financials"] = stock.financials
        except Exception:
            pass

        try:
            result["balance_sheet"] = stock.balance_sheet
        except Exception:
            pass

        try:
            result["cashflow"] = stock.cashflow
        except Exception:
            pass

        try:
            result["ttm_income"] = stock.ttm_income_stmt
        except Exception:
            pass

        try:
            result["ttm_cashflow"] = stock.ttm_cashflow
        except Exception:
            pass

        try:
            result["quarterly_balance"] = stock.quarterly_balance_sheet
        except Exception:
            pass

        try:
            result["analyst_targets"] = stock.analyst_price_targets
        except Exception:
            pass

        try:
            result["earnings_estimate"] = stock.earnings_estimate
        except Exception:
            pass

        try:
            result["revenue_estimate"] = stock.revenue_estimate
        except Exception:
            pass

        try:
            result["growth_estimate"] = stock.growth_estimates
        except Exception:
            pass

    except Exception:
        pass

    return result


# ============================================================
# BASIC DATA
# ============================================================

def get_current_price(data):

    info = data["info"]
    fast = data["fast_info"]

    candidates = [
        safe_get(info, "currentPrice"),
        safe_get(info, "regularMarketPrice"),
        safe_get(fast, "lastPrice"),
    ]

    for value in candidates:

        value = to_number(value)

        if not np.isnan(value) and value > 0:
            return value

    history = data["history"]

    if isinstance(history, pd.DataFrame) and not history.empty:

        try:
            value = float(history["Close"].dropna().iloc[-1])

            if value > 0:
                return value

        except Exception:
            pass

    return np.nan


def get_market_cap(data):

    info = data["info"]
    fast = data["fast_info"]

    for value in [
        safe_get(info, "marketCap"),
        safe_get(fast, "marketCap")
    ]:

        value = to_number(value)

        if not np.isnan(value) and value > 0:
            return value

    return np.nan


def get_sector(ticker, info):

    sector = safe_get(info, "sector", None)

    if sector and sector != "N/A":
        return sector

    return FALLBACK_SECTORS.get(
        ticker,
        "Other"
    )


# ============================================================
# VALUATION
# ============================================================

def get_valuation(data):

    info = data["info"]

    price = get_current_price(data)
    market_cap = get_market_cap(data)

    trailing_pe = clean_positive(
        safe_get(info, "trailingPE")
    )

    forward_pe = clean_positive(
        safe_get(info, "forwardPE")
    )

    ps = clean_positive(
        safe_get(info, "priceToSalesTrailing12Months")
    )

    pb = clean_positive(
        safe_get(info, "priceToBook")
    )

    trailing_eps = clean_positive(
        safe_get(info, "trailingEps")
    )

    forward_eps = clean_positive(
        safe_get(info, "forwardEps")
    )

    revenue_row = get_first_valid_row(
        data["ttm_income"],
        ALIASES["revenue"]
    )

    if revenue_row.empty:
        revenue_row = get_first_valid_row(
            data["financials"],
            ALIASES["revenue"]
        )

    revenue = (
        float(revenue_row.iloc[0])
        if not revenue_row.empty
        else np.nan
    )

    net_income_row = get_first_valid_row(
        data["ttm_income"],
        ALIASES["net_income"]
    )

    if net_income_row.empty:
        net_income_row = get_first_valid_row(
            data["financials"],
            ALIASES["net_income"]
        )

    net_income = (
        float(net_income_row.iloc[0])
        if not net_income_row.empty
        else np.nan
    )

    if np.isnan(trailing_pe):

        if (
            market_cap > 0
            and net_income > 0
        ):
            trailing_pe = (
                market_cap / net_income
            )

        elif (
            price > 0
            and trailing_eps > 0
        ):
            trailing_pe = (
                price / trailing_eps
            )

    if np.isnan(forward_pe):

        if (
            price > 0
            and forward_eps > 0
        ):
            forward_pe = (
                price / forward_eps
            )

    if np.isnan(ps):

        if (
            market_cap > 0
            and revenue > 0
        ):
            ps = market_cap / revenue

    equity_row = get_first_valid_row(
        data["quarterly_balance"],
        ["Stockholders Equity",
         "Common Stock Equity",
         "Total Equity Gross Minority Interest"]
    )

    equity = (
        float(equity_row.iloc[0])
        if not equity_row.empty
        else np.nan
    )

    if np.isnan(pb):

        if (
            market_cap > 0
            and equity > 0
        ):
            pb = market_cap / equity

    return {
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "ps": ps,
        "pb": pb
    }


# ============================================================
# FREE CASH FLOW
# ============================================================

def get_free_cash_flow(data):

    info = data["info"]

    direct_fcf = to_number(
        safe_get(info, "freeCashflow")
    )

    if (
        not np.isnan(direct_fcf)
        and direct_fcf != 0
    ):
        return direct_fcf

    cfo_row = get_first_valid_row(
        data["ttm_cashflow"],
        ALIASES["operating_cashflow"]
    )

    capex_row = get_first_valid_row(
        data["ttm_cashflow"],
        ALIASES["capex"]
    )

    if cfo_row.empty:
        cfo_row = get_first_valid_row(
            data["cashflow"],
            ALIASES["operating_cashflow"]
        )

    if capex_row.empty:
        capex_row = get_first_valid_row(
            data["cashflow"],
            ALIASES["capex"]
        )

    if cfo_row.empty:
        return np.nan

    cfo = float(cfo_row.iloc[0])

    capex = (
        float(capex_row.iloc[0])
        if not capex_row.empty
        else 0
    )

    return cfo - abs(capex)


# ============================================================
# FAIR VALUE
# ============================================================

def extract_analyst_target(data):

    targets = data["analyst_targets"]

    if isinstance(targets, pd.DataFrame) and not targets.empty:

        possible = [
            "current",
            "mean",
            "median",
            "low",
            "high"
        ]

        values = {}

        for key in possible:

            if key in targets.index:

                value = pd.to_numeric(
                    targets.loc[key],
                    errors="coerce"
                ).dropna()

                if len(value):
                    values[key] = float(value.iloc[0])

        if "median" in values:
            return values["median"]

        if "mean" in values:
            return values["mean"]

    info = data["info"]

    for key in [
        "targetMedianPrice",
        "targetMeanPrice"
    ]:

        value = clean_positive(
            safe_get(info, key)
        )

        if not np.isnan(value):
            return value

    return np.nan


def calculate_fair_value(data, metrics):

    sector = metrics["sector"]
    price = metrics["price"]

    forward_eps = metrics["forward_eps"]
    growth = metrics["earnings_growth"]
    fcf = metrics["fcf"]
    shares = metrics["shares"]
    beta = metrics["beta"]

    analyst_target = extract_analyst_target(data)

    financial = sector == "Financial Services"

    models = []

    # --------------------------------------------------------
    # NON FINANCIAL
    # --------------------------------------------------------

    if not financial:

        # Earnings FV
        if (
            not np.isnan(forward_eps)
            and forward_eps > 0
        ):

            growth_pct = (
                growth
                if not np.isnan(growth)
                else 8
            )

            fair_pe = np.clip(
                10 + 0.40 * growth_pct,
                10,
                28
            )

            earnings_fv = (
                forward_eps * fair_pe
            )

            if earnings_fv > 0:
                models.append(
                    ("earnings", earnings_fv, 0.50)
                )

        # FCF FV
        if (
            not np.isnan(fcf)
            and fcf > 0
            and not np.isnan(shares)
            and shares > 0
        ):

            fcf_per_share = (
                fcf / shares
            )

            growth_pct = (
                growth
                if not np.isnan(growth)
                else 8
            )

            target_yield = (
                0.08
                - 0.0015 * growth_pct
            )

            target_yield = np.clip(
                target_yield,
                0.035,
                0.10
            )

            if (
                not np.isnan(beta)
                and beta > 1
            ):
                target_yield += (
                    0.005 * (beta - 1)
                )

            target_yield = np.clip(
                target_yield,
                0.035,
                0.12
            )

            fcf_fv = (
                fcf_per_share
                / target_yield
            )

            if fcf_fv > 0:
                models.append(
                    ("fcf", fcf_fv, 0.30)
                )

        # Analyst FV
        if (
            not np.isnan(analyst_target)
            and analyst_target > 0
        ):
            models.append(
                ("analyst", analyst_target, 0.20)
            )

    # --------------------------------------------------------
    # FINANCIAL SERVICES
    # --------------------------------------------------------

    else:

        roe = metrics["roe"]
        pb = metrics["pb"]

        if (
            not np.isnan(roe)
            and roe > 0
            and not np.isnan(pb)
        ):

            beta_value = (
                beta
                if not np.isnan(beta)
                else 1
            )

            ke = (
                0.04
                + beta_value * 0.055
            )

            ke = np.clip(
                ke,
                0.07,
                0.14
            )

            g = 0.02

            if g >= ke:
                g = ke - 0.01

            justified_pb = (
                (roe / 100 - g)
                / (ke - g)
            )

            if justified_pb > 0:

                fair_value = (
                    price
                    * justified_pb
                    / max(pb, 0.1)
                )

                if fair_value > 0:
                    models.append(
                        ("pb_roe", fair_value, 0.45)
                    )

        if (
            not np.isnan(forward_eps)
            and forward_eps > 0
        ):

            growth_pct = (
                growth
                if not np.isnan(growth)
                else 8
            )

            fair_pe = np.clip(
                10 + 0.40 * growth_pct,
                10,
                24
            )

            earnings_fv = (
                forward_eps * fair_pe
            )

            if earnings_fv > 0:
                models.append(
                    ("earnings", earnings_fv, 0.35)
                )

        if (
            not np.isnan(analyst_target)
            and analyst_target > 0
        ):
            models.append(
                ("analyst", analyst_target, 0.20)
            )

    if not models:
        return {
            "fair_value": np.nan,
            "fair_value_score": np.nan,
            "fair_value_upside": np.nan
        }

    total_weight = sum(
        weight for _, _, weight in models
    )

    fair_value = sum(
        value * weight
        for _, value, weight in models
    ) / total_weight

    if (
        np.isnan(price)
        or price <= 0
    ):
        upside = np.nan
        fv_score = np.nan

    else:

        upside = (
            fair_value / price - 1
        ) * 100

        fv_score = np.interp(
            upside,
            [-40, -20, 0, 10, 25, 50, 75],
            [5, 15, 40, 58, 75, 90, 100]
        )

        fv_score = clip_score(
            fv_score
        )

    return {
        "fair_value": fair_value,
        "fair_value_score": fv_score,
        "fair_value_upside": upside
    }


# ============================================================
# TECHNICAL
# ============================================================

def calculate_technical(history):

    if (
        not isinstance(history, pd.DataFrame)
        or history.empty
    ):
        return {
            "sma50": np.nan,
            "sma200": np.nan,
            "rsi": np.nan,
            "macd": np.nan,
            "perf_1m": np.nan,
            "perf_6m": np.nan,
            "perf_12m": np.nan
        }

    close = pd.to_numeric(
        history["Close"],
        errors="coerce"
    ).dropna()

    if len(close) < 20:
        return {
            "sma50": np.nan,
            "sma200": np.nan,
            "rsi": np.nan,
            "macd": np.nan,
            "perf_1m": np.nan,
            "perf_6m": np.nan,
            "perf_12m": np.nan
        }

    sma50 = (
        close.rolling(50).mean().iloc[-1]
        if len(close) >= 50
        else np.nan
    )

    sma200 = (
        close.rolling(200).mean().iloc[-1]
        if len(close) >= 200
        else np.nan
    )

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi_series = 100 - (
        100 / (1 + rs)
    )

    rsi = (
        rsi_series.iloc[-1]
        if len(rsi_series)
        else np.nan
    )

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd_line = ema12 - ema26

    signal = macd_line.ewm(
        span=9,
        adjust=False
    ).mean()

    macd = (
        macd_line.iloc[-1]
        - signal.iloc[-1]
    )

    def performance(days):

        if len(close) <= days:
            return np.nan

        old = close.iloc[-days - 1]
        new = close.iloc[-1]

        if old <= 0:
            return np.nan

        return (
            new / old - 1
        ) * 100

    return {
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "macd": macd,
        "perf_1m": performance(21),
        "perf_6m": performance(126),
        "perf_12m": performance(252)
    }


# ============================================================
# EXTRACT METRICS
# ============================================================

def extract_metrics(data):

    ticker = data["ticker"]
    info = data["info"]

    price = get_current_price(data)
    market_cap = get_market_cap(data)

    sector = get_sector(
        ticker,
        info
    )

    shares = clean_positive(
        safe_get(info, "sharesOutstanding")
    )

    if (
        np.isnan(shares)
        and market_cap > 0
        and price > 0
    ):
        shares = (
            market_cap / price
        )

    valuation = get_valuation(data)

    trailing_pe = valuation["trailing_pe"]
    forward_pe = valuation["forward_pe"]
    ps = valuation["ps"]
    pb = valuation["pb"]

    beta = to_number(
        safe_get(info, "beta")
    )

    # --------------------------------------------------------
    # Income statement
    # --------------------------------------------------------

    revenue_row = get_first_valid_row(
        data["ttm_income"],
        ALIASES["revenue"]
    )

    if revenue_row.empty:
        revenue_row = get_first_valid_row(
            data["financials"],
            ALIASES["revenue"]
        )

    revenue = (
        float(revenue_row.iloc[0])
        if not revenue_row.empty
        else np.nan
    )

    net_income_row = get_first_valid_row(
        data["ttm_income"],
        ALIASES["net_income"]
    )

    if net_income_row.empty:
        net_income_row = get_first_valid_row(
            data["financials"],
            ALIASES["net_income"]
        )

    net_income = (
        float(net_income_row.iloc[0])
        if not net_income_row.empty
        else np.nan
    )

    ebit_row = get_first_valid_row(
        data["ttm_income"],
        ALIASES["ebit"]
    )

    ebit = (
        float(ebit_row.iloc[0])
        if not ebit_row.empty
        else np.nan
    )

    # --------------------------------------------------------
    # Growth
    # --------------------------------------------------------

    revenue_history = get_row(
        data["financials"],
        ALIASES["revenue"]
    )

    earnings_history = get_row(
        data["financials"],
        ALIASES["net_income"]
    )

    revenue_values = get_numeric_series(
        revenue_history
    )

    earnings_values = get_numeric_series(
        earnings_history
    )

    revenue_growth = np.nan
    earnings_growth = np.nan

    if len(revenue_values) >= 2:

        latest = revenue_values.iloc[0]
        previous = revenue_values.iloc[1]

        if previous > 0:
            revenue_growth = (
                latest / previous - 1
            ) * 100

    if len(earnings_values) >= 2:

        latest = earnings_values.iloc[0]
        previous = earnings_values.iloc[1]

        if previous != 0:

            earnings_growth = (
                latest / previous - 1
            ) * 100

    # Forward earnings growth from estimates
    growth_estimate = get_estimate_value(
        data["growth_estimate"],
        [
            "stock",
            "0q",
            "+1q",
            "0y",
            "+1y"
        ]
    )

    if (
        not np.isnan(growth_estimate)
        and abs(growth_estimate) < 1
    ):
        growth_estimate *= 100

    # Prefer explicit forward growth if available
    if not np.isnan(growth_estimate):
        forward_growth = growth_estimate
    else:
        forward_growth = earnings_growth

    # --------------------------------------------------------
    # Margins
    # --------------------------------------------------------

    net_margin = np.nan

    if (
        revenue > 0
        and not np.isnan(net_income)
    ):
        net_margin = (
            net_income / revenue
        ) * 100

    ebit_margin = np.nan

    if (
        revenue > 0
        and not np.isnan(ebit)
    ):
        ebit_margin = (
            ebit / revenue
        ) * 100

    gross_margin = clean_percentage(
        safe_get(info, "grossMargins")
    )

    # --------------------------------------------------------
    # ROE / ROIC
    # --------------------------------------------------------

    roe = clean_percentage(
        safe_get(info, "returnOnEquity")
    )

    roic = clean_percentage(
        safe_get(info, "returnOnAssets")
    )

    # Approximation: if ROIC unavailable, use ROA
    if np.isnan(roic):

        total_assets_row = get_first_valid_row(
            data["balance_sheet"],
            ALIASES["total_assets"]
        )

        assets = (
            float(total_assets_row.iloc[0])
            if not total_assets_row.empty
            else np.nan
        )

        if (
            assets > 0
            and not np.isnan(ebit)
        ):
            roic = (
                ebit / assets
            ) * 100

    # --------------------------------------------------------
    # FCF
    # --------------------------------------------------------

    fcf = get_free_cash_flow(data)

    fcf_margin = np.nan

    if (
        revenue > 0
        and not np.isnan(fcf)
    ):
        fcf_margin = (
            fcf / revenue
        ) * 100

    fcf_per_share = np.nan

    if (
        not np.isnan(fcf)
        and not np.isnan(shares)
        and shares > 0
    ):
        fcf_per_share = (
            fcf / shares
        )

    # --------------------------------------------------------
    # Balance sheet
    # --------------------------------------------------------

    debt = np.nan
    cash = np.nan

    debt_row = get_first_valid_row(
        data["quarterly_balance"],
        ALIASES["debt"]
    )

    cash_row = get_first_valid_row(
        data["quarterly_balance"],
        ALIASES["cash"]
    )

    if debt_row.empty:
        debt_row = get_first_valid_row(
            data["balance_sheet"],
            ALIASES["debt"]
        )

    if cash_row.empty:
        cash_row = get_first_valid_row(
            data["balance_sheet"],
            ALIASES["cash"]
        )

    if not debt_row.empty:
        debt = float(debt_row.iloc[0])

    if not cash_row.empty:
        cash = float(cash_row.iloc[0])

    net_debt = np.nan

    if (
        not np.isnan(debt)
        and not np.isnan(cash)
    ):
        net_debt = debt - cash

    ebitda = to_number(
        safe_get(info, "ebitda")
    )

    net_debt_ebitda = np.nan

    if (
        not np.isnan(net_debt)
        and ebitda > 0
    ):
        net_debt_ebitda = (
            net_debt / ebitda
        )

    debt_equity = clean_percentage(
        safe_get(info, "debtToEquity")
    )

    current_ratio = to_number(
        safe_get(info, "currentRatio")
    )

    cash_to_debt = np.nan

    if (
        not np.isnan(cash)
        and not np.isnan(debt)
        and debt > 0
    ):
        cash_to_debt = (
            cash / debt
        )

    # --------------------------------------------------------
    # PEG
    # --------------------------------------------------------

    peg = clean_positive(
        safe_get(info, "pegRatio")
    )

    # --------------------------------------------------------
    # Analyst
    # --------------------------------------------------------

    analyst_target = extract_analyst_target(
        data
    )

    analyst_upside = np.nan

    if (
        not np.isnan(analyst_target)
        and price > 0
    ):
        analyst_upside = (
            analyst_target / price - 1
        ) * 100

    # --------------------------------------------------------
    # Technical
    # --------------------------------------------------------

    technical = calculate_technical(
        data["history"]
    )

    # --------------------------------------------------------
    # Fair Value
    # --------------------------------------------------------

    metrics = {
        "sector": sector,
        "price": price,
        "forward_eps": clean_positive(
            safe_get(info, "forwardEps")
        ),
        "earnings_growth": forward_growth,
        "fcf": fcf,
        "shares": shares,
        "beta": beta,
        "roe": roe,
        "pb": pb
    }

    fair_value = calculate_fair_value(
        data,
        metrics
    )

    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------

    core_values = [
        price,
        revenue_growth,
        earnings_growth,
        net_margin,
        forward_pe,
        fcf_margin,
        roe,
        debt_equity,
        current_ratio
    ]

    available = sum(
        1 for x in core_values
        if x is not None
        and np.isfinite(x)
    )

    completeness = (
        available / len(core_values)
    ) * 100

    return {

        "ticker": ticker,
        "name": safe_get(
            info,
            "shortName",
            ticker
        ),
        "sector": sector,

        "price": price,
        "market_cap": market_cap,

        "revenue": revenue,
        "net_income": net_income,
        "ebit": ebit,

        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,

        "net_margin": net_margin,
        "ebit_margin": ebit_margin,
        "gross_margin": gross_margin,

        "roe": roe,
        "roic": roic,

        "fcf": fcf,
        "fcf_margin": fcf_margin,
        "fcf_per_share": fcf_per_share,

        "debt": debt,
        "cash": cash,
        "net_debt": net_debt,
        "net_debt_ebitda": net_debt_ebitda,
        "debt_equity": debt_equity,
        "current_ratio": current_ratio,
        "cash_to_debt": cash_to_debt,

        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "ps": ps,
        "pb": pb,
        "peg": peg,

        "forward_eps": clean_positive(
            safe_get(info, "forwardEps")
        ),

        "analyst_target": analyst_target,
        "analyst_upside": analyst_upside,

        "beta": beta,

        "sma50": technical["sma50"],
        "sma200": technical["sma200"],
        "rsi": technical["rsi"],
        "macd": technical["macd"],
        "perf_1m": technical["perf_1m"],
        "perf_6m": technical["perf_6m"],
        "perf_12m": technical["perf_12m"],

        "fair_value": fair_value["fair_value"],
        "fair_value_score": fair_value[
            "fair_value_score"
        ],
        "fair_value_upside": fair_value[
            "fair_value_upside"
        ],

        "completeness": completeness
    }


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality(m):

    financial = (
        m["sector"] == "Financial Services"
    )

    if financial:

        components = {
            "roe": score_roe(m["roe"]),
            "net_margin": score_margin(
                m["net_margin"]
            ),
            "earnings_growth": score_growth(
                m["earnings_growth"]
            ),
            "revenue_growth": score_growth(
                m["revenue_growth"]
            )
        }

        weights = {
            "roe": 0.40,
            "net_margin": 0.30,
            "earnings_growth": 0.20,
            "revenue_growth": 0.10
        }

    else:

        components = {
            "roic": score_roic(m["roic"]),
            "gross_margin": score_margin(
                m["gross_margin"]
            ),
            "fcf_margin": score_margin(
                m["fcf_margin"]
            ),
            "earnings_growth": score_growth(
                m["earnings_growth"]
            ),
            "revenue_growth": score_growth(
                m["revenue_growth"]
            )
        }

        weights = {
            "roic": 0.30,
            "gross_margin": 0.15,
            "fcf_margin": 0.20,
            "earnings_growth": 0.20,
            "revenue_growth": 0.10
        }

    return weighted_available(
        components,
        weights
    )


# ============================================================
# FUTURE SCORE
# ============================================================

def calculate_future(m):

    components = {

        "earnings_growth": score_growth(
            m["earnings_growth"]
        ),

        "revenue_growth": score_growth(
            m["revenue_growth"]
        ),

        "long_term_growth": score_growth(
            m["earnings_growth"]
        ),

        "analyst_upside": (
            np.interp(
                m["analyst_upside"],
                [-30, -10, 0, 10, 25, 50, 100],
                [5, 25, 40, 60, 75, 90, 100]
            )
            if not np.isnan(
                m["analyst_upside"]
            )
            else np.nan
        ),

        "fcf_margin": score_margin(
            m["fcf_margin"]
        ),

        "growth_acceleration": score_growth(
            m["revenue_growth"]
        )
    }

    weights = {
        "earnings_growth": 0.30,
        "revenue_growth": 0.20,
        "long_term_growth": 0.15,
        "analyst_upside": 0.15,
        "fcf_margin": 0.10,
        "growth_acceleration": 0.10
    }

    return weighted_available(
        components,
        weights
    )


# ============================================================
# RELATIVE VALUATION
# ============================================================

def calculate_relative_valuation(m):

    financial = (
        m["sector"] == "Financial Services"
    )

    components = {

        "forward_pe": score_forward_pe(
            m["forward_pe"]
        ),

        "trailing_pe": score_trailing_pe(
            m["trailing_pe"]
        ),

        "peg": score_peg(
            m["peg"]
        )
    }

    weights = {
        "forward_pe": 0.40,
        "trailing_pe": 0.20,
        "peg": 0.20
    }

    if financial:

        components["pb"] = score_pb(
            m["pb"]
        )

        weights["pb"] = 0.20

    else:

        components["ps"] = score_ps(
            m["ps"]
        )

        weights["ps"] = 0.20

    return weighted_available(
        components,
        weights
    )


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk(m):

    financial = (
        m["sector"] == "Financial Services"
    )

    if financial:

        financial_risk_components = {

            "debt_equity": score_debt_equity(
                m["debt_equity"]
            ),

            "cash_to_debt": (
                np.interp(
                    m["cash_to_debt"],
                    [0, 0.25, 0.5, 1, 2, 4],
                    [5, 25, 50, 70, 90, 100]
                )
                if not np.isnan(
                    m["cash_to_debt"]
                )
                else np.nan
            )
        }

    else:

        financial_risk_components = {

            "net_debt_ebitda":
                score_debt_to_ebitda(
                    m["net_debt_ebitda"]
                ),

            "current_ratio":
                score_current_ratio(
                    m["current_ratio"]
                ),

            "debt_equity":
                score_debt_equity(
                    m["debt_equity"]
                )
        }

    financial_risk = weighted_available(
        financial_risk_components,
        {
            "net_debt_ebitda": 0.45,
            "current_ratio": 0.20,
            "debt_equity": 0.35
        }
        if not financial
        else
        {
            "debt_equity": 0.60,
            "cash_to_debt": 0.40
        }
    )

    # Market risk
    history = m.get("history", None)

    volatility = np.nan
    drawdown = np.nan

    # use supplied technical return data as fallback
    if (
        not np.isnan(m["perf_12m"])
    ):
        volatility = abs(
            m["perf_12m"]
        ) / 3 + 15

    market_risk = weighted_available(
        {
            "volatility":
                score_volatility(
                    volatility
                ),

            "drawdown":
                score_drawdown(
                    drawdown
                )
        },
        {
            "volatility": 0.50,
            "drawdown": 0.50
        }
    )

    # Stability
    revenue_stability = (
        70
        if not np.isnan(
            m["revenue_growth"]
        )
        else np.nan
    )

    earnings_stability = (
        70
        if not np.isnan(
            m["earnings_growth"]
        )
        else np.nan
    )

    stability = weighted_available(
        {
            "revenue_stability":
                revenue_stability,
            "earnings_stability":
                earnings_stability
        },
        {
            "revenue_stability": 0.40,
            "earnings_stability": 0.60
        }
    )

    return weighted_available(
        {
            "financial_risk":
                financial_risk,
            "market_risk":
                market_risk,
            "stability":
                stability
        },
        {
            "financial_risk": 0.40,
            "market_risk": 0.30,
            "stability": 0.30
        }
    )


# ============================================================
# TECHNICAL SCORE
# ============================================================

def calculate_technical_score(m):

    components = {

        "sma200":
            (
                80
                if (
                    not np.isnan(m["sma200"])
                    and not np.isnan(m["price"])
                    and m["price"] > m["sma200"]
                )
                else 30
                if not np.isnan(m["sma200"])
                else np.nan
            ),

        "perf_6m":
            (
                np.interp(
                    m["perf_6m"],
                    [-50, -20, -10, 0, 10, 25, 50],
                    [5, 20, 35, 50, 65, 85, 100]
                )
                if not np.isnan(
                    m["perf_6m"]
                )
                else np.nan
            ),

        "rsi":
            (
                np.interp(
                    m["rsi"],
                    [20, 30, 40, 50, 60, 70, 80],
                    [30, 55, 75, 90, 100, 75, 35]
                )
                if not np.isnan(
                    m["rsi"]
                )
                else np.nan
            ),

        "macd":
            (
                75
                if m["macd"] > 0
                else 35
            )
            if not np.isnan(m["macd"])
            else np.nan,

        "sma50":
            (
                75
                if (
                    not np.isnan(m["sma50"])
                    and m["price"] > m["sma50"]
                )
                else 35
            )
            if (
                not np.isnan(m["sma50"])
                and not np.isnan(m["price"])
            )
            else np.nan
    }

    return weighted_available(
        components,
        {
            "sma200": 0.25,
            "perf_6m": 0.25,
            "rsi": 0.20,
            "macd": 0.20,
            "sma50": 0.10
        }
    )


# ============================================================
# KPAX + INVESTMENT SCORE
# ============================================================

def calculate_scores(
    m,
    score_weights
):

    quality = calculate_quality(m)

    future = calculate_future(m)

    kpax = weighted_available(
        {
            "quality": quality,
            "future": future
        },
        {
            "quality": 0.40,
            "future": 0.60
        }
    )

    relative_valuation = (
        calculate_relative_valuation(m)
    )

    kpax_fv = weighted_available(
        {
            "fair_value": m[
                "fair_value_score"
            ],
            "relative_valuation":
                relative_valuation
        },
        {
            "fair_value": 0.60,
            "relative_valuation": 0.40
        }
    )

    risk = calculate_risk(m)

    investment_score = weighted_available(
        {
            "kpax": kpax,
            "kpax_fv": kpax_fv,
            "risk": risk
        },
        score_weights
    )

    technical = calculate_technical_score(
        m
    )

    return {

        "quality": quality,
        "future": future,
        "kpax": kpax,

        "fair_value_score":
            m["fair_value_score"],

        "relative_valuation":
            relative_valuation,

        "kpax_fv":
            kpax_fv,

        "risk":
            risk,

        "investment_score":
            investment_score,

        "technical":
            technical
    }


# ============================================================
# VERIFY SCORE
# ============================================================

def calculate_verify(m):

    """
    Unabhängiger Plausibilitätscheck.

    Bewusst NICHT verwendet:
    - Fair Value
    - Analyst Target
    - RSI
    - MACD
    - KPAX
    - KPAX-FV

    Damit kann Verify den Hauptscore tatsächlich hinterfragen.
    """

    financial = (
        m["sector"] == "Financial Services"
    )

    # --------------------------------------------------------
    # 1. Earnings Growth – 25 %
    # --------------------------------------------------------

    earnings_score = score_growth(
        m["earnings_growth"]
    )

    # --------------------------------------------------------
    # 2. Revenue Growth – 15 %
    # --------------------------------------------------------

    revenue_score = score_growth(
        m["revenue_growth"]
    )

    # --------------------------------------------------------
    # 3. Profitability – 25 %
    # --------------------------------------------------------

    if financial:

        profitability_score = weighted_available(
            {
                "roe":
                    score_roe(m["roe"]),
                "net_margin":
                    score_margin(
                        m["net_margin"]
                    )
            },
            {
                "roe": 0.60,
                "net_margin": 0.40
            }
        )

    else:

        profitability_score = weighted_available(
            {
                "roic":
                    score_roic(m["roic"]),
                "fcf_margin":
                    score_margin(
                        m["fcf_margin"]
                    ),
                "gross_margin":
                    score_margin(
                        m["gross_margin"]
                    )
            },
            {
                "roic": 0.50,
                "fcf_margin": 0.30,
                "gross_margin": 0.20
            }
        )

    # --------------------------------------------------------
    # 4. Debt – 15 %
    # --------------------------------------------------------

    if financial:

        debt_score = weighted_available(
            {
                "debt_equity":
                    score_debt_equity(
                        m["debt_equity"]
                    ),

                "cash_to_debt":
                    (
                        np.interp(
                            m["cash_to_debt"],
                            [0, .25, .5, 1, 2, 4],
                            [5, 25, 50, 75, 90, 100]
                        )
                        if not np.isnan(
                            m["cash_to_debt"]
                        )
                        else np.nan
                    )
            },
            {
                "debt_equity": 0.60,
                "cash_to_debt": 0.40
            }
        )

    else:

        debt_score = weighted_available(
            {
                "net_debt_ebitda":
                    score_debt_to_ebitda(
                        m["net_debt_ebitda"]
                    ),

                "debt_equity":
                    score_debt_equity(
                        m["debt_equity"]
                    ),

                "current_ratio":
                    score_current_ratio(
                        m["current_ratio"]
                    )
            },
            {
                "net_debt_ebitda": 0.60,
                "debt_equity": 0.25,
                "current_ratio": 0.15
            }
        )

    # --------------------------------------------------------
    # 5. Valuation – 20 %
    # --------------------------------------------------------

    if financial:

        valuation_score = weighted_available(
            {
                "forward_pe":
                    score_forward_pe(
                        m["forward_pe"]
                    ),
                "pb":
                    score_pb(m["pb"])
            },
            {
                "forward_pe": 0.60,
                "pb": 0.40
            }
        )

    else:

        valuation_score = weighted_available(
            {
                "forward_pe":
                    score_forward_pe(
                        m["forward_pe"]
                    ),

                "peg":
                    score_peg(m["peg"]),

                "ps":
                    score_ps(m["ps"])
            },
            {
                "forward_pe": 0.50,
                "peg": 0.30,
                "ps": 0.20
            }
        )

    components = {
        "Earnings Growth":
            earnings_score,

        "Revenue Growth":
            revenue_score,

        "Profitability":
            profitability_score,

        "Debt":
            debt_score,

        "Valuation":
            valuation_score
    }

    weights = {
        "Earnings Growth": 0.25,
        "Revenue Growth": 0.15,
        "Profitability": 0.25,
        "Debt": 0.15,
        "Valuation": 0.20
    }

    verify = weighted_available(
        components,
        weights
    )

    available = sum(
        1
        for value in components.values()
        if value is not None
        and np.isfinite(value)
    )

    verify_data = (
        available / 5 * 100
    )

    return {
        "verify": verify,
        "verify_data": verify_data,
        "verify_earnings": earnings_score,
        "verify_revenue": revenue_score,
        "verify_profitability":
            profitability_score,
        "verify_debt": debt_score,
        "verify_valuation":
            valuation_score
    }


# ============================================================
# RECOMMENDATION
# ============================================================

def get_recommendation(
    investment_score,
    kpax,
    kpax_fv,
    technical
):

    if np.isnan(investment_score):
        return "—"

    if np.isnan(technical):
        technical = 50

    if (
        investment_score >= 85
        and kpax >= 80
        and kpax_fv >= 80
    ):
        recommendation = "Strong Buy"

    elif (
        investment_score >= 80
        and kpax >= 75
    ):
        recommendation = "Buy"

    elif investment_score >= 75:
        recommendation = "Accumulate"

    elif investment_score >= 68:
        recommendation = "Hold"

    elif investment_score >= 55:
        recommendation = "Reduce / Watch"

    else:
        recommendation = "Avoid"

    # Technical gate
    if technical < 30:

        if recommendation in [
            "Strong Buy",
            "Buy",
            "Accumulate"
        ]:
            recommendation = "Hold"

    elif technical < 45:

        if recommendation == "Strong Buy":
            recommendation = "Buy"

        elif recommendation == "Buy":
            recommendation = "Accumulate"

    elif technical < 60:

        if recommendation == "Strong Buy":
            recommendation = "Buy"

    return recommendation


def get_timing(technical):

    if np.isnan(technical):
        return "—"

    if technical >= 75:
        return "🟢 Stark"

    if technical >= 60:
        return "🟢 Positiv"

    if technical >= 45:
        return "🟡 Neutral"

    if technical >= 30:
        return "🟠 Schwach"

    return "🔴 Negativ"


def get_turnaround_status(m):

    earnings = m["earnings_growth"]
    revenue = m["revenue_growth"]

    if (
        not np.isnan(earnings)
        and not np.isnan(revenue)
    ):

        if (
            earnings > 15
            and revenue > 8
        ):
            return "🚀 Wachstum"

        if (
            earnings > 5
            and revenue > 0
        ):
            return "🟢 Positiv"

        if (
            earnings < -10
            and revenue < 0
        ):
            return "🔴 Schwach"

    return "🟡 Neutral"


# ============================================================
# VERIFY STATUS
# ============================================================

def get_verify_status(
    gap,
    rank_diff,
    verify_data
):

    if np.isnan(verify_data):
        return "⚪ Keine Daten"

    if verify_data < 60:
        return "⚪ Datenbasis schwach"

    if np.isnan(gap):
        return "⚪ Nicht prüfbar"

    abs_gap = abs(gap)

    abs_rank = (
        abs(rank_diff)
        if not np.isnan(rank_diff)
        else 0
    )

    if (
        abs_gap <= 8
        and abs_rank <= 2
    ):
        return "🟢 Plausibel"

    if (
        abs_gap <= 15
        and abs_rank <= 4
    ):
        return "🟡 Prüfen"

    return "🔴 Auffällig"


# ============================================================
# MAIN APP
# ============================================================

st.title("📊 Aktien-Screener V20.1")

st.caption(
    "KPAX entscheidet – Verify hinterfragt KPAX."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Einstellungen")


ticker_input = st.sidebar.text_area(
    "Aktien / Ticker",
    value=DEFAULT_TICKERS,
    height=160
)


min_investment_score = st.sidebar.number_input(
    "Minimaler Investment Score",
    min_value=0,
    max_value=100,
    value=0,
    step=1
)


st.sidebar.subheader(
    "Investment Score Gewichtung"
)


if "weights" not in st.session_state:
    st.session_state.weights = (
        INITIAL_SCORE_WEIGHTS.copy()
    )


if st.sidebar.button(
    "🔄 Gewichtung zurücksetzen"
):
    st.session_state.weights = (
        INITIAL_SCORE_WEIGHTS.copy()
    )


kpax_weight = st.sidebar.number_input(
    "KPAX %",
    min_value=0.0,
    max_value=100.0,
    value=float(
        st.session_state.weights["kpax"] * 100
    ),
    step=5.0
)

kpax_fv_weight = st.sidebar.number_input(
    "KPAX-FV %",
    min_value=0.0,
    max_value=100.0,
    value=float(
        st.session_state.weights["kpax_fv"] * 100
    ),
    step=5.0
)

risk_weight = st.sidebar.number_input(
    "Risk %",
    min_value=0.0,
    max_value=100.0,
    value=float(
        st.session_state.weights["risk"] * 100
    ),
    step=5.0
)


weight_total = (
    kpax_weight
    + kpax_fv_weight
    + risk_weight
)


if weight_total <= 0:

    st.sidebar.error(
        "Mindestens ein Gewicht muss > 0 sein."
    )

    score_weights = (
        INITIAL_SCORE_WEIGHTS.copy()
    )

else:

    score_weights = {

        "kpax":
            kpax_weight / weight_total,

        "kpax_fv":
            kpax_fv_weight / weight_total,

        "risk":
            risk_weight / weight_total
    }


st.sidebar.caption(
    f"Normalisiert: "
    f"{score_weights['kpax']:.0%} / "
    f"{score_weights['kpax_fv']:.0%} / "
    f"{score_weights['risk']:.0%}"
)


# ============================================================
# TICKER PARSING
# ============================================================

tickers = [
    x.strip().upper()
    for x in ticker_input.replace(
        "\n", ","
    ).split(",")
    if x.strip()
]

# remove duplicates but preserve order
tickers = list(
    dict.fromkeys(tickers)
)


if not tickers:

    st.warning(
        "Bitte mindestens einen Ticker eingeben."
    )

    st.stop()


# ============================================================
# DATA PROCESSING
# ============================================================

rows = []

progress = st.progress(0)

for i, ticker in enumerate(tickers):

    try:

        data = fetch_stock_data(
            ticker
        )

        metrics = extract_metrics(
            data
        )

        scores = calculate_scores(
            metrics,
            score_weights
        )

        verify = calculate_verify(
            metrics
        )

        row = {
            **metrics,
            **scores,
            **verify
        }

        rows.append(row)

    except Exception as e:

        rows.append({
            "ticker": ticker,
            "name": ticker,
            "sector": FALLBACK_SECTORS.get(
                ticker,
                "Unknown"
            ),
            "investment_score": np.nan,
            "verify": np.nan,
            "verify_data": 0,
            "completeness": 0
        })

    progress.progress(
        (i + 1) / len(tickers)
    )


progress.empty()


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


if df.empty:

    st.error(
        "Keine Daten konnten verarbeitet werden."
    )

    st.stop()


# ============================================================
# RANKINGS
# ============================================================

# Ranking über ALLE analysierten Aktien,
# nicht nur über die nach Score gefilterte Ansicht.

df["investment_rank"] = (
    df["investment_score"]
    .rank(
        ascending=False,
        method="min"
    )
)

df["verify_rank"] = (
    df["verify"]
    .rank(
        ascending=False,
        method="min"
    )
)

df["rank_diff"] = (
    df["investment_rank"]
    - df["verify_rank"]
)

df["verify_gap"] = (
    df["investment_score"]
    - df["verify"]
)


df["verify_status"] = df.apply(
    lambda row: get_verify_status(
        row["verify_gap"],
        row["rank_diff"],
        row["verify_data"]
    ),
    axis=1
)


# ============================================================
# RECOMMENDATIONS
# ============================================================

df["recommendation"] = df.apply(
    lambda row:
        get_recommendation(
            row.get(
                "investment_score",
                np.nan
            ),
            row.get(
                "kpax",
                np.nan
            ),
            row.get(
                "kpax_fv",
                np.nan
            ),
            row.get(
                "technical",
                np.nan
            )
        ),
    axis=1
)


df["timing"] = df[
    "technical"
].apply(
    get_timing
)


df["turnaround"] = df.apply(
    get_turnaround_status,
    axis=1
)


# ============================================================
# FILTER
# ============================================================

display_df = df.copy()

if min_investment_score > 0:

    display_df = display_df[
        (
            display_df[
                "investment_score"
            ] >= min_investment_score
        )
    ]


display_df = display_df.sort_values(
    "investment_score",
    ascending=False
)


# ============================================================
# FORMAT TABLE
# ============================================================

table = pd.DataFrame({

    "Ticker":
        display_df["ticker"],

    "Name":
        display_df["name"],

    "Sector":
        display_df["sector"],

    "Investment":
        display_df["investment_score"],

    "KPAX":
        display_df["kpax"],

    "KPAX-FV":
        display_df["kpax_fv"],

    "Quality":
        display_df["quality"],

    "Future":
        display_df["future"],

    "Fair Value":
        display_df["fair_value_score"],

    "Relative Val.":

        display_df[
            "relative_valuation"
        ],

    "Risk":
        display_df["risk"],

    "Technical":
        display_df["technical"],

    "Recommendation":
        display_df["recommendation"],

    "Timing":
        display_df["timing"],

    "Turnaround":
        display_df["turnaround"],

    "Forward KGV":
        display_df["forward_pe"],

    "Daten":
        display_df["completeness"]
})


st.subheader(
    f"📈 Investment Ranking "
    f"({len(display_df)} / {len(df)})"
)


st.dataframe(
    table.style.format(
        {
            "Investment": "{:.0f}",
            "KPAX": "{:.0f}",
            "KPAX-FV": "{:.0f}",
            "Quality": "{:.0f}",
            "Future": "{:.0f}",
            "Fair Value": "{:.0f}",
            "Relative Val.": "{:.0f}",
            "Risk": "{:.0f}",
            "Technical": "{:.0f}",
            "Forward KGV": "{:.1f}",
            "Daten": "{:.0f}%"
        },
        na_rep="—"
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# VERIFY TABLE
# ============================================================

st.subheader(
    "🔎 Verify / Plausibilitätscheck"
)

st.caption(
    "Der Verify Score ist bewusst unabhängig vom "
    "Fair Value, Analystenzielen und technischen Signalen. "
    "Er dient ausschließlich dazu, auffällige KPAX-Rankings "
    "zu erkennen."
)


verify_display = display_df.copy()

verify_table = pd.DataFrame({

    "Ticker":
        verify_display["ticker"],

    "Investment":
        verify_display[
            "investment_score"
        ],

    "Verify":
        verify_display["verify"],

    "Gap":
        verify_display["verify_gap"],

    "Inv. Rang":
        verify_display[
            "investment_rank"
        ],

    "Verify Rang":
        verify_display[
            "verify_rank"
        ],

    "Rang-Diff":
        verify_display["rank_diff"],

    "Verify Daten":
        verify_display[
            "verify_data"
        ],

    "Plausibilität":
        verify_display[
            "verify_status"
        ]
})


st.dataframe(
    verify_table.style.format(
        {
            "Investment": "{:.0f}",
            "Verify": "{:.0f}",
            "Gap": "{:+.0f}",
            "Inv. Rang": "{:.0f}",
            "Verify Rang": "{:.0f}",
            "Rang-Diff": "{:+.0f}",
            "Verify Daten": "{:.0f}%"
        },
        na_rep="—"
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# VERIFY EXPLANATION
# ============================================================

with st.expander(
    "🔎 Verify-Mathematik anzeigen"
):

    st.markdown(
        """
### Grundidee

**KPAX entscheidet – Verify hinterfragt KPAX.**

Der Verify Score wird absichtlich einfacher berechnet
als der eigentliche Investment Score.

Er verwendet nur fünf fundamentale Bereiche:

| Bereich | Gewicht |
|---|---:|
| Earnings Growth | 25 % |
| Revenue Growth | 15 % |
| Profitability | 25 % |
| Debt | 15 % |
| Valuation | 20 % |

Die Gewichte werden automatisch neu normalisiert,
wenn einzelne Daten fehlen.

### 1. Earnings Growth

Bewertung des Gewinnwachstums:

`Growth Score = f(Earnings Growth)`

Dabei wird ein Bereich von stark negativem Wachstum
bis zu sehr starkem Wachstum auf 5–100 Punkte abgebildet.

### 2. Revenue Growth

Analog:

`Revenue Score = f(Revenue Growth)`

### 3. Profitability

Bei normalen Unternehmen:

`Profitability = 50 % ROIC + 30 % FCF-Marge + 20 % Gross Margin`

Bei Financials:

`Profitability = 60 % ROE + 40 % Net Margin`

### 4. Debt

Normale Unternehmen:

`Debt = 60 % Net Debt / EBITDA + 25 % Debt/Equity + 15 % Current Ratio`

Financials:

`Debt = 60 % Debt/Equity + 40 % Cash/Debt`

### 5. Valuation

Normale Unternehmen:

`Valuation = 50 % Forward-KGV + 30 % PEG + 20 % KUV`

Financials:

`Valuation = 60 % Forward-KGV + 40 % KBV`

### Verify Gap

`Verify Gap = Investment Score − Verify Score`

Interpretation:

- **0 bis ±8:** 🟢 plausibel
- **±8 bis ±15:** 🟡 genauer prüfen
- **> ±15:** 🔴 auffällig

### Rangvergleich

Zusätzlich wird der Investment-Rang mit dem
Verify-Rang verglichen.

`Rang-Diff = Investment Rang − Verify Rang`

Ein großer Unterschied bedeutet:

> Der Hauptscore bewertet die Aktie deutlich anders
> als die einfache fundamentale Plausibilitätsprüfung.

Das ist kein automatisches "Fehler"-Signal.

Es ist ein **Hinweis, wo man genauer hinschauen sollte**.

### Datenbasis

Der Verify Score wird außerdem mit einer eigenen
Datenbasis bewertet.

Unter 60 %:

`⚪ Datenbasis schwach`

Damit wird verhindert, dass eine Aktie wegen
fehlender Fundamentaldaten fälschlich als
"auffällig" markiert wird.
"""
    )


# ============================================================
# MAIN MODEL MATHEMATICS
# ============================================================

with st.expander(
    "📐 Mathematik & Berechnungslogik des Hauptmodells"
):

    st.markdown(
        """
## 1. KPAX

`KPAX = 40 % Quality + 60 % Future`

---

## 2. Quality

### Normale Unternehmen

`Quality = 30 % ROIC + 15 % Gross Margin + 20 % FCF Margin + 20 % Earnings Growth + 10 % Revenue Growth`

### Financials

`Quality = 40 % ROE + 30 % Net Margin + 20 % Earnings Growth + 10 % Revenue Growth`

Fehlende Komponenten werden herausgenommen und
die verbleibenden Gewichte automatisch normalisiert.

---

## 3. Future

`Future =`

- `30 % Earnings Growth`
- `20 % Revenue Growth`
- `15 % Long-Term Growth`
- `15 % Analyst Upside`
- `10 % FCF Margin`
- `10 % Growth Acceleration`

---

## 4. Relative Valuation

Normale Unternehmen:

`40 % Forward-KGV + 20 % Trailing-KGV + 20 % PEG + 20 % KUV`

Financials:

`40 % Forward-KGV + 20 % Trailing-KGV + 20 % PEG + 20 % KBV`

---

## 5. Fair Value

### Normale Unternehmen

`50 % Earnings Fair Value`

`30 % FCF Fair Value`

`20 % Analyst Target`

Wenn ein Modell nicht verfügbar ist,
werden die vorhandenen Modelle neu gewichtet.

### Earnings Fair Value

`Fair Value = Forward EPS × Fair P/E`

mit:

`Fair P/E = 10 + 0,40 × Growth`

begrenzter Bereich:

`10 ≤ Fair P/E ≤ 28`

### FCF Fair Value

`FCF Fair Value = FCF je Aktie / Ziel-FCF-Rendite`

mit:

`Target FCF Yield = 8 % − 0,15 × Growth`

begrenzt auf:

`3,5 % bis 10 %`

Bei höherem Beta wird zusätzlich ein
Risikoaufschlag berücksichtigt.

---

## 6. KPAX-FV

`KPAX-FV = 60 % Fair Value Score + 40 % Relative Valuation`

---

## 7. Investment Score

Die Standardgewichtung ist:

`40 % KPAX`

`50 % KPAX-FV`

`10 % Risk`

Die Gewichte können in der Sidebar verändert werden.

---

## 8. Technical Score

Der Technical Score besteht aus:

- 25 % SMA200
- 25 % 6-Monats-Performance
- 20 % RSI
- 20 % MACD
- 10 % SMA50

Der Technical Score geht **nicht** in den
Investment Score ein.

Er dient nur als Timing-/Empfehlungsfilter.

---

## 9. Empfehlung

### Strong Buy

`Investment ≥ 85`

und

`KPAX ≥ 80`

und

`KPAX-FV ≥ 80`

### Buy

`Investment ≥ 80`

und

`KPAX ≥ 75`

### Accumulate

`Investment ≥ 75`

### Hold

`Investment ≥ 68`

### Reduce / Watch

`Investment ≥ 55`

### Avoid

`Investment < 55`

Der Technical Score kann die Empfehlung anschließend
nach unten begrenzen.

---

## 10. Minimum Score

Alle gültigen Scores liegen zwischen:

`5 und 100`

Die **5 ist bewusst das Minimum für einen echten,
berechneten Score**.

`—` bedeutet dagegen:

**keine ausreichende Datenbasis / nicht berechenbar.**
"""
    )


# ============================================================
# SUMMARY
# ============================================================

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Analysierte Aktien",
        len(df)
    )

with col2:
    valid_investment = df[
        "investment_score"
    ].notna().sum()

    st.metric(
        "Investment Scores",
        valid_investment
    )

with col3:
    valid_verify = df[
        "verify"
    ].notna().sum()

    st.metric(
        "Verify Scores",
        valid_verify
    )

with col4:

    suspicious = (
        df["verify_status"]
        == "🔴 Auffällig"
    ).sum()

    st.metric(
        "Auffällig",
        suspicious
    )
