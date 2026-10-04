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
    page_title="Aktien-Screener V21.0",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# VERSION
# ============================================================

VERSION = "V21.0"


# ============================================================
# INITIAL SCORE WEIGHTS
# ============================================================

INITIAL_SCORE_WEIGHTS = {
    "kpax": 0.45,
    "kpax_fv": 0.45,
    "risk": 0.10
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

    "Technology": {
        "GOOGL", "NVDA", "AAPL", "MU", "INTC", "ASML", "TSM",
        "MRVL", "000660.KS", "005930.KS", "AMD", "MSFT", "AVGO",
        "IFX.DE", "NOK", "VRT", "1810.HK", "F8P", "SSUN.F"
    },

    "Consumer Cyclical": {
        "AMZN", "BMW.DE", "NKE", "TSLA", "MCD", "ADS.DE"
    },

    "Consumer Defensive": {
        "PEP", "KO"
    },

    "Healthcare": {
        "NVO"
    },

    "Energy": {
        "SU.PA"
    },

    "Utilities": {
        "VST"
    },

    "Industrials": {
        "SMO", "ENR.DE", "SIE.DE"
    },

    "Financial Services": {
        "MUV2.DE", "ALV.DE", "MAIR"
    }
}


# ============================================================
# SESSION STATE
# ============================================================

if "score_weights" not in st.session_state:
    st.session_state.score_weights = INITIAL_SCORE_WEIGHTS.copy()


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value):
    """
    Converts values safely to float.

    None / NaN / inf / invalid strings -> None
    """
    if value is None:
        return None

    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(x):
        return None

    return x


def is_valid_number(value):
    return safe_float(value) is not None


def clamp(value, low, high):
    x = safe_float(value)

    if x is None:
        return None

    return max(low, min(high, x))


def clamp_score(value):
    return clamp(value, 5.0, 100.0)


def normalize_weights(weights):
    clean = {}

    for key, value in weights.items():
        x = safe_float(value)

        if x is not None and x >= 0:
            clean[key] = x
        else:
            clean[key] = 0.0

    total = sum(clean.values())

    if total <= 0:
        n = len(clean)

        if n == 0:
            return clean

        return {k: 1.0 / n for k in clean}

    return {
        k: v / total
        for k, v in clean.items()
    }


def weighted_score(components, weights):
    """
    Calculates a weighted score.

    Missing values are NOT treated as zero.
    Missing components are renormalized here only for
    sub-models where partial availability is explicitly intended.

    The main Investment Score handles missing KPAX-FV separately.
    """

    numerator = 0.0
    denominator = 0.0

    for key, value in components.items():

        x = safe_float(value)
        w = safe_float(weights.get(key))

        if x is None or w is None or w <= 0:
            continue

        numerator += x * w
        denominator += w

    if denominator <= 0:
        return None

    return clamp_score(numerator / denominator)


def median_valid(values):
    valid = []

    for value in values:
        x = safe_float(value)

        if x is not None:
            valid.append(x)

    if not valid:
        return None

    return float(np.median(valid))


def mean_valid(values):
    valid = []

    for value in values:
        x = safe_float(value)

        if x is not None:
            valid.append(x)

    if not valid:
        return None

    return float(np.mean(valid))


def pct_change(current, previous):
    c = safe_float(current)
    p = safe_float(previous)

    if c is None or p is None or p == 0:
        return None

    return (c / p - 1.0) * 100.0


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_number(value, decimals=1):
    x = safe_float(value)

    if x is None:
        return "—"

    return f"{x:,.{decimals}f}"


def format_price(value):
    x = safe_float(value)

    if x is None:
        return "—"

    return f"{x:,.2f}"


def format_score(value):
    x = safe_float(value)

    if x is None:
        return "—"

    return f"{x:.1f}"


def format_pct(value):
    x = safe_float(value)

    if x is None:
        return "—"

    sign = "+" if x > 0 else ""

    return f"{sign}{x:.1f}%"


# ============================================================
# DATAFRAME VALUE EXTRACTION
# ============================================================

def normalize_label(label):
    """
    Normalizes Yahoo financial statement labels.
    """

    if label is None:
        return ""

    text = str(label).strip().lower()

    replacements = {
        " ": "",
        "_": "",
        "-": "",
        "/": "",
        "(": "",
        ")": "",
        ",": "",
        ".": ""
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def find_row(df, candidates):
    """
    Finds a financial statement row using normalized labels.
    """

    if df is None or not isinstance(df, pd.DataFrame):
        return None

    if df.empty:
        return None

    normalized_candidates = {
        normalize_label(x)
        for x in candidates
    }

    for idx in df.index:

        if normalize_label(idx) in normalized_candidates:
            return idx

    # fuzzy fallback
    for idx in df.index:

        current = normalize_label(idx)

        for candidate in normalized_candidates:

            if candidate in current or current in candidate:
                return idx

    return None


def get_row_values(df, candidates):
    """
    Returns valid numeric values from a financial statement row.
    Ordered by newest columns first.
    """

    if df is None or not isinstance(df, pd.DataFrame):
        return []

    if df.empty:
        return []

    row = find_row(df, candidates)

    if row is None:
        return []

    try:
        series = df.loc[row]
    except Exception:
        return []

    values = []

    try:
        columns = list(series.index)
        columns = sorted(
            columns,
            key=lambda x: str(x),
            reverse=True
        )
    except Exception:
        columns = list(series.index)

    for col in columns:

        value = safe_float(series[col])

        if value is not None:
            values.append(value)

    return values


def get_latest_row_value(df, candidates):
    values = get_row_values(df, candidates)

    if not values:
        return None

    return values[0]


def get_median_row_value(df, candidates):
    values = get_row_values(df, candidates)

    if not values:
        return None

    return median_valid(values)


# ============================================================
# TICKER / COMPANY HELPERS
# ============================================================

def get_info(ticker_obj):
    try:
        info = ticker_obj.info

        if isinstance(info, dict):
            return info

    except Exception:
        pass

    return {}


def get_fast_info(ticker_obj):
    try:
        fi = ticker_obj.fast_info

        if fi is None:
            return {}

        try:
            return dict(fi)
        except Exception:
            return {}

    except Exception:
        return {}


def get_sector(ticker, info):
    sector = info.get("sector")

    if sector:
        return str(sector)

    for sector_name, tickers in FALLBACK_SECTORS.items():

        if ticker in tickers:
            return sector_name

    return "Unknown"


def is_financial_sector(sector):
    return sector == "Financial Services"


# ============================================================
# PRICE DATA
# ============================================================

def get_history(ticker_obj, period="5y"):

    try:
        hist = ticker_obj.history(
            period=period,
            auto_adjust=False,
            actions=False
        )

        if hist is None or hist.empty:
            return pd.DataFrame()

        hist = hist.copy()

        hist = hist.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return hist.dropna(
            subset=["Close"],
            how="all"
        )

    except Exception:
        return pd.DataFrame()


def get_current_price(ticker_obj, hist=None, info=None):

    info = info or {}

    # 1. history
    if hist is not None and not hist.empty:

        try:
            close = safe_float(hist["Close"].dropna().iloc[-1])

            if close is not None and close > 0:
                return close

        except Exception:
            pass

    # 2. fast_info
    fast_info = get_fast_info(ticker_obj)

    for key in [
        "lastPrice",
        "last_price",
        "regularMarketPrice"
    ]:

        value = safe_float(fast_info.get(key))

        if value is not None and value > 0:
            return value

    # 3. info
    for key in [
        "currentPrice",
        "regularMarketPrice",
        "previousClose"
    ]:

        value = safe_float(info.get(key))

        if value is not None and value > 0:
            return value

    return None


# ============================================================
# FINANCIAL STATEMENTS
# ============================================================

def safe_statement(ticker_obj, attribute_name):

    try:

        df = getattr(ticker_obj, attribute_name)

        if df is None:
            return pd.DataFrame()

        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        return df.copy()

    except Exception:
        return pd.DataFrame()


def load_financials(ticker_obj):

    data = {}

    # Annual
    data["income"] = safe_statement(
        ticker_obj,
        "income_stmt"
    )

    data["balance"] = safe_statement(
        ticker_obj,
        "balance_sheet"
    )

    data["cashflow"] = safe_statement(
        ticker_obj,
        "cashflow"
    )

    # TTM fallback
    data["ttm_income"] = safe_statement(
        ticker_obj,
        "ttm_income_stmt"
    )

    data["ttm_cashflow"] = safe_statement(
        ticker_obj,
        "ttm_cashflow"
    )

    # Quarterly fallback
    data["quarterly_income"] = safe_statement(
        ticker_obj,
        "quarterly_income_stmt"
    )

    data["quarterly_cashflow"] = safe_statement(
        ticker_obj,
        "quarterly_cashflow"
    )

    return data


# ============================================================
# FUNDAMENTAL DATA
# ============================================================

def get_revenue(financials):

    candidates = [
        "Total Revenue",
        "Operating Revenue",
        "Revenue"
    ]

    value = get_latest_row_value(
        financials.get("income"),
        candidates
    )

    if value is not None:
        return value

    value = get_latest_row_value(
        financials.get("ttm_income"),
        candidates
    )

    if value is not None:
        return value

    return get_latest_row_value(
        financials.get("quarterly_income"),
        candidates
    )


def get_net_income(financials):

    candidates = [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests"
    ]

    value = get_latest_row_value(
        financials.get("income"),
        candidates
    )

    if value is not None:
        return value

    value = get_latest_row_value(
        financials.get("ttm_income"),
        candidates
    )

    if value is not None:
        return value

    return get_latest_row_value(
        financials.get("quarterly_income"),
        candidates
    )


def get_operating_income(financials):

    candidates = [
        "Operating Income",
        "EBIT"
    ]

    value = get_latest_row_value(
        financials.get("income"),
        candidates
    )

    if value is not None:
        return value

    value = get_latest_row_value(
        financials.get("ttm_income"),
        candidates
    )

    if value is not None:
        return value

    return get_latest_row_value(
        financials.get("quarterly_income"),
        candidates
    )


def get_gross_profit(financials):

    return get_latest_row_value(
        financials.get("income"),
        [
            "Gross Profit"
        ]
    )


def get_eps_from_statements(financials):

    candidates = [
        "Diluted EPS",
        "Basic EPS",
        "Diluted EPS Continuing Operations",
        "Basic EPS Continuing Operations"
    ]

    value = get_latest_row_value(
        financials.get("income"),
        candidates
    )

    if value is not None:
        return value

    value = get_latest_row_value(
        financials.get("ttm_income"),
        candidates
    )

    if value is not None:
        return value

    return get_latest_row_value(
        financials.get("quarterly_income"),
        candidates
    )


def get_eps(ticker_obj, info, financials):

    # 1. financial statement
    value = get_eps_from_statements(financials)

    if value is not None:
        return value

    # 2. info
    for key in [
        "trailingEps",
        "epsTrailingTwelveMonths"
    ]:

        value = safe_float(info.get(key))

        if value is not None:
            return value

    return None


def get_forward_eps(info):

    for key in [
        "forwardEps",
        "epsForward"
    ]:

        value = safe_float(info.get(key))

        if value is not None:
            return value

    return None


def get_growth(ticker_obj, info, financials):

    # 1. info growth
    for key in [
        "earningsGrowth",
        "earningsQuarterlyGrowth"
    ]:

        value = safe_float(info.get(key))

        if value is not None:
            return value * 100.0

    # 2. historical EPS growth
    eps_values = get_row_values(
        financials.get("income"),
        [
            "Diluted EPS",
            "Basic EPS"
        ]
    )

    if len(eps_values) >= 2:

        newest = eps_values[0]
        oldest = eps_values[-1]

        if (
            newest is not None
            and oldest is not None
            and oldest > 0
            and newest > 0
        ):

            years = max(
                1,
                len(eps_values) - 1
            )

            growth = (
                (newest / oldest) ** (1 / years) - 1
            ) * 100

            return clamp(
                growth,
                -50,
                100
            )

    return None


# ============================================================
# FREE CASH FLOW
# ============================================================

def get_fcf_values(financials):

    candidates = [
        "Free Cash Flow"
    ]

    values = get_row_values(
        financials.get("cashflow"),
        candidates
    )

    if values:
        return values

    values = get_row_values(
        financials.get("ttm_cashflow"),
        candidates
    )

    if values:
        return values

    values = get_row_values(
        financials.get("quarterly_cashflow"),
        candidates
    )

    if values:
        return values

    # calculate FCF = Operating Cash Flow - CapEx
    ocf = get_row_values(
        financials.get("cashflow"),
        [
            "Operating Cash Flow",
            "Total Cash From Operating Activities"
        ]
    )

    capex = get_row_values(
        financials.get("cashflow"),
        [
            "Capital Expenditure",
            "Capital Expenditures"
        ]
    )

    if ocf and capex:

        n = min(
            len(ocf),
            len(capex)
        )

        calculated = []

        for i in range(n):

            fcf = ocf[i] - abs(capex[i])

            if np.isfinite(fcf):
                calculated.append(fcf)

        if calculated:
            return calculated

    return []


def get_normalized_fcf(financials):

    values = get_fcf_values(financials)

    positive = [
        x for x in values
        if safe_float(x) is not None
        and x > 0
    ]

    if not positive:
        return None

    # maximum 3 latest annual values
    positive = positive[:3]

    return median_valid(positive)


# ============================================================
# BALANCE SHEET
# ============================================================

def get_cash(financials):

    return get_latest_row_value(
        financials.get("balance"),
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
            "Cash Financial",
            "Cash"
        ]
    )


def get_debt(financials):

    # Prefer total debt
    value = get_latest_row_value(
        financials.get("balance"),
        [
            "Total Debt"
        ]
    )

    if value is not None:
        return abs(value)

    # fallback current + long-term debt
    current = get_latest_row_value(
        financials.get("balance"),
        [
            "Current Debt",
            "Current Debt And Capital Lease Obligation",
            "Current Debt And Lease Obligation"
        ]
    )

    long_term = get_latest_row_value(
        financials.get("balance"),
        [
            "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation",
            "Long Term Debt And Lease Obligation"
        ]
    )

    if current is not None or long_term is not None:

        return (
            abs(current or 0)
            + abs(long_term or 0)
        )

    return None


def get_equity(financials):

    return get_latest_row_value(
        financials.get("balance"),
        [
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest"
        ]
    )


def get_current_assets(financials):

    return get_latest_row_value(
        financials.get("balance"),
        [
            "Current Assets"
        ]
    )


def get_current_liabilities(financials):

    return get_latest_row_value(
        financials.get("balance"),
        [
            "Current Liabilities"
        ]
    )


# ============================================================
# SHARES / MARKET CAP
# ============================================================

def get_shares(ticker_obj, info, financials, price):

    # 1. income statement
    value = get_latest_row_value(
        financials.get("income"),
        [
            "Diluted Average Shares",
            "Basic Average Shares",
            "Diluted Average Shares Outstanding"
        ]
    )

    if value is not None and value > 0:
        return value

    # 2. info
    for key in [
        "sharesOutstanding",
        "impliedSharesOutstanding"
    ]:

        value = safe_float(info.get(key))

        if value is not None and value > 0:
            return value

    # 3. market cap / price
    market_cap = safe_float(
        info.get("marketCap")
    )

    price = safe_float(price)

    if (
        market_cap is not None
        and market_cap > 0
        and price is not None
        and price > 0
    ):
        return market_cap / price

    return None


def get_market_cap(info, price, shares):

    market_cap = safe_float(
        info.get("marketCap")
    )

    if market_cap is not None and market_cap > 0:
        return market_cap

    price = safe_float(price)
    shares = safe_float(shares)

    if (
        price is not None
        and shares is not None
        and price > 0
        and shares > 0
    ):
        return price * shares

    return None


# ============================================================
# PROFITABILITY
# ============================================================

def get_gross_margin(revenue, gross_profit):

    revenue = safe_float(revenue)
    gross_profit = safe_float(gross_profit)

    if (
        revenue is None
        or gross_profit is None
        or revenue == 0
    ):
        return None

    return (
        gross_profit / revenue
    ) * 100


def get_net_margin(revenue, net_income):

    revenue = safe_float(revenue)
    net_income = safe_float(net_income)

    if (
        revenue is None
        or net_income is None
        or revenue == 0
    ):
        return None

    return (
        net_income / revenue
    ) * 100


def get_fcf_margin(revenue, normalized_fcf):

    revenue = safe_float(revenue)
    fcf = safe_float(normalized_fcf)

    if (
        revenue is None
        or fcf is None
        or revenue == 0
    ):
        return None

    return (
        fcf / revenue
    ) * 100


def get_roa(net_income, equity, debt):

    net_income = safe_float(net_income)
    equity = safe_float(equity)
    debt = safe_float(debt)

    if (
        net_income is None
        or equity is None
        or equity <= 0
    ):
        return None

    denominator = equity

    if debt is not None and debt > 0:
        denominator += debt

    if denominator <= 0:
        return None

    return (
        net_income / denominator
    ) * 100


def get_roic(
    operating_income,
    tax_rate,
    equity,
    debt,
    cash
):

    ebit = safe_float(operating_income)
    tax = safe_float(tax_rate)
    equity = safe_float(equity)
    debt = safe_float(debt)
    cash = safe_float(cash)

    if (
        ebit is None
        or equity is None
        or equity <= 0
    ):
        return None

    if tax is None:
        tax = 20.0

    nopat = ebit * (
        1 - tax / 100
    )

    invested_capital = equity + (
        debt if debt is not None else 0
    ) - (
        cash if cash is not None else 0
    )

    if invested_capital <= 0:
        return None

    return (
        nopat / invested_capital
    ) * 100


# ============================================================
# VALUATION DATA
# ============================================================

def get_trailing_pe(price, eps, info):

    # Yahoo direct value
    value = safe_float(
        info.get("trailingPE")
    )

    if value is not None and value > 0:
        return value

    # calculate
    price = safe_float(price)
    eps = safe_float(eps)

    if (
        price is not None
        and eps is not None
        and eps > 0
    ):
        return price / eps

    return None


def get_forward_pe(price, forward_eps, info):

    value = safe_float(
        info.get("forwardPE")
    )

    if value is not None and value > 0:
        return value

    price = safe_float(price)
    forward_eps = safe_float(forward_eps)

    if (
        price is not None
        and forward_eps is not None
        and forward_eps > 0
    ):
        return price / forward_eps

    return None


def get_price_sales(
    market_cap,
    revenue,
    info
):

    value = safe_float(
        info.get(
            "priceToSalesTrailing12Months"
        )
    )

    if value is not None and value > 0:
        return value

    market_cap = safe_float(market_cap)
    revenue = safe_float(revenue)

    if (
        market_cap is not None
        and revenue is not None
        and market_cap > 0
        and revenue > 0
    ):
        return market_cap / revenue

    return None


def get_price_book(
    market_cap,
    equity,
    info
):

    value = safe_float(
        info.get("priceToBook")
    )

    if value is not None and value > 0:
        return value

    market_cap = safe_float(market_cap)
    equity = safe_float(equity)

    if (
        market_cap is not None
        and equity is not None
        and market_cap > 0
        and equity > 0
    ):
        return market_cap / equity

    return None


def get_peg(
    info,
    forward_pe,
    growth
):

    value = safe_float(
        info.get("pegRatio")
    )

    if value is not None and value > 0:
        return value

    forward_pe = safe_float(forward_pe)
    growth = safe_float(growth)

    if (
        forward_pe is None
        or growth is None
        or growth <= 0
    ):
        return None

    return forward_pe / growth


# ============================================================
# ANALYST TARGET
# ============================================================

def get_analyst_target(ticker_obj, info):

    # Current yfinance API
    try:
        targets = ticker_obj.analyst_price_targets

        if isinstance(targets, dict):

            mean = safe_float(
                targets.get("mean")
            )

            median = safe_float(
                targets.get("median")
            )

            result = median_valid([
                mean,
                median
            ])

            if result is not None and result > 0:
                return result

    except Exception:
        pass

    # info fallback
    values = [
        safe_float(
            info.get("targetMeanPrice")
        ),
        safe_float(
            info.get("targetMedianPrice")
        )
    ]

    result = median_valid(values)

    if result is not None and result > 0:
        return result

    return None


# ============================================================
# FAIR VALUE
# ============================================================

def stable_fair_pe(growth):

    growth = safe_float(growth)

    if growth is None:
        return 16.0

    fair_pe = (
        12.0
        + 0.25 * growth
    )

    return clamp(
        fair_pe,
        12.0,
        24.0
    )


def earnings_fair_value(
    price,
    forward_eps,
    trailing_eps,
    growth
):

    eps = safe_float(forward_eps)
    source = "forward"

    if eps is None:
        eps = safe_float(trailing_eps)
        source = "trailing"

    if eps is None or eps <= 0:
        return None, None, None

    fair_pe = stable_fair_pe(growth)

    fair_value = eps * fair_pe

    if fair_value <= 0:
        return None, None, None

    return fair_value, fair_pe, source


def fcf_fair_value(
    normalized_fcf,
    shares,
    growth,
    beta
):

    fcf = safe_float(normalized_fcf)
    shares = safe_float(shares)
    growth = safe_float(growth)
    beta = safe_float(beta)

    if (
        fcf is None
        or shares is None
        or fcf <= 0
        or shares <= 0
    ):
        return None, None

    fcf_per_share = (
        fcf / shares
    )

    if growth is None:
        growth = 10.0

    target_yield = (
        7.0
        - 0.08 * growth
    )

    target_yield = clamp(
        target_yield,
        4.5,
        9.0
    )

    if beta is not None and beta > 1.0:
        target_yield += (
            beta - 1.0
        ) * 0.75

    target_yield = clamp(
        target_yield,
        4.5,
        11.0
    )

    fair_value = (
        fcf_per_share
        / (target_yield / 100.0)
    )

    if fair_value <= 0:
        return None, None

    return fair_value, target_yield


def fair_value_model(
    price,
    forward_eps,
    trailing_eps,
    normalized_fcf,
    shares,
    growth,
    analyst_target,
    beta,
    financial
):

    earnings_fv, fair_pe, earnings_source = (
        earnings_fair_value(
            price,
            forward_eps,
            trailing_eps,
            growth
        )
    )

    fcf_fv, target_yield = (
        fcf_fair_value(
            normalized_fcf,
            shares,
            growth,
            beta
        )
    )

    analyst_fv = safe_float(
        analyst_target
    )

    candidates = []

    if earnings_fv is not None:
        candidates.append(
            earnings_fv
        )

    if fcf_fv is not None:
        candidates.append(
            fcf_fv
        )

    if analyst_fv is not None and analyst_fv > 0:
        candidates.append(
            analyst_fv
        )

    if not candidates:
        return {
            "fair_value": None,
            "fair_value_score": None,
            "earnings_fv": None,
            "fcf_fv": None,
            "analyst_fv": analyst_fv,
            "normalized_fcf": normalized_fcf,
            "model_count": 0,
            "model_confidence": 0.0,
            "earnings_source": earnings_source,
            "fair_pe": fair_pe,
            "target_yield": target_yield
        }

    # Model weights
    if financial:

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

    models = {
        "earnings": earnings_fv,
        "fcf": fcf_fv,
        "analyst": analyst_fv
    }

    weighted_values = []
    weighted_weights = []

    for key, value in models.items():

        x = safe_float(value)

        if x is None or x <= 0:
            continue

        weighted_values.append(
            x * weights[key]
        )

        weighted_weights.append(
            weights[key]
        )

    if not weighted_weights:
        fair_value = None

    else:
        fair_value = (
            sum(weighted_values)
            / sum(weighted_weights)
        )

    if fair_value is None:
        return {
            "fair_value": None,
            "fair_value_score": None,
            "earnings_fv": earnings_fv,
            "fcf_fv": fcf_fv,
            "analyst_fv": analyst_fv,
            "normalized_fcf": normalized_fcf,
            "model_count": len(candidates),
            "model_confidence": 0.0,
            "earnings_source": earnings_source,
            "fair_pe": fair_pe,
            "target_yield": target_yield
        }

    # Robust bounds
    price = safe_float(price)

    if price is not None and price > 0:

        fair_value = clamp(
            fair_value,
            price * 0.25,
            price * 2.50
        )

    # median bound
    median_model = median_valid(
        candidates
    )

    if median_model is not None:

        fair_value = clamp(
            fair_value,
            median_model * 0.50,
            median_model * 1.80
        )

    if price is not None and price > 0:

        fair_value = clamp(
            fair_value,
            price * 0.40,
            price * 2.50
        )

    if price is not None and price > 0:

        upside = (
            fair_value / price - 1
        ) * 100

        fv_score = clamp_score(
            55.0
            + 0.60 * upside
        )

    else:
        fv_score = None

    # Confidence
    model_count = len(candidates)

    if model_count >= 3:
        model_confidence = 100.0

    elif model_count == 2:
        model_confidence = 75.0

    elif model_count == 1:
        model_confidence = 45.0

    else:
        model_confidence = 0.0

    return {
        "fair_value": fair_value,
        "fair_value_score": fv_score,
        "earnings_fv": earnings_fv,
        "fcf_fv": fcf_fv,
        "analyst_fv": analyst_fv,
        "normalized_fcf": normalized_fcf,
        "model_count": model_count,
        "model_confidence": model_confidence,
        "earnings_source": earnings_source,
        "fair_pe": fair_pe,
        "target_yield": target_yield
    }


# ============================================================
# RELATIVE VALUATION
# ============================================================

def sigmoid_score(
    value,
    fair_value,
    sensitivity=1.0
):

    value = safe_float(value)
    fair_value = safe_float(fair_value)

    if (
        value is None
        or fair_value is None
        or value <= 0
        or fair_value <= 0
    ):
        return None

    ratio = value / fair_value

    try:

        exponent = (
            (ratio - 1.0)
            * sensitivity
        )

        exponent = clamp(
            exponent,
            -20,
            20
        )

        score = (
            100.0
            / (
                1.0
                + math.exp(exponent)
            )
        )

        return clamp_score(
            score
        )

    except Exception:
        return None


def relative_valuation(
    forward_pe,
    trailing_pe,
    peg,
    price_sales,
    price_book,
    financial
):

    components = {}

    if financial:

        # Lower PE is better
        if forward_pe is not None:
            components["forward_pe"] = (
                sigmoid_score(
                    forward_pe,
                    18.0,
                    4.0
                )
            )

        if trailing_pe is not None:
            components["trailing_pe"] = (
                sigmoid_score(
                    trailing_pe,
                    20.0,
                    3.0
                )
            )

        if peg is not None:
            components["peg"] = (
                sigmoid_score(
                    peg,
                    1.5,
                    3.0
                )
            )

        if price_book is not None:
            components["price_book"] = (
                sigmoid_score(
                    price_book,
                    2.0,
                    2.5
                )
            )

        weights = {
            "forward_pe": 0.35,
            "trailing_pe": 0.15,
            "peg": 0.15,
            "price_book": 0.35
        }

    else:

        if forward_pe is not None:
            components["forward_pe"] = (
                sigmoid_score(
                    forward_pe,
                    20.0,
                    4.0
                )
            )

        if trailing_pe is not None:
            components["trailing_pe"] = (
                sigmoid_score(
                    trailing_pe,
                    22.0,
                    3.0
                )
            )

        if peg is not None:
            components["peg"] = (
                sigmoid_score(
                    peg,
                    1.5,
                    3.0
                )
            )

        if price_sales is not None:
            components["price_sales"] = (
                sigmoid_score(
                    price_sales,
                    4.0,
                    2.0
                )
            )

        weights = {
            "forward_pe": 0.35,
            "trailing_pe": 0.10,
            "peg": 0.30,
            "price_sales": 0.25
        }

    valid_components = {
        k: v
        for k, v in components.items()
        if v is not None
    }

    if not valid_components:
        return None, 0.0

    score = weighted_score(
        valid_components,
        weights
    )

    availability = (
        sum(
            weights[k]
            for k in valid_components
        )
        / sum(weights.values())
    ) * 100.0

    return score, availability


# ============================================================
# QUALITY
# ============================================================

def quality_score(
    financial,
    roic,
    roa,
    roe,
    gross_margin,
    net_margin,
    fcf_margin,
    earnings_growth,
    revenue_growth
):

    if financial:

        components = {
            "roe": roe,
            "net_margin": net_margin,
            "earnings_growth": earnings_growth,
            "revenue_growth": revenue_growth
        }

        weights = {
            "roe": 0.40,
            "net_margin": 0.30,
            "earnings_growth": 0.20,
            "revenue_growth": 0.10
        }

        scores = {}

        if roe is not None:
            scores["roe"] = clamp_score(
                50 + roe * 2.0
            )

        if net_margin is not None:
            scores["net_margin"] = clamp_score(
                50 + net_margin * 2.0
            )

    else:

        components = {
            "roic": roic,
            "gross_margin": gross_margin,
            "fcf_margin": fcf_margin,
            "earnings_growth": earnings_growth,
            "revenue_growth": revenue_growth
        }

        weights = {
            "roic": 0.30,
            "gross_margin": 0.15,
            "fcf_margin": 0.20,
            "earnings_growth": 0.20,
            "revenue_growth": 0.15
        }

        scores = {}

        if roic is not None:
            scores["roic"] = clamp_score(
                50 + roic * 2.0
            )

        if gross_margin is not None:
            scores["gross_margin"] = clamp_score(
                30 + gross_margin * 0.80
            )

        if fcf_margin is not None:
            scores["fcf_margin"] = clamp_score(
                50 + fcf_margin * 2.0
            )

    # Growth
    if earnings_growth is not None:
        scores["earnings_growth"] = clamp_score(
            50 + earnings_growth * 1.5
        )

    if revenue_growth is not None:
        scores["revenue_growth"] = clamp_score(
            50 + revenue_growth * 1.5
        )

    return weighted_score(
        scores,
        weights
    )


# ============================================================
# FUTURE SCORE
# ============================================================

def future_score(
    earnings_growth,
    revenue_growth,
    fcf_margin,
    analyst_target,
    price
):

    scores = {}

    if earnings_growth is not None:
        scores["earnings"] = clamp_score(
            50 + earnings_growth * 1.5
        )

    if revenue_growth is not None:
        scores["revenue"] = clamp_score(
            50 + revenue_growth * 1.2
        )

    if fcf_margin is not None:
        scores["fcf"] = clamp_score(
            50 + fcf_margin * 1.5
        )

    price = safe_float(price)
    target = safe_float(analyst_target)

    if (
        price is not None
        and target is not None
        and price > 0
        and target > 0
    ):

        analyst_upside = (
            target / price - 1
        ) * 100

        scores["analyst"] = clamp_score(
            55 + analyst_upside * 0.5
        )

    weights = {
        "earnings": 0.35,
        "revenue": 0.20,
        "fcf": 0.20,
        "analyst": 0.25
    }

    return weighted_score(
        scores,
        weights
    )


# ============================================================
# KPAX
# ============================================================

def kpax_score(
    quality,
    future
):

    components = {
        "quality": quality,
        "future": future
    }

    weights = {
        "quality": 0.40,
        "future": 0.60
    }

    return weighted_score(
        components,
        weights
    )


# ============================================================
# KPAX FAIR VALUE
# ============================================================

def kpax_fv_score(
    fair_value_score,
    relative_score
):

    components = {
        "fair_value": fair_value_score,
        "relative": relative_score
    }

    weights = {
        "fair_value": 0.55,
        "relative": 0.45
    }

    score = weighted_score(
        components,
        weights
    )

    # availability
    available = 0

    if fair_value_score is not None:
        available += 0.55

    if relative_score is not None:
        available += 0.45

    availability = (
        available / 1.0
    ) * 100.0

    return score, availability


# ============================================================
# RISK
# ============================================================

def get_beta(info):

    for key in [
        "beta",
        "beta3Year"
    ]:

        value = safe_float(
            info.get(key)
        )

        if value is not None:
            return value

    return None


def risk_score(
    hist,
    beta,
    debt_to_equity
):

    market_score = None
    stability_score = None
    financial_score = None

    # Market risk
    if hist is not None and not hist.empty:

        try:

            close = (
                hist["Close"]
                .dropna()
            )

            returns = (
                close
                .pct_change()
                .dropna()
            )

            if len(returns) > 30:

                volatility = (
                    returns.std()
                    * math.sqrt(252)
                    * 100
                )

                rolling_max = (
                    close
                    .cummax()
                )

                drawdown = (
                    close / rolling_max - 1
                )

                max_drawdown = (
                    abs(drawdown.min())
                    * 100
                )

                vol_score = clamp_score(
                    100 - volatility * 1.8
                )

                dd_score = clamp_score(
                    100 - max_drawdown * 0.9
                )

                market_score = (
                    vol_score * 0.55
                    + dd_score * 0.45
                )

                stability_score = (
                    100
                    - max_drawdown * 0.8
                )

                stability_score = clamp_score(
                    stability_score
                )

        except Exception:
            pass

    # Beta fallback
    if market_score is None and beta is not None:

        market_score = clamp_score(
            100 - max(
                0,
                beta - 0.8
            ) * 45
        )

    # Financial risk
    if debt_to_equity is not None:

        financial_score = clamp_score(
            100
            - min(
                100,
                max(
                    0,
                    debt_to_equity
                ) * 0.5
            )
        )

    components = {
        "financial": financial_score,
        "market": market_score,
        "stability": stability_score
    }

    weights = {
        "financial": 0.45,
        "market": 0.35,
        "stability": 0.20
    }

    return weighted_score(
        components,
        weights
    )


# ============================================================
# TECHNICAL
# ============================================================

def technical_score(hist):

    if hist is None or hist.empty:
        return None

    try:

        close = (
            hist["Close"]
            .dropna()
        )

        if len(close) < 50:
            return None

        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]

        price = close.iloc[-1]

        score = 50.0

        if price > ma20:
            score += 20

        else:
            score -= 20

        if price > ma50:
            score += 20

        else:
            score -= 20

        if ma20 > ma50:
            score += 10

        else:
            score -= 10

        return clamp_score(
            score
        )

    except Exception:
        return None


# ============================================================
# INVESTMENT SCORE
# ============================================================

def investment_score(
    kpax,
    kpax_fv,
    risk
):

    weights = normalize_weights(
        st.session_state.score_weights
    )

    # IMPORTANT:
    # Missing KPAX-FV is NOT renormalized away.
    #
    # 50 = neutral information placeholder.
    #
    # This prevents:
    # KPAX 95 + Risk 66 -> artificial Strong Buy
    # when the entire valuation block is missing.

    valuation_component = (
        kpax_fv
        if kpax_fv is not None
        else 50.0
    )

    components = {
        "kpax": kpax,
        "kpax_fv": valuation_component,
        "risk": risk
    }

    score = (
        safe_float(kpax) * weights["kpax"]
        if kpax is not None
        else 0
    )

    score += (
        valuation_component
        * weights["kpax_fv"]
    )

    score += (
        safe_float(risk)
        * weights["risk"]
        if risk is not None
        else 0
    )

    # If KPAX itself is missing, score should not pretend
    if kpax is None:

        available_weight = (
            weights["kpax_fv"]
            + weights["risk"]
        )

        if available_weight <= 0:
            return None

    return clamp_score(
        score
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def recommendation(
    investment,
    technical,
    kpax_fv_available
):

    score = safe_float(
        investment
    )

    if score is None:
        return "—"

    if score >= 85:
        rec = "Strong Buy"

    elif score >= 80:
        rec = "Buy"

    elif score >= 75:
        rec = "Accumulate"

    elif score >= 68:
        rec = "Hold"

    elif score >= 55:
        rec = "Reduce / Watch"

    else:
        rec = "Avoid"

    # Technical timing filter
    tech = safe_float(
        technical
    )

    if tech is not None:

        if tech < 30:
            rec = min_recommendation(
                rec,
                "Hold"
            )

        elif tech < 45:
            rec = min_recommendation(
                rec,
                "Accumulate"
            )

        elif tech < 60:
            rec = min_recommendation(
                rec,
                "Buy"
            )

    # CRITICAL:
    # No complete valuation -> no Strong Buy
    if not kpax_fv_available:

        if rec == "Strong Buy":
            rec = "Buy"

        if rec == "Buy":
            rec = "Accumulate"

    return rec


def recommendation_rank(rec):

    ranks = {
        "Avoid": 0,
        "Reduce / Watch": 1,
        "Hold": 2,
        "Accumulate": 3,
        "Buy": 4,
        "Strong Buy": 5,
        "—": -1
    }

    return ranks.get(
        rec,
        -1
    )


def min_recommendation(
    current,
    maximum
):

    if (
        recommendation_rank(current)
        > recommendation_rank(maximum)
    ):
        return maximum

    return current


# ============================================================
# VERIFY
# ============================================================

def verify_score(
    financial,
    earnings_growth,
    revenue_growth,
    roic,
    roe,
    net_margin,
    fcf_margin,
    debt_to_equity,
    cash_to_debt,
    current_ratio,
    forward_pe,
    trailing_pe,
    peg,
    price_sales,
    price_book
):

    components = {}

    # Growth
    growth_scores = []

    if earnings_growth is not None:
        growth_scores.append(
            clamp_score(
                50 + earnings_growth * 1.5
            )
        )

    if revenue_growth is not None:
        growth_scores.append(
            clamp_score(
                50 + revenue_growth * 1.2
            )
        )

    if growth_scores:
        components["growth"] = mean_valid(
            growth_scores
        )

    # Profitability
    profitability_scores = []

    if financial:

        if roe is not None:
            profitability_scores.append(
                clamp_score(
                    50 + roe * 2
                )
            )

        if net_margin is not None:
            profitability_scores.append(
                clamp_score(
                    50 + net_margin * 2
                )

    else:

        if roic is not None:
            profitability_scores.append(
                clamp_score(
                    50 + roic * 2
                )
            )

        if fcf_margin is not None:
            profitability_scores.append(
                clamp_score(
                    50 + fcf_margin * 2
                )
            )

    if profitability_scores:
        components["profitability"] = (
            mean_valid(
                profitability_scores
            )
        )

    # Debt
    debt_scores = []

    if debt_to_equity is not None:

        debt_scores.append(
            clamp_score(
                100
                - max(
                    0,
                    debt_to_equity
                ) * 0.5
            )
        )

    if cash_to_debt is not None:

        debt_scores.append(
            clamp_score(
                50
                + cash_to_debt * 25
            )
        )

    if not financial and current_ratio is not None:

        debt_scores.append(
            clamp_score(
                40
                + current_ratio * 25
            )
        )

    if debt_scores:
        components["debt"] = mean_valid(
            debt_scores
        )

    # Valuation
    valuation_scores = []

    if forward_pe is not None:

        valuation_scores.append(
            sigmoid_score(
                forward_pe,
                20,
                4
            )
        )

    if trailing_pe is not None:

        valuation_scores.append(
            sigmoid_score(
                trailing_pe,
                22,
                3
            )
        )

    if peg is not None:

        valuation_scores.append(
            sigmoid_score(
                peg,
                1.5,
                3
            )
        )

    if financial and price_book is not None:

        valuation_scores.append(
            sigmoid_score(
                price_book,
                2.0,
                2.5
            )
        )

    if not financial and price_sales is not None:

        valuation_scores.append(
            sigmoid_score(
                price_sales,
                4.0,
                2
            )
        )

    valuation_scores = [
        x for x in valuation_scores
        if x is not None
    ]

    if valuation_scores:

        components["valuation"] = mean_valid(
            valuation_scores
        )

    verify = weighted_score(
        components,
        {
            "growth": 0.25,
            "profitability": 0.25,
            "debt": 0.15,
            "valuation": 0.35
        }
    )

    data_completeness = (
        len(components) / 4
    ) * 100.0

    return verify, data_completeness


# ============================================================
# MODEL CONFIDENCE
# ============================================================

def model_confidence(
    component_availability,
    fair_value_confidence,
    verify_completeness,
    history_length
):

    history_score = clamp(
        (
            safe_float(history_length)
            or 0
        ) / 1000 * 100,
        0,
        100
    )

    score = (
        (
            safe_float(
                component_availability
            ) or 0
        ) * 0.30
        +
        (
            safe_float(
                fair_value_confidence
            ) or 0
        ) * 0.35
        +
        (
            safe_float(
                verify_completeness
            ) or 0
        ) * 0.20
        +
        history_score * 0.15
    )

    return clamp_score(
        score
    )


# ============================================================
# DATA QUALITY
# ============================================================

def data_quality(
    price,
    revenue,
    eps,
    fcf,
    equity,
    debt,
    growth,
    fair_value,
    relative_valuation,
    analyst_target
):

    checks = [
        price,
        revenue,
        eps,
        fcf,
        equity,
        debt,
        growth,
        fair_value,
        relative_valuation,
        analyst_target
    ]

    available = sum(
        x is not None
        for x in checks
    )

    return (
        available
        / len(checks)
    ) * 100.0


# ============================================================
# FULL TICKER ANALYSIS
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def analyze_ticker(ticker):

    result = {
        "Ticker": ticker,
        "Price": None,
        "Fair Value": None,
        "FV Upside": None,
        "FV Score": None,
        "Relative Valuation": None,
        "KPAX": None,
        "KPAX-FV": None,
        "Risk": None,
        "Technical": None,
        "Investment": None,
        "Recommendation": "—",
        "Verify": None,
        "Gap": None,
        "Status": "weak",
        "Model Confidence": None,
        "Sector": "Unknown",
        "Data Quality": 0.0,

        # diagnostics
        "Revenue": None,
        "EPS": None,
        "Forward EPS": None,
        "FCF": None,
        "Equity": None,
        "Debt": None,
        "Cash": None,
        "Growth": None,
        "Forward PE": None,
        "Trailing PE": None,
        "PEG": None,
        "P/S": None,
        "P/B": None,
        "Analyst Target": None,
        "FV Models": 0,
        "FV Availability": 0.0,
        "Relative Availability": 0.0,
        "History Days": 0
    }

    try:

        yf_ticker = yf.Ticker(
            ticker
        )

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        info = get_info(
            yf_ticker
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        hist = get_history(
            yf_ticker,
            "5y"
        )

        result["History Days"] = (
            len(hist)
            if hist is not None
            else 0
        )

        price = get_current_price(
            yf_ticker,
            hist,
            info
        )

        result["Price"] = price

        if price is None:
            return result

        # ----------------------------------------------------
        # FINANCIAL STATEMENTS
        # ----------------------------------------------------

        financials = load_financials(
            yf_ticker
        )

        sector = get_sector(
            ticker,
            info
        )

        financial = is_financial_sector(
            sector
        )

        result["Sector"] = sector

        # ----------------------------------------------------
        # FUNDAMENTALS
        # ----------------------------------------------------

        revenue = get_revenue(
            financials
        )

        net_income = get_net_income(
            financials
        )

        operating_income = get_operating_income(
            financials
        )

        gross_profit = get_gross_profit(
            financials
        )

        trailing_eps = get_eps(
            yf_ticker,
            info,
            financials
        )

        forward_eps = get_forward_eps(
            info
        )

        growth = get_growth(
            yf_ticker,
            info,
            financials
        )

        normalized_fcf = get_normalized_fcf(
            financials
        )

        equity = get_equity(
            financials
        )

        debt = get_debt(
            financials
        )

        cash = get_cash(
            financials
        )

        current_assets = get_current_assets(
            financials
        )

        current_liabilities = (
            get_current_liabilities(
                financials
            )
        )

        shares = get_shares(
            yf_ticker,
            info,
            financials,
            price
        )

        market_cap = get_market_cap(
            info,
            price,
            shares
        )

        # ----------------------------------------------------
        # PROFITABILITY
        # ----------------------------------------------------

        gross_margin = get_gross_margin(
            revenue,
            gross_profit
        )

        net_margin = get_net_margin(
            revenue,
            net_income
        )

        fcf_margin = get_fcf_margin(
            revenue,
            normalized_fcf
        )

        tax_rate = safe_float(
            info.get(
                "effectiveTaxRate"
            )
        )

        if tax_rate is not None:
            tax_rate *= 100.0

        roic = get_roic(
            operating_income,
            tax_rate,
            equity,
            debt,
            cash
        )

        roa = get_roa(
            net_income,
            equity,
            debt
        )

        roe = None

        if (
            net_income is not None
            and equity is not None
            and equity > 0
        ):

            roe = (
                net_income
                / equity
            ) * 100

        # Revenue growth from info
        revenue_growth = safe_float(
            info.get(
                "revenueGrowth"
            )
        )

        if revenue_growth is not None:
            revenue_growth *= 100.0

        # ----------------------------------------------------
        # VALUATION
        # ----------------------------------------------------

        forward_pe = get_forward_pe(
            price,
            forward_eps,
            info
        )

        trailing_pe = get_trailing_pe(
            price,
            trailing_eps,
            info
        )

        price_sales = get_price_sales(
            market_cap,
            revenue,
            info
        )

        price_book = get_price_book(
            market_cap,
            equity,
            info
        )

        beta = get_beta(
            info
        )

        peg = get_peg(
            info,
            forward_pe,
            growth
        )

        analyst_target = get_analyst_target(
            yf_ticker,
            info
        )

        # ----------------------------------------------------
        # FAIR VALUE
        # ----------------------------------------------------

        fv = fair_value_model(
            price=price,
            forward_eps=forward_eps,
            trailing_eps=trailing_eps,
            normalized_fcf=normalized_fcf,
            shares=shares,
            growth=growth,
            analyst_target=analyst_target,
            beta=beta,
            financial=financial
        )

        fair_value = fv["fair_value"]
        fair_value_score = fv[
            "fair_value_score"
        ]

        # ----------------------------------------------------
        # RELATIVE VALUATION
        # ----------------------------------------------------

        relative_score, relative_availability = (
            relative_valuation(
                forward_pe,
                trailing_pe,
                peg,
                price_sales,
                price_book,
                financial
            )
        )

        # ----------------------------------------------------
        # QUALITY
        # ----------------------------------------------------

        quality = quality_score(
            financial=financial,
            roic=roic,
            roa=roa,
            roe=roe,
            gross_margin=gross_margin,
            net_margin=net_margin,
            fcf_margin=fcf_margin,
            earnings_growth=growth,
            revenue_growth=revenue_growth
        )

        # ----------------------------------------------------
        # FUTURE
        # ----------------------------------------------------

        future = future_score(
            earnings_growth=growth,
            revenue_growth=revenue_growth,
            fcf_margin=fcf_margin,
            analyst_target=analyst_target,
            price=price
        )

        # ----------------------------------------------------
        # KPAX
        # ----------------------------------------------------

        kpax = kpax_score(
            quality,
            future
        )

        # ----------------------------------------------------
        # KPAX-FV
        # ----------------------------------------------------

        kpax_fv, kpax_fv_availability = (
            kpax_fv_score(
                fair_value_score,
                relative_score
            )
        )

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        debt_to_equity = None

        if (
            debt is not None
            and equity is not None
            and equity > 0
        ):

            debt_to_equity = (
                debt / equity
            ) * 100

        cash_to_debt = None

        if (
            cash is not None
            and debt is not None
            and debt > 0
        ):

            cash_to_debt = (
                cash / debt
            )

        current_ratio = None

        if (
            current_assets is not None
            and current_liabilities is not None
            and current_liabilities > 0
        ):

            current_ratio = (
                current_assets
                / current_liabilities
            )

        risk = risk_score(
            hist,
            beta,
            debt_to_equity
        )

        # ----------------------------------------------------
        # TECHNICAL
        # ----------------------------------------------------

        technical = technical_score(
            hist
        )

        # ----------------------------------------------------
        # INVESTMENT
        # ----------------------------------------------------

        investment = investment_score(
            kpax,
            kpax_fv,
            risk
        )

        # Complete valuation block?
        valuation_complete = (
            fair_value_score is not None
            and relative_score is not None
        )

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        rec = recommendation(
            investment,
            technical,
            valuation_complete
        )

        # ----------------------------------------------------
        # VERIFY
        # ----------------------------------------------------

        verify, verify_completeness = (
            verify_score(
                financial=financial,
                earnings_growth=growth,
                revenue_growth=revenue_growth,
                roic=roic,
                roe=roe,
                net_margin=net_margin,
                fcf_margin=fcf_margin,
                debt_to_equity=debt_to_equity,
                cash_to_debt=cash_to_debt,
                current_ratio=current_ratio,
                forward_pe=forward_pe,
                trailing_pe=trailing_pe,
                peg=peg,
                price_sales=price_sales,
                price_book=price_book
            )
        )

        gap = None

        if (
            investment is not None
            and verify is not None
        ):

            gap = (
                investment
                - verify
            )

        if verify_completeness < 60:
            status = "weak"

        elif gap is None:
            status = "yellow"

        elif abs(gap) <= 10:
            status = "green"

        elif abs(gap) <= 20:
            status = "yellow"

        else:
            status = "red"

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        quality_data = data_quality(
            price,
            revenue,
            trailing_eps,
            normalized_fcf,
            equity,
            debt,
            growth,
            fair_value,
            relative_score,
            analyst_target
        )

        component_values = [
            quality,
            future,
            kpax,
            fair_value_score,
            relative_score,
            risk,
            technical
        ]

        component_availability = (
            sum(
                x is not None
                for x in component_values
            )
            / len(component_values)
        ) * 100.0

        confidence = model_confidence(
            component_availability,
            fv["model_confidence"],
            verify_completeness,
            len(hist)
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        result.update({

            "Fair Value": fair_value,

            "FV Upside": (
                (
                    fair_value / price
                    - 1
                ) * 100
                if (
                    fair_value is not None
                    and price > 0
                )
                else None
            ),

            "FV Score": fair_value_score,

            "Relative Valuation": relative_score,

            "KPAX": kpax,

            "KPAX-FV": kpax_fv,

            "Risk": risk,

            "Technical": technical,

            "Investment": investment,

            "Recommendation": rec,

            "Verify": verify,

            "Gap": gap,

            "Status": status,

            "Model Confidence": confidence,

            "Data Quality": quality_data,

            "Revenue": revenue,

            "EPS": trailing_eps,

            "Forward EPS": forward_eps,

            "FCF": normalized_fcf,

            "Equity": equity,

            "Debt": debt,

            "Cash": cash,

            "Growth": growth,

            "Forward PE": forward_pe,

            "Trailing PE": trailing_pe,

            "PEG": peg,

            "P/S": price_sales,

            "P/B": price_book,

            "Analyst Target": analyst_target,

            "FV Models": fv[
                "model_count"
            ],

            "FV Availability": kpax_fv_availability,

            "Relative Availability": relative_availability
        })

        return result

    except Exception as e:

        result["Status"] = (
            "error"
        )

        result["Error"] = str(e)

        return result


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    f"📊 Aktien-Screener {VERSION}"
)

st.sidebar.caption(
    "Robuste Fundamentaldaten + KPAX + Fair Value"
)

st.sidebar.subheader(
    "Investment Score Gewichte"
)

w_kpax = st.sidebar.number_input(
    "KPAX",
    min_value=0.0,
    max_value=1.0,
    value=float(
        st.session_state.score_weights["kpax"]
    ),
    step=0.05,
    format="%.2f"
)

w_fv = st.sidebar.number_input(
    "KPAX-FV",
    min_value=0.0,
    max_value=1.0,
    value=float(
        st.session_state.score_weights["kpax_fv"]
    ),
    step=0.05,
    format="%.2f"
)

w_risk = st.sidebar.number_input(
    "Risk",
    min_value=0.0,
    max_value=1.0,
    value=float(
        st.session_state.score_weights["risk"]
    ),
    step=0.05,
    format="%.2f"
)

col_a, col_b = st.sidebar.columns(2)

with col_a:

    if st.button(
        "💾 Anwenden",
        use_container_width=True
    ):

        st.session_state.score_weights = {
            "kpax": w_kpax,
            "kpax_fv": w_fv,
            "risk": w_risk
        }

        st.cache_data.clear()

        st.rerun()

with col_b:

    if st.button(
        "↩ Reset",
        use_container_width=True
    ):

        st.session_state.score_weights = (
            INITIAL_SCORE_WEIGHTS.copy()
        )

        st.cache_data.clear()

        st.rerun()

normalized = normalize_weights(
    st.session_state.score_weights
)

st.sidebar.caption(
    "Effektive Gewichte:"
)

st.sidebar.write(
    f"KPAX: **{normalized['kpax']*100:.0f}%**"
)

st.sidebar.write(
    f"KPAX-FV: **{normalized['kpax_fv']*100:.0f}%**"
)

st.sidebar.write(
    f"Risk: **{normalized['risk']*100:.0f}%**"
)


# ============================================================
# MAIN
# ============================================================

st.title(
    f"📊 Aktien-Screener {VERSION}"
)

st.markdown(
    """
**Datenengine V21:** Fundamentaldaten werden primär aus
Yahoo-Financial-Statements gelesen. `Ticker.info` dient nur
noch als Ergänzung/Fallback.
"""
)

col1, col2, col3 = st.columns(3)

with col1:

    run_button = st.button(
        "🚀 Screener aktualisieren",
        type="primary",
        use_container_width=True
    )

with col2:

    clear_button = st.button(
        "🗑 Cache löschen",
        use_container_width=True
    )

with col3:

    st.metric(
        "Aktien",
        len(TICKERS)
    )


if clear_button:

    st.cache_data.clear()

    st.success(
        "Cache wurde gelöscht."
    )

    st.rerun()


# ============================================================
# RUN ANALYSIS
# ============================================================

if run_button or "screening_results" not in st.session_state:

    results = []

    progress = st.progress(
        0
    )

    status_text = st.empty()

    for i, ticker in enumerate(TICKERS):

        status_text.write(
            f"Analysiere **{ticker}** ..."
        )

        result = analyze_ticker(
            ticker
        )

        results.append(
            result
        )

        progress.progress(
            (i + 1)
            / len(TICKERS)
        )

        # Small delay avoids aggressive request bursts
        time.sleep(
            0.05
        )

    progress.empty()
    status_text.empty()

    st.session_state.screening_results = (
        results
    )


results = st.session_state.screening_results

df = pd.DataFrame(
    results
)


# ============================================================
# SORT
# ============================================================

if not df.empty:

    df["_sort"] = pd.to_numeric(
        df["Investment"],
        errors="coerce"
    )

    df = df.sort_values(
        "_sort",
        ascending=False,
        na_position="last"
    )

    df = df.drop(
        columns=["_sort"]
    )


# ============================================================
# MAIN TABLE
# ============================================================

st.subheader(
    "🏆 Ranking"
)

display_columns = [
    "Ticker",
    "Price",
    "Fair Value",
    "FV Upside",
    "FV Score",
    "Relative Valuation",
    "KPAX",
    "KPAX-FV",
    "Risk",
    "Technical",
    "Investment",
    "Recommendation",
    "Verify",
    "Gap",
    "Status",
    "Model Confidence"
]

display_df = df[
    display_columns
].copy()

display_df["Price"] = display_df[
    "Price"
].apply(format_price)

display_df["Fair Value"] = display_df[
    "Fair Value"
].apply(format_price)

display_df["FV Upside"] = display_df[
    "FV Upside"
].apply(format_pct)

for col in [
    "FV Score",
    "Relative Valuation",
    "KPAX",
    "KPAX-FV",
    "Risk",
    "Technical",
    "Investment",
    "Verify",
    "Model Confidence"
]:

    display_df[col] = display_df[
        col
    ].apply(format_score)

display_df["Gap"] = display_df[
    "Gap"
].apply(format_number)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DATA IMPORT DIAGNOSTICS
# ============================================================

st.subheader(
    "🔎 Datenimport-Diagnose"
)

st.caption(
    "Damit lässt sich direkt erkennen, ob ein Wert wirklich "
    "von Yahoo fehlt oder erst im Score-Modell verloren geht."
)

diagnostic_columns = [
    "Ticker",
    "Price",
    "Revenue",
    "EPS",
    "Forward EPS",
    "FCF",
    "Equity",
    "Debt",
    "Cash",
    "Growth",
    "Forward PE",
    "Trailing PE",
    "PEG",
    "P/S",
    "P/B",
    "Analyst Target",
    "FV Models",
    "FV Availability",
    "Relative Availability",
    "Data Quality",
    "History Days",
    "Status"
]

diag = df[
    diagnostic_columns
].copy()

for col in [
    "Price",
    "Revenue",
    "EPS",
    "Forward EPS",
    "FCF",
    "Equity",
    "Debt",
    "Cash",
    "Analyst Target"
]:

    diag[col] = diag[col].apply(
        lambda x: (
            format_number(x, 2)
            if safe_float(x) is not None
            else "—"
        )
    )

for col in [
    "Growth",
    "FV Availability",
    "Relative Availability",
    "Data Quality"
]:

    diag[col] = diag[col].apply(
        lambda x: (
            format_pct(x)
            if safe_float(x) is not None
            else "—"
        )
    )

for col in [
    "Forward PE",
    "Trailing PE",
    "PEG",
    "P/S",
    "P/B"
]:

    diag[col] = diag[col].apply(
        lambda x: (
            format_number(x, 2)
            if safe_float(x) is not None
            else "—"
        )
    )

st.dataframe(
    diag,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FAIR VALUE DETAIL
# ============================================================

st.subheader(
    "💰 Fair Value Detail"
)

fv_columns = [
    "Ticker",
    "Price",
    "Fair Value",
    "FV Upside",
    "FV Score",
    "FV Models",
    "FV Availability",
    "Analyst Target",
    "Forward EPS",
    "EPS",
    "FCF",
    "Growth"
]

fv_df = df[
    fv_columns
].copy()

fv_df["Price"] = fv_df[
    "Price"
].apply(format_price)

fv_df["Fair Value"] = fv_df[
    "Fair Value"
].apply(format_price)

fv_df["FV Upside"] = fv_df[
    "FV Upside"
].apply(format_pct)

fv_df["FV Score"] = fv_df[
    "FV Score"
].apply(format_score)

fv_df["FV Availability"] = fv_df[
    "FV Availability"
].apply(format_pct)

fv_df["Analyst Target"] = fv_df[
    "Analyst Target"
].apply(format_price)

fv_df["Forward EPS"] = fv_df[
    "Forward EPS"
].apply(format_number)

fv_df["EPS"] = fv_df[
    "EPS"
].apply(format_number)

fv_df["FCF"] = fv_df[
    "FCF"
].apply(format_number)

fv_df["Growth"] = fv_df[
    "Growth"
].apply(format_pct)

st.dataframe(
    fv_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# VERIFY
# ============================================================

st.subheader(
    "🧪 Verify"
)

verify_columns = [
    "Ticker",
    "Investment",
    "Verify",
    "Gap",
    "Status",
    "Model Confidence",
    "Data Quality"
]

verify_df = df[
    verify_columns
].copy()

verify_df["Investment"] = verify_df[
    "Investment"
].apply(format_score)

verify_df["Verify"] = verify_df[
    "Verify"
].apply(format_score)

verify_df["Gap"] = verify_df[
    "Gap"
].apply(format_number)

verify_df["Model Confidence"] = verify_df[
    "Model Confidence"
].apply(format_score)

verify_df["Data Quality"] = verify_df[
    "Data Quality"
].apply(format_pct)

st.dataframe(
    verify_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FORMULAS
# ============================================================

st.subheader(
    "🧮 Mathematik des Modells"
)

with st.expander(
    "Berechnungsformeln anzeigen"
):

    st.markdown(
        """
### 1. Quality Score

Nicht-Finanzunternehmen:

`Quality = 30% ROIC + 15% Gross Margin + 20% FCF Margin + 20% Earnings Growth + 15% Revenue Growth`

Finanzunternehmen:

`Quality = 40% ROE + 30% Net Margin + 20% Earnings Growth + 10% Revenue Growth`

---

### 2. Future Score

`Future = 35% Earnings Growth + 20% Revenue Growth + 20% FCF Margin + 25% Analyst Upside`

Nicht verfügbare Komponenten werden innerhalb dieses Submodells
nicht als 0 bewertet.

---

### 3. KPAX

`KPAX = 40% Quality + 60% Future`

---

### 4. Earnings Fair Value

`Fair P/E = 12 + 0.25 × Growth`

begrenzt auf:

`12 ≤ Fair P/E ≤ 24`

Danach:

`Earnings FV = EPS × Fair P/E`

Primär wird Forward EPS verwendet.
Falls dieses fehlt, wird Trailing EPS verwendet.

---

### 5. FCF Fair Value

`FCF/share = normalisierter FCF / Aktienanzahl`

`Target Yield = 7.0% − 0.08 × Growth`

Danach Begrenzung auf:

`4.5% ≤ Target Yield ≤ 9.0%`

Bei Beta > 1 wird ein zusätzlicher Risikoaufschlag vorgenommen.

---

### 6. Fair Value

Nicht-Finanzunternehmen:

`45% Earnings FV + 25% FCF FV + 30% Analyst FV`

Finanzunternehmen:

`35% Earnings FV + 15% FCF FV + 50% Analyst FV`

Nicht verfügbare Modelle werden nicht als 0 bewertet.

---

### 7. Fair Value Score

`FV Score = 55 + 0.60 × FV Upside`

begrenzt auf:

`5 ≤ FV Score ≤ 100`

---

### 8. Relative Valuation

Nicht-Finanzunternehmen:

`35% Forward PE + 10% Trailing PE + 30% PEG + 25% P/S`

Finanzunternehmen:

`35% Forward PE + 15% Trailing PE + 15% PEG + 35% P/B`

Die Einzelkennzahlen werden über eine Sigmoid-Funktion in einen
0–100 Score umgerechnet.

---

### 9. KPAX-FV

`KPAX-FV = 55% Fair Value Score + 45% Relative Valuation`

---

### 10. Investment Score

Standard:

`45% KPAX + 45% KPAX-FV + 10% Risk`

**Wichtig:** Fehlt der komplette KPAX-FV-Block, wird er nicht aus
der Gewichtung entfernt. Stattdessen wird für diesen Block neutral
`50` angesetzt.

Dadurch kann eine Aktie mit fehlender Bewertung nicht mehr
fälschlicherweise durch eine Renormalisierung zum Strong Buy werden.

---

### 11. Recommendation

- `≥ 85` Strong Buy
- `≥ 80` Buy
- `≥ 75` Accumulate
- `≥ 68` Hold
- `≥ 55` Reduce / Watch
- `< 55` Avoid

Technical wirkt zusätzlich als Timing-Filter.

Eine Aktie ohne vollständige Bewertungsdaten kann nicht als
Strong Buy ausgegeben werden.
"""
    )


# ============================================================
# SUMMARY
# ============================================================

st.subheader(
    "📌 Kurzdiagnose"
)

if not df.empty:

    total = len(df)

    with_fv = df[
        df["Fair Value"].apply(
            lambda x: safe_float(x) is not None
        )
    ].shape[0]

    with_relative = df[
        df["Relative Valuation"].apply(
            lambda x: safe_float(x) is not None
        )
    ].shape[0]

    with_kpax = df[
        df["KPAX"].apply(
            lambda x: safe_float(x) is not None
        )
    ].shape[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Aktien analysiert",
            total
        )

    with col2:
        st.metric(
            "Fair Value verfügbar",
            f"{with_fv}/{total}"
        )

    with col3:
        st.metric(
            "Relative Valuation",
            f"{with_relative}/{total}"
        )

    with col4:
        st.metric(
            "KPAX",
            f"{with_kpax}/{total}"
        )

    if with_fv < total * 0.7:

        st.warning(
            "⚠️ Bei einem großen Teil der Aktien fehlen weiterhin "
            "Fair-Value-Daten. Die Diagnose-Tabelle oben zeigt "
            "jetzt, ob EPS/FCF/Analyst Target bzw. andere "
            "Fundamentaldaten bereits beim Yahoo-Import fehlen."
        )

    else:

        st.success(
            "✅ Der Fundamentaldaten-Import liefert für den "
            "Großteil des Universums verwertbare Daten."
        )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    f"Aktien-Screener {VERSION} | "
    "Yahoo Finance / yfinance | "
    "Scores sind Modellwerte und keine Anlageberatung."
)
