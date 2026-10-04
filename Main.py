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
    page_title="Aktien-Screener V20.3",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# VERSION
# ============================================================

VERSION = "V20.3"


# ============================================================
# INITIAL SCORE WEIGHTS
# ============================================================

INITIAL_SCORE_WEIGHTS = {
    "kpax": 0.40,
    "kpax_fv": 0.50,
    "risk": 0.10,
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
    "ALV.DE",
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
        "AMZN", "BMW.DE", "NKE", "TSLA",
        "MCD", "ADS.DE"
    ],

    "Consumer Defensive": [
        "PEP", "KO"
    ],

    "Healthcare": [
        "NVO", "PFE"
    ],

    "Energy": [
        "SU.PA", "TTE.PA"
    ],

    "Utilities": [
        "VST"
    ],

    "Industrials": [
        "SMO", "ENR.DE", "SIE.DE"
    ],

    "Financial Services": [
        "MUV2.DE", "ALV.DE", "ING"
    ],
}


# ============================================================
# BASIC HELPERS
# ============================================================

MIN_SCORE = 5.0
MAX_SCORE = 100.0


def safe_float(x):
    try:
        if x is None:
            return None

        if isinstance(x, (pd.Series, pd.DataFrame)):
            if len(x) == 0:
                return None
            x = x.iloc[0]

        x = float(x)

        if not np.isfinite(x):
            return None

        return x

    except Exception:
        return None


def clip_score(x):
    x = safe_float(x)

    if x is None:
        return None

    return float(np.clip(x, MIN_SCORE, MAX_SCORE))


def normalize_weights(values):
    valid = {
        k: safe_float(v)
        for k, v in values.items()
        if safe_float(v) is not None and safe_float(v) > 0
    }

    total = sum(valid.values())

    if total <= 0:
        return {}

    return {
        k: v / total
        for k, v in valid.items()
    }


def weighted_score(values, weights):
    """
    Weighted average of available values.
    Missing components are removed and weights renormalized.
    """

    available = {}

    for key, value in values.items():
        value = safe_float(value)

        if value is not None:
            available[key] = value

    if not available:
        return None

    usable_weights = {
        key: weights.get(key, 0)
        for key in available
        if weights.get(key, 0) > 0
    }

    usable_weights = normalize_weights(usable_weights)

    if not usable_weights:
        return None

    score = sum(
        available[key] * usable_weights[key]
        for key in usable_weights
    )

    return clip_score(score)


def percentage_score(
    value,
    low,
    high,
    inverse=False
):
    """
    Linear score from 5 to 100.
    """

    value = safe_float(value)

    if value is None:
        return None

    if high == low:
        return None

    if inverse:
        score = 100 - (
            (value - low) / (high - low) * 95
        )
    else:
        score = (
            (value - low) / (high - low) * 95
        ) + 5

    return clip_score(score)


def sigmoid_score(
    value,
    midpoint,
    scale,
    inverse=False
):
    """
    Smooth scoring function.

    Avoids the very hard valuation cliffs from V20.2.
    """

    value = safe_float(value)

    if value is None:
        return None

    if scale <= 0:
        return None

    z = (value - midpoint) / scale

    if inverse:
        score = 5 + 95 / (1 + math.exp(z))
    else:
        score = 5 + 95 / (1 + math.exp(-z))

    return clip_score(score)


def safe_div(a, b):
    a = safe_float(a)
    b = safe_float(b)

    if a is None or b is None:
        return None

    if abs(b) < 1e-12:
        return None

    return a / b


def get_first_number(*values):
    for value in values:
        value = safe_float(value)

        if value is not None:
            return value

    return None


# ============================================================
# YFINANCE DATA
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def load_ticker_data(ticker):

    result = {
        "ticker": ticker,
        "valid": False,
        "info": {},
        "fast_info": {},
        "history": pd.DataFrame(),
        "financials": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "quarterly_financials": pd.DataFrame(),
        "quarterly_balance": pd.DataFrame(),
        "quarterly_cashflow": pd.DataFrame(),
        "analyst_targets": {},
        "growth_estimates": pd.DataFrame(),
    }

    try:
        stock = yf.Ticker(ticker)

        # ----------------------------
        # INFO
        # ----------------------------

        try:
            result["info"] = stock.info or {}
        except Exception:
            result["info"] = {}

        # ----------------------------
        # FAST INFO
        # ----------------------------

        try:
            result["fast_info"] = dict(stock.fast_info)
        except Exception:
            result["fast_info"] = {}

        # ----------------------------
        # HISTORY
        # ----------------------------

        try:
            result["history"] = stock.history(
                period="3y",
                interval="1d",
                auto_adjust=True
            )
        except Exception:
            result["history"] = pd.DataFrame()

        # ----------------------------
        # FINANCIALS
        # ----------------------------

        try:
            result["financials"] = stock.financials
        except Exception:
            pass

        # ----------------------------
        # BALANCE SHEET
        # ----------------------------

        try:
            result["balance"] = stock.balance_sheet
        except Exception:
            pass

        # ----------------------------
        # CASH FLOW
        # ----------------------------

        try:
            result["cashflow"] = stock.cashflow
        except Exception:
            pass

        # ----------------------------
        # QUARTERLY
        # ----------------------------

        try:
            result["quarterly_financials"] = (
                stock.quarterly_financials
            )
        except Exception:
            pass

        try:
            result["quarterly_balance"] = (
                stock.quarterly_balance_sheet
            )
        except Exception:
            pass

        try:
            result["quarterly_cashflow"] = (
                stock.quarterly_cashflow
            )
        except Exception:
            pass

        # ----------------------------
        # ANALYST TARGETS
        # ----------------------------

        try:
            result["analyst_targets"] = (
                stock.analyst_price_targets or {}
            )
        except Exception:
            result["analyst_targets"] = {}

        # ----------------------------
        # GROWTH ESTIMATES
        # ----------------------------

        try:
            result["growth_estimates"] = (
                stock.growth_estimates
            )
        except Exception:
            result["growth_estimates"] = pd.DataFrame()

        result["valid"] = (
            len(result["info"]) > 0
            or len(result["history"]) > 0
        )

    except Exception:
        pass

    return result


# ============================================================
# DATA EXTRACTION
# ============================================================

def get_price(data):

    info = data["info"]
    fast = data["fast_info"]

    return get_first_number(
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
        fast.get("last_price"),
        fast.get("regularMarketPrice"),
    )


def get_market_cap(data):

    info = data["info"]

    return get_first_number(
        info.get("marketCap")
    )


def get_sector(data):

    info = data["info"]

    sector = info.get("sector")

    if sector:
        return sector

    ticker = data["ticker"]

    for sector_name, tickers in FALLBACK_SECTORS.items():
        if ticker in tickers:
            return sector_name

    return "Other"


def is_financial_sector(sector):

    return sector in [
        "Financial Services",
        "Banks",
        "Insurance",
        "Financials",
    ]


def get_info_value(data, key):

    return safe_float(
        data["info"].get(key)
    )


# ============================================================
# FINANCIAL DATA HELPERS
# ============================================================

def get_statement_value(df, names, column=None):

    if df is None or df.empty:
        return None

    if isinstance(names, str):
        names = [names]

    for name in names:

        if name not in df.index:
            continue

        try:

            if column is not None:

                if column in df.columns:
                    return safe_float(
                        df.loc[name, column]
                    )

            else:

                series = df.loc[name]

                if isinstance(series, pd.Series):

                    for value in series.values:
                        value = safe_float(value)

                        if value is not None:
                            return value

        except Exception:
            continue

    return None


def get_latest_statement_value(df, names):

    if df is None or df.empty:
        return None

    if isinstance(names, str):
        names = [names]

    for name in names:

        if name not in df.index:
            continue

        try:

            series = df.loc[name]

            for value in series.values:

                value = safe_float(value)

                if value is not None:
                    return value

        except Exception:
            continue

    return None


def get_series_from_statement(df, names):

    if df is None or df.empty:
        return None

    if isinstance(names, str):
        names = [names]

    for name in names:

        if name not in df.index:
            continue

        try:
            series = pd.to_numeric(
                df.loc[name],
                errors="coerce"
            ).dropna()

            if len(series) > 0:
                return series

        except Exception:
            continue

    return None


# ============================================================
# CORE FUNDAMENTAL METRICS
# ============================================================

def get_eps(data):

    info = data["info"]

    return get_first_number(
        info.get("forwardEps"),
        info.get("trailingEps"),
    )


def get_forward_eps(data):

    return get_info_value(
        data,
        "forwardEps"
    )


def get_trailing_eps(data):

    return get_info_value(
        data,
        "trailingEps"
    )


def get_forward_pe(data):

    return get_info_value(
        data,
        "forwardPE"
    )


def get_trailing_pe(data):

    return get_info_value(
        data,
        "trailingPE"
    )


def get_peg(data):

    return get_info_value(
        data,
        "pegRatio"
    )


def get_price_sales(data):

    return get_info_value(
        data,
        "priceToSalesTrailing12Months"
    )


def get_price_book(data):

    return get_info_value(
        data,
        "priceToBook"
    )


def get_revenue(data):

    return get_first_number(
        get_latest_statement_value(
            data["quarterly_financials"],
            [
                "Total Revenue",
                "Operating Revenue",
            ]
        ),
        get_latest_statement_value(
            data["financials"],
            [
                "Total Revenue",
                "Operating Revenue",
            ]
        )
    )


def get_net_income(data):

    return get_first_number(
        get_latest_statement_value(
            data["quarterly_financials"],
            [
                "Net Income",
                "Net Income Common Stockholders",
            ]
        ),
        get_latest_statement_value(
            data["financials"],
            [
                "Net Income",
                "Net Income Common Stockholders",
            ]
        )
    )


def get_total_assets(data):

    return get_latest_statement_value(
        data["balance"],
        ["Total Assets"]
    )


def get_total_equity(data):

    return get_latest_statement_value(
        data["balance"],
        [
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
        ]
    )


def get_total_debt(data):

    return get_first_number(
        get_latest_statement_value(
            data["balance"],
            [
                "Total Debt",
                "Long Term Debt And Capital Lease Obligation",
                "Long Term Debt",
            ]
        ),
        get_info_value(
            data,
            "totalDebt"
        )
    )


def get_cash(data):

    return get_first_number(
        get_latest_statement_value(
            data["balance"],
            [
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents",
                "Cash Financial",
            ]
        ),
        get_info_value(
            data,
            "totalCash"
        )
    )


def get_current_assets(data):

    return get_latest_statement_value(
        data["balance"],
        [
            "Current Assets"
        ]
    )


def get_current_liabilities(data):

    return get_latest_statement_value(
        data["balance"],
        [
            "Current Liabilities"
        ]
    )


def get_gross_profit(data):

    return get_latest_statement_value(
        data["financials"],
        [
            "Gross Profit"
        ]
    )


def get_operating_income(data):

    return get_latest_statement_value(
        data["financials"],
        [
            "Operating Income"
        ]
    )


def get_tax_expense(data):

    return get_latest_statement_value(
        data["financials"],
        [
            "Tax Provision",
            "Tax Effect Of Unusual Items",
        ]
    )


def get_pretax_income(data):

    return get_latest_statement_value(
        data["financials"],
        [
            "Pretax Income"
        ]
    )


def get_fcf(data):

    return get_first_number(
        get_latest_statement_value(
            data["cashflow"],
            [
                "Free Cash Flow"
            ]
        ),
        get_info_value(
            data,
            "freeCashflow"
        )
    )


# ============================================================
# GROWTH ESTIMATES
# ============================================================

def get_growth_estimate(data, row_candidates, column_candidates):

    df = data["growth_estimates"]

    if df is None or df.empty:
        return None

    try:

        for row in row_candidates:

            if row not in df.index:
                continue

            for col in column_candidates:

                if col not in df.columns:
                    continue

                value = safe_float(
                    df.loc[row, col]
                )

                if value is not None:
                    return value * 100

    except Exception:
        pass

    return None


def get_forward_earnings_growth(data):

    info = data["info"]

    direct = get_first_number(
        info.get("earningsQuarterlyGrowth"),
        info.get("earningsGrowth"),
    )

    if direct is not None:
        return direct * 100

    return get_growth_estimate(
        data,
        ["stock"],
        ["+1y", "0y"]
    )


def get_long_term_growth(data):

    return get_growth_estimate(
        data,
        ["stock"],
        ["+5y"]
    )


def get_historical_earnings_growth(data):

    info = data["info"]

    value = get_first_number(
        info.get("earningsGrowth"),
        info.get("earningsQuarterlyGrowth"),
    )

    if value is not None:
        return value * 100

    series = get_series_from_statement(
        data["financials"],
        [
            "Net Income",
            "Net Income Common Stockholders",
        ]
    )

    if series is not None and len(series) >= 2:

        latest = safe_float(series.iloc[0])
        previous = safe_float(series.iloc[1])

        if (
            latest is not None
            and previous is not None
            and previous != 0
        ):
            return (
                (latest / previous) - 1
            ) * 100

    return None


def get_revenue_growth(data):

    info = data["info"]

    value = get_first_number(
        info.get("revenueGrowth")
    )

    if value is not None:
        return value * 100

    series = get_series_from_statement(
        data["financials"],
        [
            "Total Revenue",
            "Operating Revenue",
        ]
    )

    if series is not None and len(series) >= 2:

        latest = safe_float(series.iloc[0])
        previous = safe_float(series.iloc[1])

        if (
            latest is not None
            and previous is not None
            and previous != 0
        ):
            return (
                (latest / previous) - 1
            ) * 100

    return None


def get_fcf_growth(data):

    series = get_series_from_statement(
        data["cashflow"],
        [
            "Free Cash Flow"
        ]
    )

    if series is None or len(series) < 2:
        return None

    latest = safe_float(series.iloc[0])
    previous = safe_float(series.iloc[1])

    if (
        latest is None
        or previous is None
        or previous == 0
        or latest * previous <= 0
    ):
        return None

    return (
        latest / previous - 1
    ) * 100


# ============================================================
# ANALYST TARGET
# ============================================================

def get_analyst_target(data):

    targets = data["analyst_targets"]

    if not isinstance(targets, dict):
        return None

    return get_first_number(
        targets.get("current"),
        targets.get("mean"),
        targets.get("targetMeanPrice"),
    )


def get_analyst_upside(data):

    price = get_price(data)
    target = get_analyst_target(data)

    if (
        price is None
        or target is None
        or price <= 0
    ):
        return None

    return (
        target / price - 1
    ) * 100


# ============================================================
# PROFITABILITY
# ============================================================

def calculate_roic(data):

    """
    Attempts real ROIC:

    NOPAT / Invested Capital

    Invested Capital ≈ Equity + Debt - Cash

    If insufficient data exists, returns ROA proxy.
    """

    operating_income = get_operating_income(data)
    pretax_income = get_pretax_income(data)
    tax_expense = get_tax_expense(data)

    equity = get_total_equity(data)
    debt = get_total_debt(data)
    cash = get_cash(data)

    if (
        operating_income is not None
        and equity is not None
        and debt is not None
        and cash is not None
    ):

        tax_rate = 0.21

        if (
            pretax_income is not None
            and tax_expense is not None
            and pretax_income > 0
        ):

            calculated_tax = (
                tax_expense / pretax_income
            )

            if 0 <= calculated_tax <= 0.5:
                tax_rate = calculated_tax

        nopat = operating_income * (
            1 - tax_rate
        )

        invested_capital = (
            equity + debt - cash
        )

        if invested_capital > 0:

            return {
                "value": (
                    nopat /
                    invested_capital
                ) * 100,
                "method": "ROIC"
            }

    roa = get_info_value(
        data,
        "returnOnAssets"
    )

    if roa is not None:

        return {
            "value": roa * 100,
            "method": "ROA-Proxy"
        }

    return {
        "value": None,
        "method": None
    }


def calculate_quality(data):

    sector = get_sector(data)
    financial = is_financial_sector(sector)

    info = data["info"]

    roe = get_info_value(
        data,
        "returnOnEquity"
    )

    net_margin = get_info_value(
        data,
        "profitMargins"
    )

    gross_margin = get_info_value(
        data,
        "grossMargins"
    )

    fcf_margin = get_info_value(
        data,
        "freeCashflow"
    )

    revenue = get_revenue(data)
    fcf = get_fcf(data)

    if (
        fcf_margin is None
        and revenue is not None
        and fcf is not None
        and revenue != 0
    ):
        fcf_margin = fcf / revenue

    growth_e = get_historical_earnings_growth(data)
    growth_r = get_revenue_growth(data)

    if financial:

        roe_score = (
            sigmoid_score(
                roe * 100,
                midpoint=15,
                scale=10
            )
            if roe is not None
            else None
        )

        margin_score = (
            sigmoid_score(
                net_margin * 100,
                midpoint=15,
                scale=10
            )
            if net_margin is not None
            else None
        )

        growth_score = (
            sigmoid_score(
                growth_e,
                midpoint=8,
                scale=12
            )
            if growth_e is not None
            else None
        )

        revenue_score = (
            sigmoid_score(
                growth_r,
                midpoint=6,
                scale=10
            )
            if growth_r is not None
            else None
        )

        return weighted_score(
            {
                "roe": roe_score,
                "margin": margin_score,
                "earnings_growth": growth_score,
                "revenue_growth": revenue_score,
            },
            {
                "roe": 0.40,
                "margin": 0.30,
                "earnings_growth": 0.20,
                "revenue_growth": 0.10,
            }
        )

    roic_result = calculate_roic(data)

    roic = roic_result["value"]

    roic_score = (
        sigmoid_score(
            roic,
            midpoint=12,
            scale=10
        )
        if roic is not None
        else None
    )

    gross_score = (
        sigmoid_score(
            gross_margin * 100,
            midpoint=40,
            scale=15
        )
        if gross_margin is not None
        else None
    )

    fcf_score = (
        sigmoid_score(
            fcf_margin * 100,
            midpoint=10,
            scale=8
        )
        if fcf_margin is not None
        else None
    )

    earnings_score = (
        sigmoid_score(
            growth_e,
            midpoint=8,
            scale=12
        )
        if growth_e is not None
        else None
    )

    revenue_score = (
        sigmoid_score(
            growth_r,
            midpoint=6,
            scale=10
        )
        if growth_r is not None
        else None
    )

    return weighted_score(
        {
            "roic": roic_score,
            "gross_margin": gross_score,
            "fcf_margin": fcf_score,
            "earnings_growth": earnings_score,
            "revenue_growth": revenue_score,
        },
        {
            "roic": 0.30,
            "gross_margin": 0.15,
            "fcf_margin": 0.20,
            "earnings_growth": 0.20,
            "revenue_growth": 0.15,
        }
    )


# ============================================================
# FUTURE SCORE V20.3
# ============================================================

def calculate_future(data):

    """
    Important V20.3 change:

    No longer:
        long_term_growth = earnings_growth
        acceleration = revenue_growth

    Instead we use independent signals.
    """

    historical_earnings = (
        get_historical_earnings_growth(data)
    )

    revenue_growth = get_revenue_growth(data)

    forward_earnings = (
        get_forward_earnings_growth(data)
    )

    long_term_growth = (
        get_long_term_growth(data)
    )

    fcf_growth = get_fcf_growth(data)

    revenue = get_revenue(data)
    fcf = get_fcf(data)

    fcf_margin = (
        fcf / revenue * 100
        if (
            fcf is not None
            and revenue is not None
            and revenue != 0
        )
        else None
    )

    # --------------------------------------------
    # Growth acceleration
    # --------------------------------------------

    acceleration = None

    if (
        forward_earnings is not None
        and historical_earnings is not None
    ):

        acceleration = (
            forward_earnings
            - historical_earnings
        )

    earnings_score = (
        sigmoid_score(
            historical_earnings,
            midpoint=8,
            scale=12
        )
        if historical_earnings is not None
        else None
    )

    revenue_score = (
        sigmoid_score(
            revenue_growth,
            midpoint=6,
            scale=10
        )
        if revenue_growth is not None
        else None
    )

    forward_score = (
        sigmoid_score(
            forward_earnings,
            midpoint=10,
            scale=12
        )
        if forward_earnings is not None
        else None
    )

    long_term_score = (
        sigmoid_score(
            long_term_growth,
            midpoint=10,
            scale=10
        )
        if long_term_growth is not None
        else None
    )

    fcf_score = (
        sigmoid_score(
            fcf_margin,
            midpoint=10,
            scale=8
        )
        if fcf_margin is not None
        else None
    )

    acceleration_score = None

    if acceleration is not None:

        # +15pp acceleration = very strong
        # 0pp = neutral
        # -15pp = weak

        acceleration_score = clip_score(
            55 + acceleration * 3
        )

    return weighted_score(
        {
            "earnings_growth": earnings_score,
            "revenue_growth": revenue_score,
            "forward_earnings": forward_score,
            "long_term_growth": long_term_score,
            "fcf_margin": fcf_score,
            "acceleration": acceleration_score,
        },
        {
            "earnings_growth": 0.25,
            "revenue_growth": 0.20,
            "forward_earnings": 0.20,
            "long_term_growth": 0.15,
            "fcf_margin": 0.10,
            "acceleration": 0.10,
        }
    )


# ============================================================
# FAIR VALUE ENGINE V20.3
# ============================================================

def calculate_fair_value(data):

    price = get_price(data)

    if price is None or price <= 0:
        return {
            "score": None,
            "fair_value": None,
            "earnings_fv": None,
            "fcf_fv": None,
            "analyst_fv": None,
            "confidence": 0,
            "method": "Keine Kursdaten"
        }

    sector = get_sector(data)
    financial = is_financial_sector(sector)

    forward_eps = get_forward_eps(data)
    trailing_eps = get_trailing_eps(data)

    growth = get_forward_earnings_growth(data)

    if growth is None:
        growth = get_historical_earnings_growth(data)

    analyst_target = get_analyst_target(data)

    # ========================================================
    # MODEL 1 — EARNINGS FAIR VALUE
    # ========================================================

    earnings_fv = None

    if forward_eps is not None and forward_eps > 0:

        growth_for_pe = (
            np.clip(
                growth if growth is not None else 8,
                -5,
                40
            )
        )

        # Less aggressive than V20.2:
        #
        # V20.2:
        # P/E = 10 + 0.40 * growth
        #
        # V20.3:
        # P/E = 12 + 0.25 * growth
        #
        # This avoids assigning 30x+ P/E
        # merely because growth is high.

        fair_pe = (
            12
            + 0.25 * growth_for_pe
        )

        fair_pe = float(
            np.clip(
                fair_pe,
                12,
                24
            )
        )

        earnings_fv = (
            forward_eps * fair_pe
        )

    # ========================================================
    # MODEL 2 — FCF FAIR VALUE
    # ========================================================

    fcf_fv = None

    fcf = get_fcf(data)
    shares = get_info_value(
        data,
        "sharesOutstanding"
    )

    if (
        fcf is not None
        and shares is not None
        and shares > 0
        and fcf > 0
    ):

        fcf_per_share = (
            fcf / shares
        )

        growth_for_yield = (
            np.clip(
                growth if growth is not None else 8,
                0,
                35
            )
        )

        beta = get_info_value(
            data,
            "beta"
        )

        # Base target yield.
        #
        # V20.2 could become excessively optimistic
        # for high-growth companies.
        #
        # V20.3 deliberately keeps the range tighter.

        target_yield = (
            0.070
            - 0.0008 * growth_for_yield
        )

        if beta is not None and beta > 1:
            target_yield += (
                min(beta - 1, 1.5)
                * 0.005
            )

        target_yield = float(
            np.clip(
                target_yield,
                0.045,
                0.09
            )
        )

        fcf_fv = (
            fcf_per_share
            / target_yield
        )

    # ========================================================
    # MODEL 3 — ANALYST FAIR VALUE
    # ========================================================

    analyst_fv = analyst_target

    # ========================================================
    # PLAUSIBILITY FILTER
    # ========================================================

    raw_models = {
        "earnings": earnings_fv,
        "fcf": fcf_fv,
        "analyst": analyst_fv,
    }

    available = {
        key: value
        for key, value in raw_models.items()
        if value is not None
        and value > 0
        and np.isfinite(value)
    }

    if not available:

        return {
            "score": None,
            "fair_value": None,
            "earnings_fv": None,
            "fcf_fv": None,
            "analyst_fv": None,
            "confidence": 0,
            "method": "Keine FV-Modelle"
        }

    # --------------------------------------------------------
    # First calculate median.
    # This makes the filter robust against one absurd model.
    # --------------------------------------------------------

    median_model = float(
        np.median(
            list(available.values())
        )
    )

    # --------------------------------------------------------
    # Limit individual model values.
    #
    # No single model may be more than:
    # 0.40x or 2.50x the model median.
    # --------------------------------------------------------

    bounded = {}

    for key, value in available.items():

        lower = median_model * 0.40
        upper = median_model * 2.50

        bounded[key] = float(
            np.clip(
                value,
                lower,
                upper
            )
        )

    # ========================================================
    # WEIGHTS
    # ========================================================

    if financial:

        weights = {
            "earnings": 0.35,
            "fcf": 0.15,
            "analyst": 0.50,
        }

    else:

        weights = {
            "earnings": 0.45,
            "fcf": 0.25,
            "analyst": 0.30,
        }

    usable_weights = normalize_weights(
        {
            key: weights[key]
            for key in bounded
            if key in weights
        }
    )

    fundamental_fv = sum(
        bounded[key] * usable_weights[key]
        for key in usable_weights
    )

    # ========================================================
    # FINAL FAIR VALUE
    # ========================================================

    fair_value = fundamental_fv

    # Additional global sanity limit:
    # fair value should not explode to hundreds of %
    # above/below the market price just because one
    # data source is distorted.

    fair_value = float(
        np.clip(
            fair_value,
            price * 0.35,
            price * 3.0
        )
    )

    # ========================================================
    # FV SCORE
    # ========================================================

    upside = (
        fair_value / price - 1
    ) * 100

    # Smooth scoring.
    #
    # +50% upside -> ~90
    # +25% -> ~75
    # 0% -> ~50
    # -25% -> ~30
    #
    # Much less cliff-like than V20.2.

    fv_score = clip_score(
        55 + 0.72 * upside
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    model_count = len(bounded)

    confidence = (
        model_count / 3
    ) * 70

    if model_count >= 2:
        confidence += 15

    if analyst_fv is not None:
        confidence += 10

    if earnings_fv is not None:
        confidence += 5

    confidence = float(
        np.clip(
            confidence,
            0,
            100
        )
    )

    method = (
        f"{model_count}/3 Modelle"
    )

    return {
        "score": fv_score,
        "fair_value": fair_value,
        "earnings_fv": earnings_fv,
        "fcf_fv": fcf_fv,
        "analyst_fv": analyst_fv,
        "confidence": confidence,
        "method": method,
    }


# ============================================================
# RELATIVE VALUATION V20.3
# ============================================================

def calculate_relative_valuation(data):

    sector = get_sector(data)
    financial = is_financial_sector(sector)

    forward_pe = get_forward_pe(data)
    trailing_pe = get_trailing_pe(data)
    peg = get_peg(data)

    pb = get_price_book(data)
    ps = get_price_sales(data)

    # --------------------------------------------------------
    # Forward P/E
    # --------------------------------------------------------

    forward_score = (
        sigmoid_score(
            forward_pe,
            midpoint=22,
            scale=10,
            inverse=True
        )
        if (
            forward_pe is not None
            and forward_pe > 0
        )
        else None
    )

    # --------------------------------------------------------
    # Trailing P/E
    # --------------------------------------------------------

    trailing_score = (
        sigmoid_score(
            trailing_pe,
            midpoint=25,
            scale=12,
            inverse=True
        )
        if (
            trailing_pe is not None
            and trailing_pe > 0
        )
        else None
    )

    # --------------------------------------------------------
    # PEG
    # --------------------------------------------------------

    peg_score = (
        sigmoid_score(
            peg,
            midpoint=1.8,
            scale=0.9,
            inverse=True
        )
        if (
            peg is not None
            and peg > 0
        )
        else None
    )

    if financial:

        pb_score = (
            sigmoid_score(
                pb,
                midpoint=1.8,
                scale=0.8,
                inverse=True
            )
            if (
                pb is not None
                and pb > 0
            )
            else None
        )

        return weighted_score(
            {
                "forward_pe": forward_score,
                "trailing_pe": trailing_score,
                "peg": peg_score,
                "pb": pb_score,
            },
            {
                "forward_pe": 0.35,
                "trailing_pe": 0.15,
                "peg": 0.15,
                "pb": 0.35,
            }
        )

    ps_score = (
        sigmoid_score(
            ps,
            midpoint=5,
            scale=3,
            inverse=True
        )
        if (
            ps is not None
            and ps > 0
        )
        else None
    )

    return weighted_score(
        {
            "forward_pe": forward_score,
            "trailing_pe": trailing_score,
            "peg": peg_score,
            "ps": ps_score,
        },
        {
            "forward_pe": 0.35,
            "trailing_pe": 0.10,
            "peg": 0.30,
            "ps": 0.25,
        }
    )


# ============================================================
# KPAX
# ============================================================

def calculate_kpax(quality, future):

    return weighted_score(
        {
            "quality": quality,
            "future": future,
        },
        {
            "quality": 0.40,
            "future": 0.60,
        }
    )


def calculate_kpax_fv(
    fair_value_score,
    relative_score
):

    return weighted_score(
        {
            "fair_value": fair_value_score,
            "relative": relative_score,
        },
        {
            "fair_value": 0.60,
            "relative": 0.40,
        }
    )


# ============================================================
# RISK
# ============================================================

def calculate_risk(data):

    info = data["info"]

    debt = get_total_debt(data)
    cash = get_cash(data)
    equity = get_total_equity(data)

    # --------------------------------------------------------
    # Financial risk
    # --------------------------------------------------------

    debt_equity = None

    if (
        debt is not None
        and equity is not None
        and equity != 0
    ):
        debt_equity = debt / equity

    cash_debt = None

    if (
        cash is not None
        and debt is not None
        and debt > 0
    ):
        cash_debt = cash / debt

    current_assets = get_current_assets(data)
    current_liabilities = get_current_liabilities(data)

    current_ratio = None

    if (
        current_assets is not None
        and current_liabilities is not None
        and current_liabilities > 0
    ):
        current_ratio = (
            current_assets /
            current_liabilities
        )

    debt_score = weighted_score(
        {
            "debt_equity":
                sigmoid_score(
                    debt_equity,
                    midpoint=0.8,
                    scale=0.7,
                    inverse=True
                )
                if debt_equity is not None
                else None,

            "cash_debt":
                sigmoid_score(
                    cash_debt,
                    midpoint=0.5,
                    scale=0.5,
                    inverse=False
                )
                if cash_debt is not None
                else None,

            "current_ratio":
                sigmoid_score(
                    current_ratio,
                    midpoint=1.5,
                    scale=0.8,
                    inverse=False
                )
                if current_ratio is not None
                else None,
        },
        {
            "debt_equity": 0.50,
            "cash_debt": 0.30,
            "current_ratio": 0.20,
        }
    )

    # --------------------------------------------------------
    # Market risk
    # --------------------------------------------------------

    history = data["history"]

    volatility_score = None
    drawdown_score = None

    if (
        history is not None
        and not history.empty
        and "Close" in history.columns
    ):

        close = history["Close"].dropna()

        if len(close) >= 100:

            returns = close.pct_change().dropna()

            annual_vol = (
                returns.std() *
                np.sqrt(252)
            ) * 100

            volatility_score = sigmoid_score(
                annual_vol,
                midpoint=30,
                scale=12,
                inverse=True
            )

            rolling_max = close.cummax()

            drawdown = (
                close / rolling_max - 1
            )

            max_drawdown = (
                abs(drawdown.min())
                * 100
            )

            drawdown_score = sigmoid_score(
                max_drawdown,
                midpoint=35,
                scale=15,
                inverse=True
            )

    market_score = weighted_score(
        {
            "volatility": volatility_score,
            "drawdown": drawdown_score,
        },
        {
            "volatility": 0.55,
            "drawdown": 0.45,
        }
    )

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    beta = get_info_value(
        data,
        "beta"
    )

    beta_score = (
        sigmoid_score(
            beta,
            midpoint=1.0,
            scale=0.5,
            inverse=True
        )
        if beta is not None
        else None
    )

    stability = beta_score

    return weighted_score(
        {
            "financial": debt_score,
            "market": market_score,
            "stability": stability,
        },
        {
            "financial": 0.45,
            "market": 0.35,
            "stability": 0.20,
        }
    )


# ============================================================
# TECHNICAL SCORE
# ============================================================

def calculate_technical(data):

    history = data["history"]

    if (
        history is None
        or history.empty
        or "Close" not in history.columns
    ):
        return None, {}

    close = history["Close"].dropna()

    if len(close) < 200:
        return None, {}

    current = safe_float(close.iloc[-1])

    sma50 = safe_float(
        close.rolling(50).mean().iloc[-1]
    )

    sma200 = safe_float(
        close.rolling(200).mean().iloc[-1]
    )

    perf6m = None

    if len(close) >= 126:

        old = safe_float(
            close.iloc[-126]
        )

        if old and old > 0:
            perf6m = (
                current / old - 1
            ) * 100

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    rsi = (
        100 -
        100 / (1 + rs)
    )

    current_rsi = safe_float(
        rsi.iloc[-1]
    )

    # RSI score:
    # 50 neutral
    # ~30 oversold
    # ~70 overbought

    rsi_score = None

    if current_rsi is not None:

        rsi_score = clip_score(
            100 - abs(current_rsi - 50) * 1.5
        )

        if current_rsi < 50:
            rsi_score = min(
                100,
                rsi_score + 5
            )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    histogram = macd - signal

    macd_value = safe_float(
        histogram.iloc[-1]
    )

    macd_score = None

    if macd_value is not None:

        recent_scale = (
            close.pct_change()
            .rolling(60)
            .std()
            .iloc[-1]
        )

        recent_scale = safe_float(
            recent_scale
        )

        if recent_scale is None:
            recent_scale = 0.02

        normalized = (
            macd_value /
            max(current * recent_scale, 1e-9)
        )

        macd_score = clip_score(
            55 + normalized * 20
        )

    # --------------------------------------------------------
    # SMA scores
    # --------------------------------------------------------

    sma200_score = None

    if (
        current is not None
        and sma200 is not None
        and sma200 > 0
    ):

        diff = (
            current / sma200 - 1
        ) * 100

        sma200_score = clip_score(
            55 + diff * 2
        )

    sma50_score = None

    if (
        current is not None
        and sma50 is not None
        and sma50 > 0
    ):

        diff = (
            current / sma50 - 1
        ) * 100

        sma50_score = clip_score(
            55 + diff * 2
        )

    performance_score = (
        sigmoid_score(
            perf6m,
            midpoint=10,
            scale=20
        )
        if perf6m is not None
        else None
    )

    score = weighted_score(
        {
            "sma200": sma200_score,
            "performance": performance_score,
            "rsi": rsi_score,
            "macd": macd_score,
            "sma50": sma50_score,
        },
        {
            "sma200": 0.25,
            "performance": 0.25,
            "rsi": 0.20,
            "macd": 0.20,
            "sma50": 0.10,
        }
    )

    details = {
        "SMA200": sma200_score,
        "6M": performance_score,
        "RSI": rsi_score,
        "MACD": macd_score,
        "SMA50": sma50_score,
        "RSI_value": current_rsi,
        "SMA50_value": sma50,
        "SMA200_value": sma200,
        "6M_value": perf6m,
    }

    return score, details


# ============================================================
# VERIFY ENGINE
# ============================================================

def calculate_verify(data):

    sector = get_sector(data)
    financial = is_financial_sector(sector)

    growth_e = get_historical_earnings_growth(data)
    growth_r = get_revenue_growth(data)

    growth_score = weighted_score(
        {
            "earnings": (
                sigmoid_score(
                    growth_e,
                    midpoint=8,
                    scale=12
                )
                if growth_e is not None
                else None
            ),
            "revenue": (
                sigmoid_score(
                    growth_r,
                    midpoint=6,
                    scale=10
                )
                if growth_r is not None
                else None
            ),
        },
        {
            "earnings": 0.65,
            "revenue": 0.35,
        }
    )

    # --------------------------------------------------------
    # Profitability
    # --------------------------------------------------------

    roe = get_info_value(
        data,
        "returnOnEquity"
    )

    net_margin = get_info_value(
        data,
        "profitMargins"
    )

    roic_result = calculate_roic(data)

    roic = roic_result["value"]

    revenue = get_revenue(data)
    fcf = get_fcf(data)

    fcf_margin = (
        fcf / revenue * 100
        if (
            fcf is not None
            and revenue is not None
            and revenue != 0
        )
        else None
    )

    gross_margin = get_info_value(
        data,
        "grossMargins"
    )

    if gross_margin is not None:
        gross_margin *= 100

    if financial:

        profitability = weighted_score(
            {
                "roe": (
                    sigmoid_score(
                        roe * 100,
                        midpoint=15,
                        scale=10
                    )
                    if roe is not None
                    else None
                ),
                "margin": (
                    sigmoid_score(
                        net_margin * 100,
                        midpoint=15,
                        scale=10
                    )
                    if net_margin is not None
                    else None
                ),
            },
            {
                "roe": 0.60,
                "margin": 0.40,
            }
        )

    else:

        profitability = weighted_score(
            {
                "roic": (
                    sigmoid_score(
                        roic,
                        midpoint=12,
                        scale=10
                    )
                    if roic is not None
                    else None
                ),
                "fcf_margin": (
                    sigmoid_score(
                        fcf_margin,
                        midpoint=10,
                        scale=8
                    )
                    if fcf_margin is not None
                    else None
                ),
                "gross_margin": (
                    sigmoid_score(
                        gross_margin,
                        midpoint=40,
                        scale=15
                    )
                    if gross_margin is not None
                    else None
                ),
            },
            {
                "roic": 0.50,
                "fcf_margin": 0.30,
                "gross_margin": 0.20,
            }
        )

    # --------------------------------------------------------
    # Debt
    # --------------------------------------------------------

    debt = get_total_debt(data)
    cash = get_cash(data)
    equity = get_total_equity(data)

    debt_equity = (
        debt / equity
        if (
            debt is not None
            and equity is not None
            and equity != 0
        )
        else None
    )

    cash_debt = (
        cash / debt
        if (
            cash is not None
            and debt is not None
            and debt > 0
        )
        else None
    )

    current_assets = get_current_assets(data)
    current_liabilities = get_current_liabilities(data)

    current_ratio = (
        current_assets /
        current_liabilities
        if (
            current_assets is not None
            and current_liabilities is not None
            and current_liabilities > 0
        )
        else None
    )

    if financial:

        debt_score = weighted_score(
            {
                "de": (
                    sigmoid_score(
                        debt_equity,
                        midpoint=0.8,
                        scale=0.7,
                        inverse=True
                    )
                    if debt_equity is not None
                    else None
                ),
                "cash_debt": (
                    sigmoid_score(
                        cash_debt,
                        midpoint=0.5,
                        scale=0.5
                    )
                    if cash_debt is not None
                    else None
                ),
            },
            {
                "de": 0.60,
                "cash_debt": 0.40,
            }
        )

    else:

        debt_score = weighted_score(
            {
                "de": (
                    sigmoid_score(
                        debt_equity,
                        midpoint=0.8,
                        scale=0.7,
                        inverse=True
                    )
                    if debt_equity is not None
                    else None
                ),
                "cash_debt": (
                    sigmoid_score(
                        cash_debt,
                        midpoint=0.5,
                        scale=0.5
                    )
                    if cash_debt is not None
                    else None
                ),
                "current": (
                    sigmoid_score(
                        current_ratio,
                        midpoint=1.5,
                        scale=0.8
                    )
                    if current_ratio is not None
                    else None
                ),
            },
            {
                "de": 0.60,
                "cash_debt": 0.25,
                "current": 0.15,
            }
        )

    # --------------------------------------------------------
    # Valuation
    # --------------------------------------------------------

    forward_pe = get_forward_pe(data)
    peg = get_peg(data)

    if financial:

        pb = get_price_book(data)

        valuation = weighted_score(
            {
                "forward_pe": (
                    sigmoid_score(
                        forward_pe,
                        midpoint=22,
                        scale=10,
                        inverse=True
                    )
                    if forward_pe is not None
                    and forward_pe > 0
                    else None
                ),
                "pb": (
                    sigmoid_score(
                        pb,
                        midpoint=1.8,
                        scale=0.8,
                        inverse=True
                    )
                    if pb is not None
                    and pb > 0
                    else None
                ),
            },
            {
                "forward_pe": 0.60,
                "pb": 0.40,
            }
        )

    else:

        ps = get_price_sales(data)

        valuation = weighted_score(
            {
                "forward_pe": (
                    sigmoid_score(
                        forward_pe,
                        midpoint=22,
                        scale=10,
                        inverse=True
                    )
                    if forward_pe is not None
                    and forward_pe > 0
                    else None
                ),
                "peg": (
                    sigmoid_score(
                        peg,
                        midpoint=1.8,
                        scale=0.9,
                        inverse=True
                    )
                    if peg is not None
                    and peg > 0
                    else None
                ),
                "ps": (
                    sigmoid_score(
                        ps,
                        midpoint=5,
                        scale=3,
                        inverse=True
                    )
                    if ps is not None
                    and ps > 0
                    else None
                ),
            },
            {
                "forward_pe": 0.50,
                "peg": 0.30,
                "ps": 0.20,
            }
        )

    # --------------------------------------------------------
    # Final Verify
    # --------------------------------------------------------

    verify = weighted_score(
        {
            "growth": growth_score,
            "profitability": profitability,
            "debt": debt_score,
            "valuation": valuation,
        },
        {
            "growth": 0.25,
            "profitability": 0.25,
            "debt": 0.15,
            "valuation": 0.35,
        }
    )

    components = [
        growth_score,
        profitability,
        debt_score,
        valuation,
    ]

    available = sum(
        x is not None
        for x in components
    )

    verify_data = (
        available / 4 * 100
    )

    return {
        "verify": verify,
        "growth": growth_score,
        "profitability": profitability,
        "debt": debt_score,
        "valuation": valuation,
        "data": verify_data,
    }


# ============================================================
# MODEL CONFIDENCE
# ============================================================

def calculate_model_confidence(
    data,
    fair_value_result,
    verify_result,
    kpax,
    kpax_fv,
    risk,
    technical
):

    components = [
        kpax,
        kpax_fv,
        risk,
        technical,
    ]

    available_score = (
        sum(
            x is not None
            for x in components
        )
        / len(components)
        * 100
    )

    fv_confidence = (
        fair_value_result["confidence"]
    )

    verify_confidence = (
        verify_result["data"]
    )

    # Market history quality

    history = data["history"]

    if (
        history is not None
        and not history.empty
    ):

        history_score = min(
            100,
            len(history) / 500 * 100
        )

    else:

        history_score = 0

    confidence = (
        available_score * 0.35
        + fv_confidence * 0.30
        + verify_confidence * 0.20
        + history_score * 0.15
    )

    return clip_score(confidence)


# ============================================================
# RECOMMENDATION
# ============================================================

def base_recommendation(
    investment_score,
    kpax,
    kpax_fv
):

    if investment_score is None:
        return "⚪ Keine Bewertung"

    if (
        investment_score >= 85
        and kpax is not None
        and kpax >= 80
        and kpax_fv is not None
        and kpax_fv >= 80
    ):
        return "🟢 Strong Buy"

    if (
        investment_score >= 80
        and kpax is not None
        and kpax >= 75
    ):
        return "🟢 Buy"

    if investment_score >= 75:
        return "🟢 Accumulate"

    if investment_score >= 68:
        return "🟡 Hold"

    if investment_score >= 55:
        return "🟠 Reduce / Watch"

    return "🔴 Avoid"


def apply_technical_filter(
    recommendation,
    technical
):

    if technical is None:
        return recommendation

    if technical < 30:

        if (
            recommendation
            in ["🟢 Strong Buy", "🟢 Buy",
                "🟢 Accumulate", "🟡 Hold"]
        ):
            return "🟡 Hold"

    elif technical < 45:

        if recommendation in [
            "🟢 Strong Buy",
            "🟢 Buy",
            "🟢 Accumulate"
        ]:
            return "🟢 Accumulate"

    elif technical < 60:

        if recommendation == "🟢 Strong Buy":
            return "🟢 Buy"

    return recommendation


# ============================================================
# MODEL CALCULATION
# ============================================================

def calculate_stock(data, score_weights):

    price = get_price(data)

    sector = get_sector(data)

    quality = calculate_quality(data)

    future = calculate_future(data)

    kpax = calculate_kpax(
        quality,
        future
    )

    fair_value_result = (
        calculate_fair_value(data)
    )

    fair_value_score = (
        fair_value_result["score"]
    )

    relative = (
        calculate_relative_valuation(data)
    )

    kpax_fv = calculate_kpax_fv(
        fair_value_score,
        relative
    )

    risk = calculate_risk(data)

    technical, technical_details = (
        calculate_technical(data)
    )

    # --------------------------------------------------------
    # Investment Score
    # --------------------------------------------------------

    investment_score = weighted_score(
        {
            "kpax": kpax,
            "kpax_fv": kpax_fv,
            "risk": risk,
        },
        score_weights
    )

    verify_result = calculate_verify(data)

    confidence = calculate_model_confidence(
        data,
        fair_value_result,
        verify_result,
        kpax,
        kpax_fv,
        risk,
        technical
    )

    recommendation = base_recommendation(
        investment_score,
        kpax,
        kpax_fv
    )

    recommendation = apply_technical_filter(
        recommendation,
        technical
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify = verify_result["verify"]

    gap = None

    if (
        investment_score is not None
        and verify is not None
    ):
        gap = (
            investment_score
            - verify
        )

    if verify_result["data"] < 60:

        verify_status = (
            "⚪ Datenbasis schwach"
        )

    elif gap is None:

        verify_status = (
            "⚪ Nicht prüfbar"
        )

    elif abs(gap) <= 10:

        verify_status = (
            "🟢 Plausibel"
        )

    elif abs(gap) <= 20:

        verify_status = (
            "🟡 Abweichung"
        )

    else:

        verify_status = (
            "🔴 Stark abweichend"
        )

    return {
        "Ticker": data["ticker"],
        "Price": price,
        "Sector": sector,

        "Quality": quality,
        "Future": future,
        "KPAX": kpax,

        "Fair Value Score": fair_value_score,
        "Fair Value": fair_value_result["fair_value"],
        "Relative Valuation": relative,
        "KPAX-FV": kpax_fv,

        "Risk": risk,
        "Technical": technical,

        "Investment Score": investment_score,
        "Recommendation": recommendation,

        "Model Confidence": confidence,

        "Verify": verify,
        "Verify Gap": gap,
        "Verify Status": verify_status,
        "Verify Data": verify_result["data"],

        "FV Confidence": (
            fair_value_result["confidence"]
        ),

        "Earnings FV": (
            fair_value_result["earnings_fv"]
        ),

        "FCF FV": (
            fair_value_result["fcf_fv"]
        ),

        "Analyst FV": (
            fair_value_result["analyst_fv"]
        ),

        "Analyst Upside": (
            get_analyst_upside(data)
        ),

        "Earnings Growth": (
            get_historical_earnings_growth(data)
        ),

        "Forward Earnings Growth": (
            get_forward_earnings_growth(data)
        ),

        "Revenue Growth": (
            get_revenue_growth(data)
        ),

        "Long-Term Growth": (
            get_long_term_growth(data)
        ),

        "FCF Growth": (
            get_fcf_growth(data)
        ),

        "Forward PE": (
            get_forward_pe(data)
        ),

        "PEG": (
            get_peg(data)
        ),

        "P/S": (
            get_price_sales(data)
        ),

        "P/B": (
            get_price_book(data)
        ),

        "Technical Details": technical_details,

        "Verify Growth": (
            verify_result["growth"]
        ),

        "Verify Profitability": (
            verify_result["profitability"]
        ),

        "Verify Debt": (
            verify_result["debt"]
        ),

        "Verify Valuation": (
            verify_result["valuation"]
        ),
    }


# ============================================================
# STREAMLIT SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Modell")

st.sidebar.caption(
    f"Aktien-Screener {VERSION}"
)

st.sidebar.markdown(
    "### Investment Score"
)

kpax_weight = st.sidebar.number_input(
    "KPAX",
    min_value=0.0,
    max_value=1.0,
    value=INITIAL_SCORE_WEIGHTS["kpax"],
    step=0.05,
    format="%.2f"
)

kpax_fv_weight = st.sidebar.number_input(
    "KPAX-FV",
    min_value=0.0,
    max_value=1.0,
    value=INITIAL_SCORE_WEIGHTS["kpax_fv"],
    step=0.05,
    format="%.2f"
)

risk_weight = st.sidebar.number_input(
    "Risk",
    min_value=0.0,
    max_value=1.0,
    value=INITIAL_SCORE_WEIGHTS["risk"],
    step=0.05,
    format="%.2f"
)

if st.sidebar.button(
    "🔄 Gewichte zurücksetzen",
    use_container_width=True
):

    st.session_state[
        "reset_weights"
    ] = True

    st.rerun()


if st.session_state.get(
    "reset_weights",
    False
):

    kpax_weight = INITIAL_SCORE_WEIGHTS["kpax"]
    kpax_fv_weight = INITIAL_SCORE_WEIGHTS["kpax_fv"]
    risk_weight = INITIAL_SCORE_WEIGHTS["risk"]

    st.session_state[
        "reset_weights"
    ] = False


score_weights = normalize_weights(
    {
        "kpax": kpax_weight,
        "kpax_fv": kpax_fv_weight,
        "risk": risk_weight,
    }
)

weight_total = (
    kpax_weight
    + kpax_fv_weight
    + risk_weight
)

st.sidebar.caption(
    f"Summe: {weight_total:.2f}"
)

if abs(weight_total - 1) > 0.001:

    st.sidebar.info(
        "Gewichte werden automatisch "
        "auf 100 % normalisiert."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    f"📊 Aktien-Screener {VERSION}"
)

st.caption(
    "Fundamentaler Multi-Faktor-Screener "
    "mit robustem Fair Value, KPAX, Risk, "
    "Technical Filter und unabhängigem Verify."
)


# ============================================================
# LOAD DATA
# ============================================================

progress = st.progress(0)

results = []

for i, ticker in enumerate(TICKERS):

    data = load_ticker_data(ticker)

    if data["valid"]:

        try:

            result = calculate_stock(
                data,
                score_weights
            )

            results.append(result)

        except Exception as e:

            results.append({
                "Ticker": ticker,
                "Price": get_price(data),
                "Investment Score": None,
                "Recommendation":
                    "⚪ Berechnungsfehler",
                "Model Confidence": 0,
                "Verify": None,
                "Verify Gap": None,
                "Verify Status":
                    f"Fehler: {str(e)[:60]}",
            })

    else:

        results.append({
            "Ticker": ticker,
            "Price": None,
            "Investment Score": None,
            "Recommendation":
                "⚪ Keine Daten",
            "Model Confidence": 0,
            "Verify": None,
            "Verify Gap": None,
            "Verify Status":
                "⚪ Keine Daten",
        })

    progress.progress(
        (i + 1) / len(TICKERS)
    )

progress.empty()


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(results)


if df.empty:

    st.error(
        "Keine Daten verfügbar."
    )

    st.stop()


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    by="Investment Score",
    ascending=False,
    na_position="last"
).reset_index(drop=True)


df.insert(
    0,
    "Rank",
    np.arange(1, len(df) + 1)
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def fmt_score(x):

    x = safe_float(x)

    if x is None:
        return "—"

    return f"{x:.1f}"


def fmt_price(x):

    x = safe_float(x)

    if x is None:
        return "—"

    return f"{x:.2f}"


def fmt_pct(x):

    x = safe_float(x)

    if x is None:
        return "—"

    return f"{x:+.1f}%"


# ============================================================
# MAIN TABLE
# ============================================================

st.subheader(
    "🏆 Investment Ranking"
)

display_columns = [
    "Rank",
    "Ticker",
    "Price",
    "Investment Score",
    "Recommendation",
    "KPAX",
    "KPAX-FV",
    "Fair Value Score",
    "Fair Value",
    "Relative Valuation",
    "Risk",
    "Technical",
    "Model Confidence",
    "Verify",
    "Verify Gap",
    "Verify Status",
]


display_df = df[
    [
        col
        for col in display_columns
        if col in df.columns
    ]
].copy()


if "Price" in display_df.columns:

    display_df["Price"] = (
        display_df["Price"]
        .map(fmt_price)
    )


score_columns = [
    "Investment Score",
    "KPAX",
    "KPAX-FV",
    "Fair Value Score",
    "Relative Valuation",
    "Risk",
    "Technical",
    "Model Confidence",
    "Verify",
    "Verify Gap",
]


for col in score_columns:

    if col in display_df.columns:

        display_df[col] = (
            display_df[col]
            .map(fmt_score)
        )


if "Fair Value" in display_df.columns:

    display_df["Fair Value"] = (
        display_df["Fair Value"]
        .map(fmt_price)
    )


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=850
)


# ============================================================
# MODEL DIAGNOSTICS
# ============================================================

st.divider()

st.subheader(
    "🔍 Modell-Diagnose"
)

col1, col2, col3, col4 = st.columns(4)

valid_scores = df[
    "Investment Score"
].dropna()

valid_verify = df[
    "Verify"
].dropna()

valid_conf = df[
    "Model Confidence"
].dropna()

with col1:

    st.metric(
        "Aktien",
        len(df)
    )

with col2:

    st.metric(
        "Ø Investment Score",
        (
            f"{valid_scores.mean():.1f}"
            if len(valid_scores)
            else "—"
        )
    )

with col3:

    st.metric(
        "Ø Verify",
        (
            f"{valid_verify.mean():.1f}"
            if len(valid_verify)
            else "—"
        )
    )

with col4:

    st.metric(
        "Ø Model Confidence",
        (
            f"{valid_conf.mean():.1f}"
            if len(valid_conf)
            else "—"
        )
    )


# ============================================================
# VERIFY TABLE
# ============================================================

st.subheader(
    "🧪 Independent Verify"
)

verify_columns = [
    "Ticker",
    "Investment Score",
    "Verify",
    "Verify Gap",
    "Verify Status",
    "Verify Data",
    "Verify Growth",
    "Verify Profitability",
    "Verify Debt",
    "Verify Valuation",
]

verify_df = df[
    [
        c
        for c in verify_columns
        if c in df.columns
    ]
].copy()

for col in [
    "Investment Score",
    "Verify",
    "Verify Gap",
    "Verify Data",
    "Verify Growth",
    "Verify Profitability",
    "Verify Debt",
    "Verify Valuation",
]:

    if col in verify_df.columns:

        verify_df[col] = (
            verify_df[col]
            .map(fmt_score)
        )


st.dataframe(
    verify_df,
    use_container_width=True,
    hide_index=True,
    height=700
)


# ============================================================
# FAIR VALUE DETAIL
# ============================================================

st.subheader(
    "💰 Fair Value Diagnose"
)

fv_columns = [
    "Ticker",
    "Price",
    "Fair Value",
    "Fair Value Score",
    "FV Confidence",
    "Earnings FV",
    "FCF FV",
    "Analyst FV",
    "Analyst Upside",
]

fv_df = df[
    [
        c
        for c in fv_columns
        if c in df.columns
    ]
].copy()


for col in [
    "Price",
    "Fair Value",
    "Earnings FV",
    "FCF FV",
]:

    if col in fv_df.columns:

        fv_df[col] = (
            fv_df[col]
            .map(fmt_price)
        )


for col in [
    "Fair Value Score",
    "FV Confidence",
]:

    if col in fv_df.columns:

        fv_df[col] = (
            fv_df[col]
            .map(fmt_score)
        )


if "Analyst FV" in fv_df.columns:

    fv_df["Analyst FV"] = (
        fv_df["Analyst FV"]
        .map(fmt_price)
    )


if "Analyst Upside" in fv_df.columns:

    fv_df["Analyst Upside"] = (
        fv_df["Analyst Upside"]
        .map(fmt_pct)
    )


st.dataframe(
    fv_df,
    use_container_width=True,
    hide_index=True,
    height=700
)


# ============================================================
# GROWTH DIAGNOSTICS
# ============================================================

st.subheader(
    "🚀 Growth / Future Diagnose"
)

growth_columns = [
    "Ticker",
    "Earnings Growth",
    "Forward Earnings Growth",
    "Revenue Growth",
    "Long-Term Growth",
    "FCF Growth",
    "Quality",
    "Future",
    "KPAX",
]

growth_df = df[
    [
        c
        for c in growth_columns
        if c in df.columns
    ]
].copy()


for col in [
    "Earnings Growth",
    "Forward Earnings Growth",
    "Revenue Growth",
    "Long-Term Growth",
    "FCF Growth",
]:

    if col in growth_df.columns:

        growth_df[col] = (
            growth_df[col]
            .map(fmt_pct)
        )


for col in [
    "Quality",
    "Future",
    "KPAX",
]:

    if col in growth_df.columns:

        growth_df[col] = (
            growth_df[col]
            .map(fmt_score)
        )


st.dataframe(
    growth_df,
    use_container_width=True,
    hide_index=True,
    height=700
)


# ============================================================
# FORMULAS
# ============================================================

st.divider()

st.subheader(
    "📐 Mathematik des Modells"
)

with st.expander(
    "Alle Berechnungsformeln anzeigen",
    expanded=False
):

    st.markdown(
        """
## 1. Investment Score

Der finale Investment Score besteht aus drei unabhängigen Bereichen:

**Investment Score =**

`KPAX × w₁ + KPAX-FV × w₂ + Risk × w₃`

Die Standardgewichtung lautet:

- KPAX = **40 %**
- KPAX-FV = **50 %**
- Risk = **10 %**

Wenn die Gewichte geändert werden, werden sie automatisch auf 100 % normalisiert.

---

## 2. KPAX

KPAX misst die operative Qualität und die zukünftige Entwicklung.

`KPAX = 40 % × Quality + 60 % × Future`

### Quality

Bei Nicht-Finanzunternehmen:

- ROIC: 30 %
- Bruttomarge: 15 %
- FCF-Marge: 20 %
- Gewinnwachstum: 20 %
- Umsatzwachstum: 15 %

Bei Finanzunternehmen:

- ROE: 40 %
- Nettomarge: 30 %
- Gewinnwachstum: 20 %
- Umsatzwachstum: 10 %

Fehlende Komponenten werden nicht mit 0 bewertet. Die vorhandenen Gewichte werden renormalisiert.

---

## 3. Future

V20.3 verwendet bewusst unterschiedliche Wachstumsinformationen.

`Future =`

- Gewinnwachstum historisch: 25 %
- Umsatzwachstum: 20 %
- erwartetes Gewinnwachstum: 20 %
- langfristiges Wachstum: 15 %
- FCF-Marge: 10 %
- Wachstumsbeschleunigung: 10 %

### Wachstumsbeschleunigung

`Acceleration = Forward Earnings Growth − Historical Earnings Growth`

Dadurch wird nicht mehr einfach ein vorhandener Wachstumswert ein zweites Mal verwendet.

---

## 4. Fair Value

V20.3 verwendet drei unabhängige Modelle.

### Earnings Fair Value

`Fair P/E = 12 + 0,25 × Growth`

anschließend:

`Fair P/E = min(24, max(12, Fair P/E))`

und:

`Earnings FV = Forward EPS × Fair P/E`

Das ist bewusst weniger aggressiv als das alte Modell.

---

### FCF Fair Value

Zunächst:

`FCF je Aktie = FCF / Aktienanzahl`

Dann:

`Target FCF Yield = 7,0 % − 0,08 × Growth`

mit Begrenzung auf:

`4,5 % ≤ Target Yield ≤ 9,0 %`

Bei höherem Beta wird zusätzlich ein Risikoaufschlag auf die Renditeforderung angewendet.

Danach:

`FCF FV = FCF je Aktie / Target FCF Yield`

---

### Analyst Fair Value

Hier wird das durchschnittliche Analystenkursziel verwendet.

---

### Robustheitsfilter

Die einzelnen Fair-Value-Modelle werden zunächst verglichen.

Der Median der verfügbaren Modelle dient als Referenz.

Kein einzelnes Modell darf anschließend mehr als:

`0,40 × Median`

oder weniger als:

`2,50 × Median`

vom Median entfernt liegen.

Dadurch kann ein einzelner fehlerhafter Yahoo-Datenpunkt den Fair Value nicht mehr komplett zerstören.

---

### Fair Value Score

Zunächst:

`Upside = Fair Value / aktueller Kurs − 1`

Dann:

`FV Score = 55 + 0,72 × Upside[%]`

anschließend Begrenzung auf:

`5 ... 100`

---

## 5. KPAX-FV

`KPAX-FV = 60 % × Fair Value Score + 40 % × Relative Valuation`

---

## 6. Relative Valuation

Für normale Unternehmen:

- Forward P/E: 35 %
- Trailing P/E: 10 %
- PEG: 30 %
- P/S: 25 %

Für Finanzunternehmen:

- Forward P/E: 35 %
- Trailing P/E: 15 %
- PEG: 15 %
- P/B: 35 %

Die einzelnen Multiples werden über eine Sigmoid-Funktion bewertet.

Das verhindert harte Score-Sprünge.

---

## 7. Risk

Risk ist ein **Sicherheits-Score**:

Hoher Score = geringeres Risiko.

Er besteht aus:

- Financial Risk: 45 %
- Market Risk: 35 %
- Stability/Beta: 20 %

Market Risk verwendet echte historische Daten:

- annualisierte Volatilität
- maximaler Drawdown

---

## 8. Technical

Technical wird **nicht** in den Investment Score eingerechnet.

Er dient ausschließlich als Timing-/Trendfilter.

Enthalten sind:

- SMA200: 25 %
- 6-Monats-Performance: 25 %
- RSI: 20 %
- MACD: 20 %
- SMA50: 10 %

Der Technical Score kann lediglich die Empfehlung nach unten korrigieren.

---

## 9. Recommendation

### Strong Buy

`Investment Score ≥ 85`

und gleichzeitig:

`KPAX ≥ 80`

und:

`KPAX-FV ≥ 80`

### Buy

`Investment Score ≥ 80`

und:

`KPAX ≥ 75`

### Accumulate

`Investment Score ≥ 75`

### Hold

`Investment Score ≥ 68`

### Reduce / Watch

`Investment Score ≥ 55`

### Avoid

`Investment Score < 55`

Der Technical Score kann diese Empfehlung bei schwacher Charttechnik nach unten korrigieren.

---

## 10. Independent Verify

Der Verify ist bewusst **nicht identisch** mit dem Investment Score.

Er verwendet:

- Wachstum: 25 %
- Profitabilität: 25 %
- Verschuldung: 15 %
- Bewertung: 35 %

Damit kann der Verify erkennen, ob der eigentliche Score durch einzelne Modellkomponenten stark verzerrt wird.

### Interpretation

**Gap = Investment Score − Verify**

- ≤ 10 Punkte Differenz → 🟢 plausibel
- 10–20 Punkte → 🟡 Abweichung
- > 20 Punkte → 🔴 starke Abweichung

Bei weniger als 60 % Datenbasis:

**⚪ Datenbasis schwach**

---

## 11. Model Confidence

Model Confidence ist **kein zusätzlicher Investment Score**.

Sie beantwortet ausschließlich:

> Wie vollständig und belastbar ist die Datenbasis?

Einbezogen werden:

- verfügbare Score-Komponenten
- Anzahl Fair-Value-Modelle
- Verify-Datenbasis
- verfügbare Kurs-Historie

Damit kann man beispielsweise unterscheiden:

`Investment Score 85 / Confidence 95`

von:

`Investment Score 85 / Confidence 48`

Der erste Wert ist wesentlich belastbarer.
        """
    )


# ============================================================
# RAW DATA / DEBUG
# ============================================================

with st.expander(
    "🛠️ Technische Debug-Daten",
    expanded=False
):

    debug_columns = [
        "Ticker",
        "Price",
        "Investment Score",
        "KPAX",
        "KPAX-FV",
        "Fair Value Score",
        "Relative Valuation",
        "Risk",
        "Technical",
        "Model Confidence",
        "Verify",
        "Verify Gap",
    ]

    st.dataframe(
        df[
            [
                c
                for c in debug_columns
                if c in df.columns
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    f"Aktien-Screener {VERSION} • "
    "yfinance • Multi-Faktor-Modell • "
    "Scores 5–100"
)
