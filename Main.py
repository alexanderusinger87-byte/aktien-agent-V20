import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aktien-Screener V20.2",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# DEFAULT TICKERS
# ============================================================

DEFAULT_TICKERS = (
    "SMO, BMW.DE, MAIR, GOOGL, IFX.DE, 1810.HK, PEP, KO, "
    "NVDA, AAPL, MU, INTC, AMZN, F8P, NOK, VST, ASML, NKE, "
    "VRT, TSM, NVO, MRVL, 000660.KS, TSLA, 005930.KS, AMD, "
    "ADS.DE, SU.PA, ENR.DE, SIE.DE, MSFT, AVGO, SSUN.F, MCD, "
    "MUV2.DE, ALV.DE"
)


# ============================================================
# INITIAL WEIGHTS
# ============================================================

INITIAL_SCORE_WEIGHTS = {
    "kpax": 0.40,
    "kpax_fv": 0.50,
    "risk": 0.10
}

MIN_VALID_SCORE = 5.0


# ============================================================
# FALLBACK SECTORS
# ============================================================

FALLBACK_SECTORS = {
    "SMO": "Industrials",
    "BMW.DE": "Consumer Cyclical",
    "MAIR": "Technology",
    "GOOGL": "Technology",
    "IFX.DE": "Technology",
    "1810.HK": "Technology",
    "PEP": "Consumer Defensive",
    "KO": "Consumer Defensive",
    "NVDA": "Technology",
    "AAPL": "Technology",
    "MU": "Technology",
    "INTC": "Technology",
    "AMZN": "Consumer Cyclical",
    "F8P": "Technology",
    "NOK": "Technology",
    "VST": "Utilities",
    "ASML": "Technology",
    "NKE": "Consumer Cyclical",
    "VRT": "Technology",
    "TSM": "Technology",
    "NVO": "Healthcare",
    "MRVL": "Technology",
    "000660.KS": "Technology",
    "TSLA": "Consumer Cyclical",
    "005930.KS": "Technology",
    "AMD": "Technology",
    "ADS.DE": "Consumer Cyclical",
    "SU.PA": "Energy",
    "ENR.DE": "Industrials",
    "SIE.DE": "Industrials",
    "MSFT": "Technology",
    "AVGO": "Technology",
    "SSUN.F": "Technology",
    "MCD": "Consumer Cyclical",
    "MUV2.DE": "Financial Services",
    "ALV.DE": "Financial Services",
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

    "total_assets": [
        "Total Assets",
        "TotalAssets"
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
# BASIC HELPERS
# ============================================================

def safe_get(obj, key, default=np.nan):

    try:

        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(key, default)

        return getattr(obj, key, default)

    except Exception:
        return default


def to_number(value):

    try:

        if value is None:
            return np.nan

        if isinstance(value, str):
            value = (
                value.replace(",", "")
                .replace("%", "")
            )

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:
        return np.nan


def clean_positive(value):

    value = to_number(value)

    if np.isnan(value) or value <= 0:
        return np.nan

    return value


def clean_percentage(value):

    value = to_number(value)

    if np.isnan(value):
        return np.nan

    if abs(value) < 1:
        return value * 100

    return value


def clip_score(value):

    if value is None:
        return np.nan

    try:
        value = float(value)
    except Exception:
        return np.nan

    if not np.isfinite(value):
        return np.nan

    return float(
        np.clip(
            value,
            MIN_VALID_SCORE,
            100
        )
    )


def weighted_available(values, weights):

    valid = []

    for key, weight in weights.items():

        value = values.get(key, np.nan)

        if (
            value is not None
            and np.isfinite(value)
            and weight > 0
        ):
            valid.append(
                (float(value), weight)
            )

    if not valid:
        return np.nan

    total_weight = sum(
        weight for _, weight in valid
    )

    if total_weight <= 0:
        return np.nan

    result = sum(
        value * weight
        for value, weight in valid
    ) / total_weight

    return clip_score(result)


def get_row(df, aliases):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
    ):
        return pd.Series(dtype=float)

    for alias in aliases:

        if alias in df.index:
            return df.loc[alias]

    return pd.Series(dtype=float)


def get_first_valid_row(df, aliases):

    row = get_row(
        df,
        aliases
    )

    if row.empty:
        return row

    return row.dropna()


def get_numeric_series(row):

    if row is None or len(row) == 0:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        row,
        errors="coerce"
    ).dropna()


def safe_mean(values):

    values = [
        x for x in values
        if x is not None
        and np.isfinite(x)
    ]

    if not values:
        return np.nan

    return float(
        np.mean(values)
    )


# ============================================================
# SCORE TRANSFORMATIONS
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


def score_cash_debt(value):

    value = to_number(value)

    if np.isnan(value):
        return np.nan

    return np.interp(
        value,
        [0, .25, .5, 1, 2, 4],
        [5, 25, 50, 75, 90, 100]
    )


# ============================================================
# DATA FETCH
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
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
        "growth_estimate": pd.DataFrame(),
    }

    try:

        stock = yf.Ticker(ticker)

        try:
            result["info"] = (
                stock.info or {}
            )
        except Exception:
            pass

        try:
            result["fast_info"] = dict(
                stock.fast_info
            )
        except Exception:
            pass

        try:
            result["history"] = stock.history(
                period="2y",
                auto_adjust=False
            )
        except Exception:
            pass

        try:
            result["financials"] = (
                stock.financials
            )
        except Exception:
            pass

        try:
            result["balance_sheet"] = (
                stock.balance_sheet
            )
        except Exception:
            pass

        try:
            result["cashflow"] = (
                stock.cashflow
            )
        except Exception:
            pass

        try:
            result["ttm_income"] = (
                stock.ttm_income_stmt
            )
        except Exception:
            pass

        try:
            result["ttm_cashflow"] = (
                stock.ttm_cashflow
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
            result["analyst_targets"] = (
                stock.analyst_price_targets
            )
        except Exception:
            pass

        try:
            result["growth_estimate"] = (
                stock.growth_estimates
            )
        except Exception:
            pass

    except Exception:
        pass

    return result


# ============================================================
# MARKET DATA
# ============================================================

def get_current_price(data):

    info = data["info"]
    fast = data["fast_info"]

    candidates = [
        safe_get(
            info,
            "currentPrice"
        ),
        safe_get(
            info,
            "regularMarketPrice"
        ),
        safe_get(
            fast,
            "lastPrice"
        )
    ]

    for value in candidates:

        value = to_number(value)

        if (
            not np.isnan(value)
            and value > 0
        ):
            return value

    history = data["history"]

    if (
        isinstance(history, pd.DataFrame)
        and not history.empty
        and "Close" in history
    ):

        values = pd.to_numeric(
            history["Close"],
            errors="coerce"
        ).dropna()

        if len(values):
            return float(
                values.iloc[-1]
            )

    return np.nan


def get_market_cap(data):

    info = data["info"]
    fast = data["fast_info"]

    for value in [
        safe_get(info, "marketCap"),
        safe_get(fast, "marketCap")
    ]:

        value = to_number(value)

        if (
            not np.isnan(value)
            and value > 0
        ):
            return value

    return np.nan


def get_sector(ticker, info):

    sector = safe_get(
        info,
        "sector",
        None
    )

    if (
        sector
        and sector != "N/A"
    ):
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
        safe_get(
            info,
            "trailingPE"
        )
    )

    forward_pe = clean_positive(
        safe_get(
            info,
            "forwardPE"
        )
    )

    ps = clean_positive(
        safe_get(
            info,
            "priceToSalesTrailing12Months"
        )
    )

    pb = clean_positive(
        safe_get(
            info,
            "priceToBook"
        )
    )

    trailing_eps = clean_positive(
        safe_get(
            info,
            "trailingEps"
        )
    )

    forward_eps = clean_positive(
        safe_get(
            info,
            "forwardEps"
        )
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
            ps = (
                market_cap / revenue
            )

    equity_row = get_first_valid_row(
        data["quarterly_balance"],
        [
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest"
        ]
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
            pb = (
                market_cap / equity
            )

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

    direct = to_number(
        safe_get(
            data["info"],
            "freeCashflow"
        )
    )

    if (
        not np.isnan(direct)
        and direct != 0
    ):
        return direct

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

    cfo = float(
        cfo_row.iloc[0]
    )

    capex = (
        float(capex_row.iloc[0])
        if not capex_row.empty
        else 0
    )

    return cfo - abs(capex)


# ============================================================
# ANALYST TARGET
# ============================================================

def extract_analyst_target(data):

    targets = data[
        "analyst_targets"
    ]

    if (
        isinstance(
            targets,
            pd.DataFrame
        )
        and not targets.empty
    ):

        for key in [
            "median",
            "mean"
        ]:

            if key in targets.index:

                values = pd.to_numeric(
                    targets.loc[key],
                    errors="coerce"
                ).dropna()

                if len(values):

                    value = float(
                        values.iloc[0]
                    )

                    if value > 0:
                        return value

    info = data["info"]

    for key in [
        "targetMedianPrice",
        "targetMeanPrice"
    ]:

        value = clean_positive(
            safe_get(
                info,
                key
            )
        )

        if not np.isnan(value):
            return value

    return np.nan


# ============================================================
# FAIR VALUE
# ============================================================

def calculate_fair_value(
    data,
    metrics
):

    sector = metrics["sector"]
    price = metrics["price"]

    forward_eps = metrics[
        "forward_eps"
    ]

    growth = metrics[
        "earnings_growth"
    ]

    fcf = metrics["fcf"]
    shares = metrics["shares"]
    beta = metrics["beta"]

    analyst_target = (
        extract_analyst_target(data)
    )

    financial = (
        sector == "Financial Services"
    )

    models = []

    # --------------------------------------------------------
    # Earnings FV
    # --------------------------------------------------------

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
                (
                    "earnings",
                    earnings_fv,
                    0.50
                )
            )

    # --------------------------------------------------------
    # FCF FV
    # --------------------------------------------------------

    if (
        not financial
        and not np.isnan(fcf)
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
                0.005
                * (beta - 1)
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
                (
                    "fcf",
                    fcf_fv,
                    0.30
                )
            )

    # --------------------------------------------------------
    # Financials: simplified justified PB
    # --------------------------------------------------------

    if financial:

        roe = metrics["roe"]
        pb = metrics["pb"]

        if (
            not np.isnan(roe)
            and roe > 0
            and not np.isnan(pb)
            and pb > 0
            and price > 0
        ):

            beta_value = (
                beta
                if not np.isnan(beta)
                else 1
            )

            ke = np.clip(
                0.04
                + beta_value * 0.055,
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
                        (
                            "pb_roe",
                            fair_value,
                            0.45
                        )
                    )

    # --------------------------------------------------------
    # Analyst
    # --------------------------------------------------------

    if (
        not np.isnan(analyst_target)
        and analyst_target > 0
    ):

        models.append(
            (
                "analyst",
                analyst_target,
                0.20
            )
        )

    if not models:

        return {
            "fair_value": np.nan,
            "fair_value_score": np.nan,
            "fair_value_upside": np.nan
        }

    total_weight = sum(
        x[2] for x in models
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
        score = np.nan

    else:

        upside = (
            fair_value / price - 1
        ) * 100

        score = np.interp(
            upside,
            [-40, -20, 0, 10, 25, 50, 75],
            [5, 15, 40, 58, 75, 90, 100]
        )

        score = clip_score(score)

    return {
        "fair_value": fair_value,
        "fair_value_score": score,
        "fair_value_upside": upside
    }


# ============================================================
# TECHNICAL DATA
# ============================================================

def calculate_technical(
    history
):

    empty = {
        "sma50": np.nan,
        "sma200": np.nan,
        "rsi": np.nan,
        "macd": np.nan,
        "perf_1m": np.nan,
        "perf_6m": np.nan,
        "perf_12m": np.nan,
        "volatility_1y": np.nan,
        "max_drawdown_1y": np.nan
    }

    if (
        not isinstance(
            history,
            pd.DataFrame
        )
        or history.empty
        or "Close" not in history
    ):
        return empty

    close = pd.to_numeric(
        history["Close"],
        errors="coerce"
    ).dropna()

    if len(close) < 20:
        return empty

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

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    rsi_series = (
        100
        - 100 / (1 + rs)
    )

    rsi = (
        rsi_series.iloc[-1]
        if len(rsi_series)
        else np.nan
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

    macd_line = (
        ema12 - ema26
    )

    signal = macd_line.ewm(
        span=9,
        adjust=False
    ).mean()

    macd = (
        macd_line.iloc[-1]
        - signal.iloc[-1]
    )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    def performance(days):

        if len(close) <= days:
            return np.nan

        old = close.iloc[
            -days - 1
        ]

        new = close.iloc[-1]

        if old <= 0:
            return np.nan

        return (
            new / old - 1
        ) * 100

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    daily_returns = (
        close.pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(daily_returns) >= 60:

        volatility_1y = (
            daily_returns.tail(252)
            .std()
            * np.sqrt(252)
            * 100
        )

    else:
        volatility_1y = np.nan

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    recent = close.tail(252)

    if len(recent) >= 30:

        rolling_max = (
            recent.cummax()
        )

        drawdown = (
            recent / rolling_max - 1
        )

        max_drawdown_1y = (
            drawdown.min() * 100
        )

    else:
        max_drawdown_1y = np.nan

    return {
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "macd": macd,
        "perf_1m": performance(21),
        "perf_6m": performance(126),
        "perf_12m": performance(252),
        "volatility_1y": volatility_1y,
        "max_drawdown_1y": max_drawdown_1y
    }


# ============================================================
# METRIC EXTRACTION
# ============================================================

def extract_metrics(data):

    ticker = data["ticker"]
    info = data["info"]

    price = get_current_price(
        data
    )

    market_cap = get_market_cap(
        data
    )

    sector = get_sector(
        ticker,
        info
    )

    shares = clean_positive(
        safe_get(
            info,
            "sharesOutstanding"
        )
    )

    if (
        np.isnan(shares)
        and market_cap > 0
        and price > 0
    ):
        shares = (
            market_cap / price
        )

    valuation = get_valuation(
        data
    )

    # --------------------------------------------------------
    # Income
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
    # Historical growth
    # --------------------------------------------------------

    revenue_history = get_row(
        data["financials"],
        ALIASES["revenue"]
    )

    earnings_history = get_row(
        data["financials"],
        ALIASES["net_income"]
    )

    revenue_values = (
        get_numeric_series(
            revenue_history
        )
    )

    earnings_values = (
        get_numeric_series(
            earnings_history
        )
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

    # --------------------------------------------------------
    # Forward growth
    # --------------------------------------------------------

    growth_estimate = np.nan

    ge = data[
        "growth_estimate"
    ]

    if (
        isinstance(ge, pd.DataFrame)
        and not ge.empty
    ):

        preferred_rows = [
            "stock",
            "0y",
            "+1y"
        ]

        for row_name in preferred_rows:

            if row_name in ge.index:

                values = pd.to_numeric(
                    ge.loc[row_name],
                    errors="coerce"
                ).dropna()

                if len(values):

                    growth_estimate = float(
                        values.iloc[0]
                    )

                    break

    if (
        not np.isnan(growth_estimate)
        and abs(growth_estimate) < 1
    ):
        growth_estimate *= 100

    if not np.isnan(
        growth_estimate
    ):
        forward_growth = (
            growth_estimate
        )
    else:
        forward_growth = (
            earnings_growth
        )

    # --------------------------------------------------------
    # Margins
    # --------------------------------------------------------

    net_margin = np.nan

    if (
        revenue > 0
        and not np.isnan(net_income)
    ):

        net_margin = (
            net_income
            / revenue
        ) * 100

    ebit_margin = np.nan

    if (
        revenue > 0
        and not np.isnan(ebit)
    ):

        ebit_margin = (
            ebit
            / revenue
        ) * 100

    gross_margin = clean_percentage(
        safe_get(
            info,
            "grossMargins"
        )
    )

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    roe = clean_percentage(
        safe_get(
            info,
            "returnOnEquity"
        )
    )

    roic = clean_percentage(
        safe_get(
            info,
            "returnOnAssets"
        )
    )

    if np.isnan(roic):

        assets_row = get_first_valid_row(
            data["balance_sheet"],
            ALIASES["total_assets"]
        )

        assets = (
            float(assets_row.iloc[0])
            if not assets_row.empty
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

    fcf = get_free_cash_flow(
        data
    )

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

    debt = (
        float(debt_row.iloc[0])
        if not debt_row.empty
        else np.nan
    )

    cash = (
        float(cash_row.iloc[0])
        if not cash_row.empty
        else np.nan
    )

    net_debt = np.nan

    if (
        not np.isnan(debt)
        and not np.isnan(cash)
    ):

        net_debt = (
            debt - cash
        )

    ebitda = clean_positive(
        safe_get(
            info,
            "ebitda"
        )
    )

    net_debt_ebitda = np.nan

    if (
        not np.isnan(net_debt)
        and not np.isnan(ebitda)
        and ebitda > 0
    ):

        net_debt_ebitda = (
            net_debt / ebitda
        )

    debt_equity = clean_percentage(
        safe_get(
            info,
            "debtToEquity"
        )
    )

    current_ratio = to_number(
        safe_get(
            info,
            "currentRatio"
        )
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
    # Valuation
    # --------------------------------------------------------

    peg = clean_positive(
        safe_get(
            info,
            "pegRatio"
        )
    )

    beta = to_number(
        safe_get(
            info,
            "beta"
        )
    )

    # --------------------------------------------------------
    # Analyst
    # --------------------------------------------------------

    analyst_target = (
        extract_analyst_target(data)
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

    fair_value_input = {
        "sector": sector,
        "price": price,
        "forward_eps": clean_positive(
            safe_get(
                info,
                "forwardEps"
            )
        ),
        "earnings_growth":
            forward_growth,
        "fcf": fcf,
        "shares": shares,
        "beta": beta,
        "roe": roe,
        "pb": valuation["pb"]
    }

    fair_value = (
        calculate_fair_value(
            data,
            fair_value_input
        )
    )

    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------

    core_values = [
        price,
        revenue_growth,
        earnings_growth,
        net_margin,
        valuation["forward_pe"],
        fcf_margin,
        roe,
        debt_equity,
        current_ratio
    ]

    available = sum(
        1
        for x in core_values
        if x is not None
        and np.isfinite(x)
    )

    completeness = (
        available
        / len(core_values)
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

        "revenue_growth":
            revenue_growth,

        "earnings_growth":
            earnings_growth,

        "net_margin":
            net_margin,

        "ebit_margin":
            ebit_margin,

        "gross_margin":
            gross_margin,

        "roe": roe,
        "roic": roic,

        "fcf": fcf,
        "fcf_margin": fcf_margin,
        "fcf_per_share":
            fcf_per_share,

        "debt": debt,
        "cash": cash,
        "net_debt": net_debt,

        "net_debt_ebitda":
            net_debt_ebitda,

        "debt_equity":
            debt_equity,

        "current_ratio":
            current_ratio,

        "cash_to_debt":
            cash_to_debt,

        "trailing_pe":
            valuation["trailing_pe"],

        "forward_pe":
            valuation["forward_pe"],

        "ps": valuation["ps"],
        "pb": valuation["pb"],
        "peg": peg,

        "forward_eps":
            clean_positive(
                safe_get(
                    info,
                    "forwardEps"
                )
            ),

        "analyst_target":
            analyst_target,

        "analyst_upside":
            analyst_upside,

        "beta": beta,

        **technical,

        "fair_value":
            fair_value["fair_value"],

        "fair_value_score":
            fair_value[
                "fair_value_score"
            ],

        "fair_value_upside":
            fair_value[
                "fair_value_upside"
            ],

        "completeness":
            completeness
    }


# ============================================================
# QUALITY
# ============================================================

def calculate_quality(m):

    financial = (
        m["sector"]
        == "Financial Services"
    )

    if financial:

        values = {

            "roe":
                score_roe(
                    m["roe"]
                ),

            "net_margin":
                score_margin(
                    m["net_margin"]
                ),

            "earnings_growth":
                score_growth(
                    m["earnings_growth"]
                ),

            "revenue_growth":
                score_growth(
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

        values = {

            "roic":
                score_roic(
                    m["roic"]
                ),

            "gross_margin":
                score_margin(
                    m["gross_margin"]
                ),

            "fcf_margin":
                score_margin(
                    m["fcf_margin"]
                ),

            "earnings_growth":
                score_growth(
                    m["earnings_growth"]
                ),

            "revenue_growth":
                score_growth(
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
        values,
        weights
    )


# ============================================================
# FUTURE
# ============================================================

def calculate_future(m):

    analyst_score = np.nan

    if not np.isnan(
        m["analyst_upside"]
    ):

        analyst_score = np.interp(
            m["analyst_upside"],
            [-30, -10, 0, 10, 25, 50, 100],
            [5, 25, 40, 60, 75, 90, 100]
        )

    values = {

        "earnings_growth":
            score_growth(
                m["earnings_growth"]
            ),

        "revenue_growth":
            score_growth(
                m["revenue_growth"]
            ),

        "long_term_growth":
            score_growth(
                m["earnings_growth"]
            ),

        "analyst_upside":
            analyst_score,

        "fcf_margin":
            score_margin(
                m["fcf_margin"]
            ),

        "growth_acceleration":
            score_growth(
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
        values,
        weights
    )


# ============================================================
# RELATIVE VALUATION
# ============================================================

def calculate_relative_valuation(m):

    financial = (
        m["sector"]
        == "Financial Services"
    )

    values = {

        "forward_pe":
            score_forward_pe(
                m["forward_pe"]
            ),

        "trailing_pe":
            score_trailing_pe(
                m["trailing_pe"]
            ),

        "peg":
            score_peg(
                m["peg"]
            )
    }

    weights = {

        "forward_pe": 0.40,
        "trailing_pe": 0.20,
        "peg": 0.20
    }

    if financial:

        values["pb"] = score_pb(
            m["pb"]
        )

        weights["pb"] = 0.20

    else:

        values["ps"] = score_ps(
            m["ps"]
        )

        weights["ps"] = 0.20

    return weighted_available(
        values,
        weights
    )


# ============================================================
# RISK
# ============================================================

def calculate_risk(m):

    financial = (
        m["sector"]
        == "Financial Services"
    )

    if financial:

        financial_score = (
            weighted_available(
                {
                    "debt_equity":
                        score_debt_equity(
                            m["debt_equity"]
                        ),

                    "cash_to_debt":
                        score_cash_debt(
                            m["cash_to_debt"]
                        )
                },
                {
                    "debt_equity": 0.60,
                    "cash_to_debt": 0.40
                }
            )
        )

    else:

        financial_score = (
            weighted_available(
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
        )

    # --------------------------------------------------------
    # Market risk
    # --------------------------------------------------------

    volatility = m[
        "volatility_1y"
    ]

    max_drawdown = m[
        "max_drawdown_1y"
    ]

    volatility_score = np.nan

    if not np.isnan(
        volatility
    ):

        volatility_score = np.interp(
            volatility,
            [10, 15, 20, 25, 30, 40, 60],
            [100, 90, 78, 65, 50, 25, 5]
        )

    drawdown_score = np.nan

    if not np.isnan(
        max_drawdown
    ):

        drawdown_score = np.interp(
            max_drawdown,
            [-80, -60, -40, -25, -15, -5, 0],
            [5, 15, 30, 50, 70, 90, 100]
        )

    market_score = weighted_available(
        {
            "volatility":
                volatility_score,

            "drawdown":
                drawdown_score
        },
        {
            "volatility": 0.50,
            "drawdown": 0.50
        }
    )

    # --------------------------------------------------------
    # Business stability
    # --------------------------------------------------------

    stability_values = {

        "revenue":
            score_growth(
                m["revenue_growth"]
            ),

        "earnings":
            score_growth(
                m["earnings_growth"]
            )
    }

    stability = weighted_available(
        stability_values,
        {
            "revenue": 0.40,
            "earnings": 0.60
        }
    )

    return weighted_available(
        {
            "financial":
                financial_score,

            "market":
                market_score,

            "stability":
                stability
        },
        {
            "financial": 0.45,
            "market": 0.30,
            "stability": 0.25
        }
    )


# ============================================================
# TECHNICAL SCORE
# ============================================================

def calculate_technical_score(m):

    sma200_score = np.nan

    if (
        not np.isnan(m["sma200"])
        and not np.isnan(m["price"])
    ):

        sma200_score = (
            80
            if m["price"]
            > m["sma200"]
            else 30
        )

    perf_score = np.nan

    if not np.isnan(
        m["perf_6m"]
    ):

        perf_score = np.interp(
            m["perf_6m"],
            [-50, -20, -10, 0, 10, 25, 50],
            [5, 20, 35, 50, 65, 85, 100]
        )

    rsi_score = np.nan

    if not np.isnan(
        m["rsi"]
    ):

        rsi_score = np.interp(
            m["rsi"],
            [20, 30, 40, 50, 60, 70, 80],
            [30, 55, 75, 90, 100, 75, 35]
        )

    macd_score = np.nan

    if not np.isnan(
        m["macd"]
    ):

        macd_score = (
            75
            if m["macd"] > 0
            else 35
        )

    sma50_score = np.nan

    if (
        not np.isnan(m["sma50"])
        and not np.isnan(m["price"])
    ):

        sma50_score = (
            75
            if m["price"]
            > m["sma50"]
            else 35
        )

    return weighted_available(
        {
            "sma200":
                sma200_score,

            "perf_6m":
                perf_score,

            "rsi":
                rsi_score,

            "macd":
                macd_score,

            "sma50":
                sma50_score
        },
        {
            "sma200": 0.25,
            "perf_6m": 0.25,
            "rsi": 0.20,
            "macd": 0.20,
            "sma50": 0.10
        }
    )


# ============================================================
# MAIN SCORE CALCULATION
# ============================================================

def calculate_scores(
    m,
    score_weights
):

    quality = calculate_quality(
        m
    )

    future = calculate_future(
        m
    )

    kpax = weighted_available(
        {
            "quality":
                quality,

            "future":
                future
        },
        {
            "quality": 0.40,
            "future": 0.60
        }
    )

    relative_valuation = (
        calculate_relative_valuation(
            m
        )
    )

    kpax_fv = weighted_available(
        {
            "fair_value":
                m["fair_value_score"],

            "relative":
                relative_valuation
        },
        {
            "fair_value": 0.60,
            "relative": 0.40
        }
    )

    risk = calculate_risk(
        m
    )

    investment = weighted_available(
        {
            "kpax":
                kpax,

            "kpax_fv":
                kpax_fv,

            "risk":
                risk
        },
        score_weights
    )

    technical = (
        calculate_technical_score(m)
    )

    return {

        "quality":
            quality,

        "future":
            future,

        "kpax":
            kpax,

        "relative_valuation":
            relative_valuation,

        "kpax_fv":
            kpax_fv,

        "risk":
            risk,

        "investment_score":
            investment,

        "technical":
            technical
    }


# ============================================================
# VERIFY SCORE
# ============================================================

def calculate_verify(m):

    """
    Unabhängiger fundamentaler Plausibilitätscheck.

    NICHT verwendet:
    - Fair Value
    - Analyst Targets
    - RSI
    - MACD
    - KPAX
    - KPAX-FV

    Dadurch kann Verify den Hauptscore tatsächlich
    hinterfragen.
    """

    financial = (
        m["sector"]
        == "Financial Services"
    )

    # --------------------------------------------------------
    # Growth
    # --------------------------------------------------------

    earnings_score = score_growth(
        m["earnings_growth"]
    )

    revenue_score = score_growth(
        m["revenue_growth"]
    )

    growth_score = weighted_available(
        {
            "earnings":
                earnings_score,

            "revenue":
                revenue_score
        },
        {
            "earnings": 0.625,
            "revenue": 0.375
        }
    )

    # entspricht insgesamt:
    # Earnings 25 %
    # Revenue 15 %

    # --------------------------------------------------------
    # Profitability
    # --------------------------------------------------------

    if financial:

        profitability = weighted_available(
            {
                "roe":
                    score_roe(
                        m["roe"]
                    ),

                "margin":
                    score_margin(
                        m["net_margin"]
                    )
            },
            {
                "roe": 0.60,
                "margin": 0.40
            }
        )

    else:

        profitability = weighted_available(
            {
                "roic":
                    score_roic(
                        m["roic"]
                    ),

                "fcf":
                    score_margin(
                        m["fcf_margin"]
                    ),

                "gross":
                    score_margin(
                        m["gross_margin"]
                    )
            },
            {
                "roic": 0.50,
                "fcf": 0.30,
                "gross": 0.20
            }
        )

    # --------------------------------------------------------
    # Debt
    # --------------------------------------------------------

    if financial:

        debt_score = weighted_available(
            {
                "debt_equity":
                    score_debt_equity(
                        m["debt_equity"]
                    ),

                "cash_debt":
                    score_cash_debt(
                        m["cash_to_debt"]
                    )
            },
            {
                "debt_equity": 0.60,
                "cash_debt": 0.40
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
    # Valuation
    # --------------------------------------------------------

    if financial:

        valuation = weighted_available(
            {
                "forward_pe":
                    score_forward_pe(
                        m["forward_pe"]
                    ),

                "pb":
                    score_pb(
                        m["pb"]
                    )
            },
            {
                "forward_pe": 0.60,
                "pb": 0.40
            }
        )

    else:

        valuation = weighted_available(
            {
                "forward_pe":
                    score_forward_pe(
                        m["forward_pe"]
                    ),

                "peg":
                    score_peg(
                        m["peg"]
                    ),

                "ps":
                    score_ps(
                        m["ps"]
                    )
            },
            {
                "forward_pe": 0.50,
                "peg": 0.30,
                "ps": 0.20
            }
        )

    # --------------------------------------------------------
    # Final Verify
    # --------------------------------------------------------

    components = {

        "earnings":
            earnings_score,

        "revenue":
            revenue_score,

        "profitability":
            profitability,

        "debt":
            debt_score,

        "valuation":
            valuation
    }

    weights = {

        "earnings": 0.25,
        "revenue": 0.15,
        "profitability": 0.25,
        "debt": 0.15,
        "valuation": 0.20
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

        "verify":
            verify,

        "verify_data":
            verify_data,

        "verify_earnings":
            earnings_score,

        "verify_revenue":
            revenue_score,

        "verify_growth":
            growth_score,

        "verify_profitability":
            profitability,

        "verify_debt":
            debt_score,

        "verify_valuation":
            valuation
    }


# ============================================================
# VERIFY EXPLANATION
# ============================================================

def explain_verify_difference(row):

    investment = row[
        "investment_score"
    ]

    verify = row["verify"]

    if (
        np.isnan(investment)
        or np.isnan(verify)
    ):
        return "Keine ausreichende Datenbasis"

    gap = (
        investment - verify
    )

    components = {

        "Growth":
            row["verify_growth"],

        "Profitability":
            row["verify_profitability"],

        "Debt":
            row["verify_debt"],

        "Valuation":
            row["verify_valuation"]
    }

    valid = {
        k: v
        for k, v in components.items()
        if v is not None
        and np.isfinite(v)
    }

    if not valid:
        return "Keine ausreichende Datenbasis"

    weakest = min(
        valid,
        key=valid.get
    )

    strongest = max(
        valid,
        key=valid.get
    )

    if gap < -20:

        return (
            f"Hauptscore deutlich niedriger. "
            f"Verify sieht besonders {strongest} "
            f"stark; Hauptmodell wird vermutlich "
            f"durch KPAX/KPAX-FV belastet."
        )

    if gap < -10:

        return (
            f"Verify moderat höher. "
            f"{strongest} stützt die Aktie."
        )

    if gap > 20:

        return (
            f"Hauptscore deutlich höher. "
            f"Verify sieht besonders {weakest} "
            f"schwach."
        )

    if gap > 10:

        return (
            f"Verify moderat niedriger. "
            f"{weakest} belastet die Aktie."
        )

    return "Scores grundsätzlich konsistent"


# ============================================================
# MODEL CHECK
# ============================================================

def model_check(row):

    investment = row[
        "investment_score"
    ]

    verify = row[
        "verify"
    ]

    if (
        np.isnan(investment)
        or np.isnan(verify)
    ):
        return "Nicht prüfbar"

    gap = (
        investment - verify
    )

    if abs(gap) <= 10:
        return "🟢 Modelle konsistent"

    if gap < -20:

        # Hauptscore niedriger
        if (
            row["kpax_fv"]
            < row["kpax"]
            - 10
        ):
            return (
                "🔴 Hauptmodell: "
                "KPAX-FV belastet"
            )

        if (
            row["kpax"]
            < row["kpax_fv"]
            - 10
        ):
            return (
                "🔴 Hauptmodell: "
                "KPAX belastet"
            )

        return (
            "🟠 Hauptscore deutlich "
            "niedriger"
        )

    if gap > 20:

        if (
            row["kpax_fv"]
            > row["kpax"]
            + 10
        ):
            return (
                "🟠 Hauptmodell: "
                "KPAX-FV stützt"
            )

        return (
            "🟠 Hauptscore deutlich "
            "höher"
        )

    return "🟡 Moderate Abweichung"


# ============================================================
# RECOMMENDATION
# ============================================================

def get_recommendation(
    investment,
    kpax,
    kpax_fv,
    technical
):

    if np.isnan(investment):
        return "—"

    if np.isnan(kpax):
        kpax = investment

    if np.isnan(kpax_fv):
        kpax_fv = investment

    if np.isnan(technical):
        technical = 50

    if (
        investment >= 85
        and kpax >= 80
        and kpax_fv >= 80
    ):

        recommendation = (
            "Strong Buy"
        )

    elif (
        investment >= 80
        and kpax >= 75
    ):

        recommendation = "Buy"

    elif investment >= 75:

        recommendation = (
            "Accumulate"
        )

    elif investment >= 68:

        recommendation = "Hold"

    elif investment >= 55:

        recommendation = (
            "Reduce / Watch"
        )

    else:

        recommendation = "Avoid"

    # --------------------------------------------------------
    # Technical downgrade
    # --------------------------------------------------------

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


def get_timing(
    technical
):

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

    earnings = m[
        "earnings_growth"
    ]

    revenue = m[
        "revenue_growth"
    ]

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
    verify_data
):

    if (
        np.isnan(verify_data)
        or verify_data < 60
    ):
        return (
            "⚪ Datenbasis schwach"
        )

    if np.isnan(gap):
        return "⚪ Nicht prüfbar"

    absolute_gap = abs(gap)

    if absolute_gap <= 10:
        return "🟢 Plausibel"

    if absolute_gap <= 20:
        return "🟡 Abweichung"

    return "🔴 Stark abweichend"


# ============================================================
# SIDEBAR
# ============================================================

st.title(
    "📊 Quant-Aktien-Screener V20.2"
)

st.caption(
    "KPAX entscheidet – Verify hinterfragt KPAX."
)

st.sidebar.header(
    "⚙️ Einstellungen"
)

ticker_input = st.sidebar.text_area(
    "Aktien / Ticker",
    value=DEFAULT_TICKERS,
    height=180
)

min_investment_score = (
    st.sidebar.number_input(
        "Minimaler Investment Score",
        min_value=0,
        max_value=100,
        value=0,
        step=1
    )
)


# ============================================================
# WEIGHTS
# ============================================================

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


kpax_weight = (
    st.sidebar.number_input(
        "KPAX %",
        min_value=0.0,
        max_value=100.0,
        value=(
            st.session_state
            .weights["kpax"]
            * 100
        ),
        step=5.0
    )
)

kpax_fv_weight = (
    st.sidebar.number_input(
        "KPAX-FV %",
        min_value=0.0,
        max_value=100.0,
        value=(
            st.session_state
            .weights["kpax_fv"]
            * 100
        ),
        step=5.0
    )
)

risk_weight = (
    st.sidebar.number_input(
        "Risk %",
        min_value=0.0,
        max_value=100.0,
        value=(
            st.session_state
            .weights["risk"]
            * 100
        ),
        step=5.0
    )
)

weight_total = (
    kpax_weight
    + kpax_fv_weight
    + risk_weight
)

if weight_total <= 0:

    st.sidebar.error(
        "Mindestens ein Gewicht > 0."
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
    "Normalisiert: "
    f"{score_weights['kpax']:.0%} / "
    f"{score_weights['kpax_fv']:.0%} / "
    f"{score_weights['risk']:.0%}"
)


# ============================================================
# PARSE TICKERS
# ============================================================

tickers = [
    x.strip().upper()
    for x in ticker_input
    .replace("\n", ",")
    .split(",")
    if x.strip()
]

tickers = list(
    dict.fromkeys(tickers)
)

if not tickers:

    st.warning(
        "Keine Aktien angegeben."
    )

    st.stop()


# ============================================================
# ANALYSIS
# ============================================================

rows = []

progress = st.progress(0)

for i, ticker in enumerate(
    tickers
):

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

            "sector":
                FALLBACK_SECTORS.get(
                    ticker,
                    "Unknown"
                ),

            "investment_score":
                np.nan,

            "verify":
                np.nan,

            "verify_data":
                0,

            "completeness":
                0
        })

    progress.progress(
        (i + 1)
        / len(tickers)
    )

progress.empty()


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    rows
)

if df.empty:

    st.error(
        "Keine Daten."
    )

    st.stop()


# ============================================================
# RANKING
# ============================================================

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
    lambda r:
        get_verify_status(
            r["verify_gap"],
            r["verify_data"]
        ),
    axis=1
)

df["verify_reason"] = df.apply(
    explain_verify_difference,
    axis=1
)

df["model_check"] = df.apply(
    model_check,
    axis=1
)


# ============================================================
# RECOMMENDATIONS
# ============================================================

df["recommendation"] = df.apply(
    lambda r:
        get_recommendation(
            r.get(
                "investment_score",
                np.nan
            ),
            r.get(
                "kpax",
                np.nan
            ),
            r.get(
                "kpax_fv",
                np.nan
            ),
            r.get(
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
        display_df[
            "investment_score"
        ] >= min_investment_score
    ]

display_df = display_df.sort_values(
    "investment_score",
    ascending=False
)


# ============================================================
# MAIN TABLE
# ============================================================

st.subheader(
    f"📈 Investment Ranking "
    f"({len(display_df)} / {len(df)})"
)

main_table = pd.DataFrame({

    "Ticker":
        display_df["ticker"],

    "Name":
        display_df["name"],

    "Sektor":
        display_df["sector"],

    "Investment":
        display_df[
            "investment_score"
        ],

    "KPAX":
        display_df["kpax"],

    "KPAX-FV":
        display_df["kpax_fv"],

    "Quality":
        display_df["quality"],

    "Future":
        display_df["future"],

    "Fair Value":
        display_df[
            "fair_value_score"
        ],

    "Relative Val.":
        display_df[
            "relative_valuation"
        ],

    "Risk":
        display_df["risk"],

    "Technical":
        display_df["technical"],

    "Empfehlung":
        display_df[
            "recommendation"
        ],

    "Timing":
        display_df["timing"],

    "Turnaround":
        display_df["turnaround"],

    "Forward-KGV":
        display_df["forward_pe"],

    "Daten":
        display_df["completeness"]
})


st.dataframe(
    main_table.style.format(
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
            "Forward-KGV": "{:.1f}",
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
    "🔎 Verify – Fundamentaler Plausibilitätscheck"
)

st.caption(
    "Der Verify Score ist unabhängig vom Investment Score. "
    "Er prüft nur, ob Wachstum, Profitabilität, Verschuldung "
    "und Bewertung den Hauptscore grundsätzlich plausibel machen."
)

verify_table = pd.DataFrame({

    "Ticker":
        display_df["ticker"],

    "Investment":
        display_df[
            "investment_score"
        ],

    "Verify":
        display_df["verify"],

    "Gap":
        display_df["verify_gap"],

    "Verify Growth":
        display_df["verify_growth"],

    "Profitability":
        display_df[
            "verify_profitability"
        ],

    "Debt":
        display_df["verify_debt"],

    "Valuation":
        display_df[
            "verify_valuation"
        ],

    "Inv. Rang":
        display_df[
            "investment_rank"
        ],

    "Verify Rang":
        display_df[
            "verify_rank"
        ],

    "Rang-Diff":
        display_df["rank_diff"],

    "Plausibilität":
        display_df[
            "verify_status"
        ]
})


st.dataframe(
    verify_table.style.format(
        {
            "Investment": "{:.0f}",
            "Verify": "{:.0f}",
            "Gap": "{:+.0f}",
            "Verify Growth": "{:.0f}",
            "Profitability": "{:.0f}",
            "Debt": "{:.0f}",
            "Valuation": "{:.0f}",
            "Inv. Rang": "{:.0f}",
            "Verify Rang": "{:.0f}",
            "Rang-Diff": "{:+.0f}"
        },
        na_rep="—"
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# VERIFY DIAGNOSIS
# ============================================================

st.subheader(
    "🧪 Verify-Diagnose"
)

diagnosis_table = pd.DataFrame({

    "Ticker":
        display_df["ticker"],

    "Investment":
        display_df[
            "investment_score"
        ],

    "Verify":
        display_df["verify"],

    "Gap":
        display_df["verify_gap"],

    "Hauptmodell":
        display_df["model_check"],

    "Grund":
        display_df["verify_reason"],

    "Verify-Daten":
        display_df["verify_data"]
})


st.dataframe(
    diagnosis_table.style.format(
        {
            "Investment": "{:.0f}",
            "Verify": "{:.0f}",
            "Gap": "{:+.0f}",
            "Verify-Daten": "{:.0f}%"
        },
        na_rep="—"
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# VERIFY MATH
# ============================================================

with st.expander(
    "🔎 Verify-Mathematik"
):

    st.markdown(
        """
## Zweck

Der Verify Score ist **kein zusätzlicher Investment Score**.

Er ist ein unabhängiger Plausibilitätscheck.

### Gewichtung

| Bereich | Gewicht |
|---|---:|
| Earnings Growth | 25 % |
| Revenue Growth | 15 % |
| Profitability | 25 % |
| Debt | 15 % |
| Valuation | 20 % |

---

### Growth

`Earnings Growth Score = f(Earnings Growth)`

`Revenue Growth Score = f(Revenue Growth)`

---

### Profitability

Normale Unternehmen:

`Profitability = 50 % ROIC + 30 % FCF-Marge + 20 % Gross Margin`

Financials:

`Profitability = 60 % ROE + 40 % Net Margin`

---

### Debt

Normale Unternehmen:

`Debt = 60 % Net Debt / EBITDA + 25 % Debt/Equity + 15 % Current Ratio`

Financials:

`Debt = 60 % Debt/Equity + 40 % Cash/Debt`

---

### Valuation

Normale Unternehmen:

`Valuation = 50 % Forward-KGV + 30 % PEG + 20 % KUV`

Financials:

`Valuation = 60 % Forward-KGV + 40 % KBV`

---

## Gap

`Verify Gap = Investment Score − Verify Score`

### Interpretation

**🟢 Plausibel**

`|Gap| ≤ 10`

Die beiden Modelle sind grundsätzlich konsistent.

**🟡 Abweichung**

`10 < |Gap| ≤ 20`

Es gibt eine relevante Differenz.

**🔴 Stark abweichend**

`|Gap| > 20`

Die beiden Modelle bewerten die Aktie deutlich unterschiedlich.

Das ist **kein automatisches Kaufsignal oder Verkaufssignal**.

Es bedeutet:

> Hier sollte die Berechnungslogik bzw. die Aktie genauer untersucht werden.

---

## Rang-Differenz

`Rang-Diff = Investment Rang − Verify Rang`

Der Rang ist nur eine Zusatzinformation.

Ein großer Rangunterschied kann interessant sein,
ist aber weniger aussagekräftig als die tatsächliche
Score-Differenz.

---

## Datenbasis

Der Verify Score wird nur dann als belastbar betrachtet,
wenn mindestens 60 % der Verify-Komponenten verfügbar sind.

Bei schwacher Datenbasis:

`⚪ Datenbasis schwach`

Dadurch werden beispielsweise SMO oder F8P nicht
künstlich mit einem schlechten Score versehen.
"""
    )


# ============================================================
# MAIN MODEL MATH
# ============================================================

with st.expander(
    "📐 Mathematik des Hauptmodells"
):

    st.markdown(
        """
## KPAX

`KPAX = 40 % Quality + 60 % Future`

---

## Quality

### Normale Unternehmen

`Quality = 30 % ROIC + 15 % Gross Margin + 20 % FCF Margin + 20 % Earnings Growth + 10 % Revenue Growth`

### Financials

`Quality = 40 % ROE + 30 % Net Margin + 20 % Earnings Growth + 10 % Revenue Growth`

Fehlende Werte werden herausgenommen und die
verbleibenden Gewichte automatisch normalisiert.

---

## Future

`Future = 30 % Earnings Growth + 20 % Revenue Growth + 15 % Long-Term Growth + 15 % Analyst Upside + 10 % FCF Margin + 10 % Growth Acceleration`

---

## Relative Valuation

Normale Unternehmen:

`40 % Forward-KGV + 20 % Trailing-KGV + 20 % PEG + 20 % KUV`

Financials:

`40 % Forward-KGV + 20 % Trailing-KGV + 20 % PEG + 20 % KBV`

---

## Fair Value

Normale Unternehmen:

`50 % Earnings FV + 30 % FCF FV + 20 % Analyst Target`

Nicht verfügbare Modelle werden automatisch
herausgenommen und die übrigen Gewichte normalisiert.

### Earnings Fair Value

`Fair Value = Forward EPS × Fair P/E`

`Fair P/E = 10 + 0,40 × Growth`

mit:

`10 ≤ Fair P/E ≤ 28`

### FCF Fair Value

`FCF Fair Value = FCF je Aktie / Target FCF Yield`

`Target FCF Yield = 0,08 − 0,0015 × Growth`

mit Begrenzung auf 3,5–10 %.

---

## KPAX-FV

`KPAX-FV = 60 % Fair Value Score + 40 % Relative Valuation`

---

## Investment Score

Standard:

`40 % KPAX + 50 % KPAX-FV + 10 % Risk`

Die Gewichtung kann in der Sidebar verändert werden.

---

## Technical Score

`25 % SMA200`

`25 % 6M Performance`

`20 % RSI`

`20 % MACD`

`10 % SMA50`

Der Technical Score fließt **nicht** in den
Investment Score ein.

Er dient als Timing- und Empfehlungsfilter.

---

## Scorebereich

Ein gültiger Score liegt zwischen:

`5 und 100`

Dabei bedeutet:

**5 = sehr schwaches reales Ergebnis**

während:

**— = keine ausreichende Datenbasis**

"""
    )


# ============================================================
# SUMMARY
# ============================================================

st.markdown("---")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Aktien",
        len(df)
    )

with c2:

    st.metric(
        "Investment Scores",
        int(
            df[
                "investment_score"
            ].notna().sum()
        )
    )

with c3:

    st.metric(
        "Verify Scores",
        int(
            df[
                "verify"
            ].notna().sum()
        )
    )

with c4:

    st.metric(
        "Stark abweichend",
        int(
            (
                df[
                    "verify_status"
                ]
                == "🔴 Stark abweichend"
            ).sum()
        )
    )
