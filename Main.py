import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aktien-Screener V20.0",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# FALLBACK SECTORS
# ============================================================

FALLBACK_SECTORS = {
    'NVDA': 'Technology',
    'MSFT': 'Technology',
    'AAPL': 'Technology',
    'GOOGL': 'Technology',
    'AMZN': 'Consumer Cyclical',
    'ASML': 'Technology',
    'TSM': 'Technology',
    'BMW.DE': 'Consumer Cyclical',
    'TTE.PA': 'Energy',
    'ING': 'Financial Services',
    'PFE': 'Healthcare',
    'KO': 'Consumer Defensive',
    'NKE': 'Consumer Cyclical',
    'MRVL': 'Technology',
    'MU': 'Technology',
    '000660.KS': 'Technology'
}


# ============================================================
# FINANCIAL STATEMENT ALIASES
# ============================================================

REVENUE_KEYS = [
    'Total Revenue',
    'Operating Revenue',
    'Revenue',
    'TotalRevenue'
]

NET_INCOME_KEYS = [
    'Net Income',
    'Net Income Common Stockholders',
    'Net Income Including Noncontrolling Interests',
    'Net Income From Continuing Operation Net Minority Interest',
    'Net Income From Continuing Operation',
    'NetIncome'
]

EBIT_KEYS = [
    'EBIT',
    'Operating Income',
    'OperatingIncome'
]

EBITDA_KEYS = [
    'Normalized EBITDA',
    'EBITDA',
    'NormalizedEBITDA'
]

PRETAX_KEYS = [
    'Pretax Income',
    'Income Before Tax',
    'PretaxIncome'
]

TAX_KEYS = [
    'Tax Provision',
    'Income Tax Expense',
    'TaxProvision'
]

TOTAL_ASSETS_KEYS = [
    'Total Assets',
    'TotalAssets'
]

CURRENT_LIABILITY_KEYS = [
    'Current Liabilities',
    'Total Current Liabilities',
    'CurrentLiabilities'
]

CASH_KEYS = [
    'Cash And Cash Equivalents',
    'Cash Financial',
    'CashAndCashEquivalents',
    'Cash Cash Equivalents And Short Term Investments'
]

DEBT_KEYS = [
    'Total Debt',
    'TotalDebt',
    'Long Term Debt',
    'Long Term Debt And Capital Lease Obligation'
]

OPERATING_CF_KEYS = [
    'Operating Cash Flow',
    'Total Cash From Operating Activities',
    'Cash Flow From Continuing Operating Activities',
    'OperatingCashFlow'
]

CAPEX_KEYS = [
    'Capital Expenditure',
    'Capital Expenditures',
    'CapEx',
    'CapitalExpenditures',
    'Purchase Of Property And Equipment'
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_get(dictionary, key, default=np.nan):

    if not isinstance(dictionary, dict):
        return default

    try:
        value = dictionary.get(key, default)
    except Exception:
        return default

    if value is None:
        return default

    return value


def to_number(value):

    try:

        if value is None:
            return np.nan

        if isinstance(value, (pd.Series, pd.DataFrame)):

            if value.empty:
                return np.nan

            value = value.iloc[0]

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:

        return np.nan


def clean_percentage(value):

    value = to_number(value)

    if pd.isna(value):
        return np.nan

    if abs(value) > 2.0:
        return value / 100.0

    return value


def clean_positive(value):

    value = to_number(value)

    if pd.isna(value) or value <= 0:
        return np.nan

    return value


def get_row(df, possible_keys):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return np.nan

    for key in possible_keys:

        if key not in df.index:
            continue

        try:

            row = df.loc[key]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            row = pd.to_numeric(
                row,
                errors="coerce"
            ).dropna()

            if row.empty:
                continue

            return to_number(
                row.iloc[0]
            )

        except Exception:
            continue

    return np.nan


def get_first_valid_row(df, possible_keys):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return np.nan

    for key in possible_keys:

        if key not in df.index:
            continue

        try:

            row = df.loc[key]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            values = pd.to_numeric(
                row,
                errors="coerce"
            ).dropna()

            if not values.empty:

                return to_number(
                    values.iloc[0]
                )

        except Exception:
            continue

    return np.nan


def get_numeric_series(df, possible_keys):

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return pd.Series(dtype=float)

    for key in possible_keys:

        if key not in df.index:
            continue

        try:

            row = df.loc[key]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            values = pd.to_numeric(
                row,
                errors="coerce"
            ).dropna()

            if values.empty:
                continue

            try:
                values.index = pd.to_datetime(
                    values.index
                )
                values = values.sort_index()
            except Exception:
                pass

            return values.astype(float)

        except Exception:
            continue

    return pd.Series(dtype=float)


def calculate_growth_volatility(series):

    if (
        series is None
        or not isinstance(series, pd.Series)
        or len(series) < 3
    ):
        return np.nan

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(values) < 3:
        return np.nan

    growth_rates = []

    for previous, current in zip(
        values.iloc[:-1],
        values.iloc[1:]
    ):

        if (
            pd.notna(previous)
            and pd.notna(current)
            and previous > 0
            and current > 0
        ):
            growth_rates.append(
                current / previous - 1
            )

    if len(growth_rates) < 2:
        return np.nan

    return float(
        pd.Series(growth_rates).std(ddof=1)
    )


def get_estimate_value(df, row_name, column='avg'):

    if df is None:
        return np.nan

    if not isinstance(df, pd.DataFrame):
        return np.nan

    if df.empty:
        return np.nan

    try:

        if row_name not in df.index:
            return np.nan

        if column not in df.columns:
            return np.nan

        return to_number(
            df.loc[row_name, column]
        )

    except Exception:

        return np.nan


def safe_mean(values):

    values = [
        x for x in values
        if pd.notna(x)
    ]

    if not values:
        return np.nan

    return float(np.mean(values))


def extract_analyst_target(targets, key):

    value = np.nan

    if isinstance(targets, dict):
        value = targets.get(key, np.nan)

    elif isinstance(targets, pd.Series):
        try:
            if key in targets.index:
                value = targets.loc[key]
        except Exception:
            pass

    elif isinstance(targets, pd.DataFrame):
        try:
            if key in targets.index:
                value = targets.loc[key].iloc[0]
            elif key in targets.columns:
                value = targets[key].iloc[0]
        except Exception:
            pass

    return clean_positive(value)


# ============================================================
# COMPANY NAME
# ============================================================

def get_stock_name(info, search_quotes, ticker_symbol):

    for key in [
        'longName',
        'shortName',
        'displayName'
    ]:

        value = safe_get(
            info,
            key,
            None
        )

        if isinstance(value, str):

            value = value.strip()

            if (
                value
                and value.lower() != 'nan'
                and value.upper() != ticker_symbol.upper()
            ):
                return value

    if isinstance(search_quotes, list):

        ticker_upper = ticker_symbol.upper()

        for quote in search_quotes:

            if not isinstance(quote, dict):
                continue

            symbol = str(
                quote.get(
                    'symbol',
                    ''
                )
            ).upper()

            if symbol != ticker_upper:
                continue

            for key in [
                'longname',
                'longName',
                'shortname',
                'shortName',
                'displayName'
            ]:

                value = quote.get(key)

                if (
                    isinstance(value, str)
                    and value.strip()
                    and value.strip().upper()
                    != ticker_upper
                ):
                    return value.strip()

        for quote in search_quotes:

            if not isinstance(quote, dict):
                continue

            for key in [
                'longname',
                'longName',
                'shortname',
                'shortName',
                'displayName'
            ]:

                value = quote.get(key)

                if (
                    isinstance(value, str)
                    and value.strip()
                ):
                    return value.strip()

    return ticker_symbol


# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def fetch_stock_data(ticker_symbol):

    try:

        ticker = yf.Ticker(
            ticker_symbol
        )

        try:
            info = dict(
                ticker.info
            )
        except Exception:
            info = {}

        try:
            fast_info = dict(
                ticker.fast_info
            )
        except Exception:
            fast_info = {}

        search_quotes = []

        try:

            search_result = yf.Search(
                ticker_symbol,
                max_results=5,
                news_count=0,
                lists_count=0,
                include_nav_links=False,
                include_research=False
            )

            search_quotes = search_result.quotes

            if search_quotes is None:
                search_quotes = []

        except Exception:

            search_quotes = []

        try:

            analyst_targets = dict(
                ticker.get_analyst_price_targets()
            )

        except Exception:

            try:
                analyst_targets = dict(
                    ticker.analyst_price_targets
                )
            except Exception:
                analyst_targets = {}

        try:

            earnings_estimate = (
                ticker.get_earnings_estimate()
            )

        except Exception:

            try:
                earnings_estimate = (
                    ticker.earnings_estimate
                )
            except Exception:
                earnings_estimate = pd.DataFrame()

        try:

            revenue_estimate = (
                ticker.get_revenue_estimate()
            )

        except Exception:

            try:
                revenue_estimate = (
                    ticker.revenue_estimate
                )
            except Exception:
                revenue_estimate = pd.DataFrame()

        try:

            growth_estimates = (
                ticker.get_growth_estimates()
            )

        except Exception:

            try:
                growth_estimates = (
                    ticker.growth_estimates
                )
            except Exception:
                growth_estimates = pd.DataFrame()

        try:

            hist = ticker.history(
                period="2y",
                auto_adjust=False
            )

        except Exception:

            hist = pd.DataFrame()

        if (
            hist.empty
            or len(hist) < 30
        ):
            return None

        try:

            financials = ticker.get_income_stmt(
                freq="yearly"
            )

        except Exception:

            try:
                financials = ticker.financials
            except Exception:
                financials = pd.DataFrame()

        try:

            balance_sheet = ticker.get_balance_sheet(
                freq="yearly"
            )

        except Exception:

            try:
                balance_sheet = ticker.balance_sheet
            except Exception:
                balance_sheet = pd.DataFrame()

        try:

            cashflow = ticker.get_cashflow(
                freq="yearly"
            )

        except Exception:

            try:
                cashflow = ticker.cashflow
            except Exception:
                cashflow = pd.DataFrame()

        return {

            'ticker': ticker_symbol,
            'info': info,
            'fast_info': fast_info,
            'search_quotes': search_quotes,
            'analyst_targets': analyst_targets,
            'earnings_estimate': earnings_estimate,
            'revenue_estimate': revenue_estimate,
            'growth_estimates': growth_estimates,
            'hist': hist,
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cashflow': cashflow
        }

    except Exception:

        return None


# ============================================================
# MARKET DATA
# ============================================================

def get_market_cap(info, fast_info):

    market_cap = to_number(
        safe_get(
            info,
            'marketCap'
        )
    )

    if (
        pd.notna(market_cap)
        and market_cap > 0
    ):
        return market_cap

    for key in [
        'market_cap',
        'marketCap'
    ]:

        value = to_number(
            safe_get(
                fast_info,
                key
            )
        )

        if (
            pd.notna(value)
            and value > 0
        ):
            return value

    return np.nan


def get_current_price(
    info,
    fast_info,
    hist
):

    for key in [
        'currentPrice',
        'regularMarketPrice',
        'previousClose'
    ]:

        value = to_number(
            safe_get(
                info,
                key
            )
        )

        if (
            pd.notna(value)
            and value > 0
        ):
            return value

    for key in [
        'last_price',
        'regularMarketPrice'
    ]:

        value = to_number(
            safe_get(
                fast_info,
                key
            )
        )

        if (
            pd.notna(value)
            and value > 0
        ):
            return value

    if (
        hist is not None
        and not hist.empty
    ):

        try:

            value = to_number(
                hist['Close'].iloc[-1]
            )

            if (
                pd.notna(value)
                and value > 0
            ):
                return value

        except Exception:
            pass

    return np.nan


def get_shares_outstanding(
    info,
    fast_info,
    market_cap,
    current_price
):

    for key in [
        'sharesOutstanding',
        'impliedSharesOutstanding'
    ]:

        value = to_number(
            safe_get(
                info,
                key
            )
        )

        if (
            pd.notna(value)
            and value > 0
        ):
            return value

    for key in [
        'shares',
        'sharesOutstanding'
    ]:

        value = to_number(
            safe_get(
                fast_info,
                key
            )
        )

        if (
            pd.notna(value)
            and value > 0
        ):
            return value

    if (
        pd.notna(market_cap)
        and market_cap > 0
        and pd.notna(current_price)
        and current_price > 0
    ):

        shares = (
            market_cap
            / current_price
        )

        if (
            pd.notna(shares)
            and shares > 0
        ):
            return shares

    return np.nan


# ============================================================
# SECTOR
# ============================================================

def get_sector(
    ticker_symbol,
    info
):

    sector = safe_get(
        info,
        'sector',
        None
    )

    if (
        isinstance(sector, str)
        and sector.strip()
    ):
        return sector

    return FALLBACK_SECTORS.get(
        ticker_symbol.upper(),
        'Default'
    )


# ============================================================
# VALUATION HELPERS
# ============================================================

def calculate_trailing_pe(
    info,
    market_cap,
    net_income,
    current_price
):

    trailing_pe = to_number(
        safe_get(
            info,
            'trailingPE'
        )
    )

    if (
        pd.notna(trailing_pe)
        and trailing_pe > 0
    ):
        return trailing_pe

    trailing_eps = to_number(
        safe_get(
            info,
            'trailingEps'
        )
    )

    if (
        pd.notna(current_price)
        and current_price > 0
        and pd.notna(trailing_eps)
        and trailing_eps > 0
    ):

        return (
            current_price
            / trailing_eps
        )

    if (
        pd.notna(market_cap)
        and market_cap > 0
        and pd.notna(net_income)
        and net_income > 0
    ):

        return (
            market_cap
            / net_income
        )

    return np.nan


def calculate_forward_pe(info):

    value = to_number(
        safe_get(
            info,
            'forwardPE'
        )
    )

    if (
        pd.notna(value)
        and value > 0
    ):
        return value

    return np.nan


def calculate_ps(
    info,
    market_cap,
    revenue
):

    ps = to_number(
        safe_get(
            info,
            'priceToSalesTrailing12Months'
        )
    )

    if (
        pd.notna(ps)
        and ps > 0
    ):
        return ps

    if (
        pd.notna(market_cap)
        and market_cap > 0
        and pd.notna(revenue)
        and revenue > 0
    ):

        return (
            market_cap
            / revenue
        )

    return np.nan


def calculate_pb(
    info,
    market_cap,
    equity
):

    pb = to_number(
        safe_get(
            info,
            'priceToBook'
        )
    )

    if (
        pd.notna(pb)
        and pb > 0
    ):
        return pb

    if (
        pd.notna(market_cap)
        and market_cap > 0
        and pd.notna(equity)
        and equity > 0
    ):

        return (
            market_cap
            / equity
        )

    return np.nan


# ============================================================
# FREE CASH FLOW
# ============================================================

def get_free_cash_flow(cashflow):

    if (
        cashflow is None
        or not isinstance(cashflow, pd.DataFrame)
        or cashflow.empty
    ):
        return np.nan

    direct_statement_fcf = get_first_valid_row(
        cashflow,
        [
            'Free Cash Flow',
            'FreeCashFlow',
            'Free Cashflow',
            'FreeCashflow'
        ]
    )

    if pd.notna(direct_statement_fcf):
        return direct_statement_fcf

    operating_cf = get_first_valid_row(
        cashflow,
        [
            'Operating Cash Flow',
            'Total Cash From Operating Activities',
            'Cash Flow From Continuing Operating Activities',
            'OperatingCashFlow'
        ]
    )

    capex = get_first_valid_row(
        cashflow,
        [
            'Capital Expenditure',
            'Capital Expenditures',
            'CapitalExpenditures',
            'CapEx',
            'Purchase Of Property And Equipment'
        ]
    )

    if (
        pd.notna(operating_cf)
        and pd.notna(capex)
    ):

        if capex < 0:
            return operating_cf + capex

        return operating_cf - capex

    return np.nan


# ============================================================
# FAIR VALUE MODEL
# ============================================================

def calculate_fcf_fair_value(
    fcf,
    market_cap,
    current_price,
    growth_rate
):

    if (
        pd.isna(fcf)
        or fcf <= 0
        or pd.isna(market_cap)
        or market_cap <= 0
        or pd.isna(current_price)
        or current_price <= 0
    ):
        return np.nan

    fcf_yield = (
        fcf / market_cap
    )

    if (
        pd.isna(fcf_yield)
        or fcf_yield <= 0
        or fcf_yield > 0.50
    ):
        return np.nan

    fcf_per_share = (
        fcf_yield
        * current_price
    )

    if (
        pd.isna(fcf_per_share)
        or fcf_per_share <= 0
    ):
        return np.nan

    if pd.isna(growth_rate):
        growth_rate = 0.05

    growth_rate = np.clip(
        growth_rate,
        -0.03,
        0.12
    )

    discount_rate = 0.09
    terminal_growth = 0.025

    if terminal_growth >= discount_rate:
        return np.nan

    pv_fcf = 0.0
    projected_fcf = fcf_per_share

    for year in range(1, 6):

        year_growth = (
            growth_rate
            + (
                terminal_growth
                - growth_rate
            )
            * (year - 1)
            / 4
        )

        projected_fcf *= (
            1 + year_growth
        )

        pv_fcf += (
            projected_fcf
            / (
                (1 + discount_rate)
                ** year
            )
        )

    terminal_value = (
        projected_fcf
        * (1 + terminal_growth)
        / (
            discount_rate
            - terminal_growth
        )
    )

    pv_terminal = (
        terminal_value
        / (
            (1 + discount_rate)
            ** 5
        )
    )

    fair_value = (
        pv_fcf
        + pv_terminal
    )

    if (
        pd.isna(fair_value)
        or fair_value <= 0
        or fair_value > current_price * 10
        or fair_value < current_price * 0.05
    ):
        return np.nan

    dcf_lower_bound = current_price * 0.60
    dcf_upper_bound = current_price * 1.75

    if (
        fair_value < dcf_lower_bound
        or fair_value > dcf_upper_bound
    ):
        return np.nan

    return fair_value


def calculate_fair_value_score(
    current_price,
    analyst_target,
    fcf_fair_value
):

    fair_values = []

    if (
        pd.notna(analyst_target)
        and analyst_target > 0
    ):
        fair_values.append(
            ('analyst', analyst_target)
        )

    if (
        pd.notna(fcf_fair_value)
        and fcf_fair_value > 0
    ):
        fair_values.append(
            ('fcf', fcf_fair_value)
        )

    if not fair_values:
        return np.nan, np.nan

    if len(fair_values) == 2:

        analyst_value = next(
            value
            for source, value in fair_values
            if source == 'analyst'
        )

        fcf_value = next(
            value
            for source, value in fair_values
            if source == 'fcf'
        )

        fair_value_price = (
            0.75 * analyst_value
            + 0.25 * fcf_value
        )

    else:

        fair_value_price = fair_values[0][1]

    if (
        pd.isna(current_price)
        or current_price <= 0
    ):
        return fair_value_price, np.nan

    upside = (
        fair_value_price
        / current_price
        - 1
    )

    fair_value_score = np.interp(
        upside,
        [
            -0.40,
            -0.20,
            0.00,
            0.10,
            0.25,
            0.50,
            0.75
        ],
        [
            0,
            15,
            40,
            58,
            75,
            90,
            100
        ]
    )

    return (
        fair_value_price,
        fair_value_score
    )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(close, period=14):

    if close is None or len(close) < period + 2:
        return np.nan

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(period)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(period)
        .mean()
    )

    latest_gain = avg_gain.iloc[-1]
    latest_loss = avg_loss.iloc[-1]

    if pd.isna(latest_gain) or pd.isna(latest_loss):
        return np.nan

    if latest_loss == 0:

        if latest_gain > 0:
            return 100.0

        return 50.0

    rs = (
        latest_gain
        / latest_loss
    )

    return (
        100
        - 100 / (1 + rs)
    )


def calculate_macd(close):

    if close is None or len(close) < 35:
        return (
            np.nan,
            np.nan,
            np.nan
        )

    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    macd = (
        ema12
        - ema26
    )

    signal = (
        macd
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    histogram = (
        macd
        - signal
    )

    return (
        macd.iloc[-1],
        signal.iloc[-1],
        histogram.iloc[-1]
    )


# ============================================================
# METRIC EXTRACTION
# ============================================================

def extract_metrics(data):

    info = data['info']
    fast_info = data['fast_info']
    search_quotes = data['search_quotes']

    analyst_targets = data['analyst_targets']
    earnings_estimate = data['earnings_estimate']
    revenue_estimate = data['revenue_estimate']
    growth_estimates = data['growth_estimates']

    hist = data['hist']
    financials = data['financials']
    balance_sheet = data['balance_sheet']
    cashflow = data['cashflow']

    ticker_symbol = data['ticker']

    market_cap = get_market_cap(
        info,
        fast_info
    )

    current_price = get_current_price(
        info,
        fast_info,
        hist
    )

    shares_outstanding = get_shares_outstanding(
        info,
        fast_info,
        market_cap,
        current_price
    )

    sector = get_sector(
        ticker_symbol,
        info
    )

    name = get_stock_name(
        info,
        search_quotes,
        ticker_symbol
    )

    revenue = get_row(
        financials,
        REVENUE_KEYS
    )

    net_income = get_row(
        financials,
        NET_INCOME_KEYS
    )

    ebit = get_row(
        financials,
        EBIT_KEYS
    )

    ebitda = get_row(
        financials,
        EBITDA_KEYS
    )

    pretax_income = get_row(
        financials,
        PRETAX_KEYS
    )

    tax_provision = get_row(
        financials,
        TAX_KEYS
    )

    total_assets = get_row(
        balance_sheet,
        TOTAL_ASSETS_KEYS
    )

    current_liabilities = get_row(
        balance_sheet,
        CURRENT_LIABILITY_KEYS
    )

    cash = get_row(
        balance_sheet,
        CASH_KEYS
    )

    total_debt = get_row(
        balance_sheet,
        DEBT_KEYS
    )

    equity = get_row(
        balance_sheet,
        [
            'Stockholders Equity',
            'Total Equity Gross Minority Interest',
            'Common Stock Equity',
            'StockholdersEquity'
        ]
    )

    current_assets = get_row(
        balance_sheet,
        [
            'Current Assets',
            'Total Current Assets',
            'CurrentAssets'
        ]
    )

    trailing_eps = to_number(
        safe_get(
            info,
            'trailingEps'
        )
    )

    forward_eps = to_number(
        safe_get(
            info,
            'forwardEps'
        )
    )

    earnings_growth = clean_percentage(
        safe_get(
            info,
            'earningsGrowth'
        )
    )

    revenue_growth = clean_percentage(
        safe_get(
            info,
            'revenueGrowth'
        )
    )

    estimate_growth = get_estimate_value(
        earnings_estimate,
        '+1y',
        'growth'
    )

    if pd.notna(estimate_growth):
        estimate_growth = clean_percentage(
            estimate_growth
        )

    if pd.isna(earnings_growth):
        earnings_growth = estimate_growth

    estimate_revenue_growth = get_estimate_value(
        revenue_estimate,
        '+1y',
        'growth'
    )

    if pd.notna(estimate_revenue_growth):
        estimate_revenue_growth = clean_percentage(
            estimate_revenue_growth
        )

    if pd.isna(revenue_growth):
        revenue_growth = estimate_revenue_growth

    long_term_growth = np.nan

    try:

        if (
            isinstance(
                growth_estimates,
                pd.DataFrame
            )
            and not growth_estimates.empty
            and '+5y' in growth_estimates.index
            and 'stock' in growth_estimates.columns
        ):

            long_term_growth = clean_percentage(
                growth_estimates.loc[
                    '+5y',
                    'stock'
                ]
            )

    except Exception:
        pass

    if pd.isna(long_term_growth):
        long_term_growth = earnings_growth

    trailing_pe = calculate_trailing_pe(
        info,
        market_cap,
        net_income,
        current_price
    )

    forward_pe = calculate_forward_pe(
        info
    )

    ps_ratio = calculate_ps(
        info,
        market_cap,
        revenue
    )

    pb_ratio = calculate_pb(
        info,
        market_cap,
        equity
    )

    forward_peg = np.nan

    if (
        pd.notna(forward_pe)
        and forward_pe > 0
        and pd.notna(earnings_growth)
        and earnings_growth > 0
    ):

        growth_percent = (
            earnings_growth * 100
        )

        if growth_percent > 0:

            forward_peg = (
                forward_pe
                / growth_percent
            )

    if pd.isna(forward_peg):

        yahoo_peg = clean_positive(
            safe_get(
                info,
                'pegRatio'
            )
        )

        if pd.notna(yahoo_peg):
            forward_peg = yahoo_peg

    gross_margin = clean_percentage(
        safe_get(
            info,
            'grossMargins'
        )
    )

    if pd.isna(gross_margin):

        gross_profit = get_row(
            financials,
            [
                'Gross Profit',
                'GrossProfit'
            ]
        )

        if (
            pd.notna(gross_profit)
            and pd.notna(revenue)
            and revenue > 0
        ):

            gross_margin = (
                gross_profit
                / revenue
            )

    net_margin = np.nan

    if (
        pd.notna(net_income)
        and pd.notna(revenue)
        and revenue > 0
    ):

        net_margin = (
            net_income
            / revenue
        )

    roe = clean_percentage(
        safe_get(
            info,
            'returnOnEquity'
        )
    )

    if pd.isna(roe):

        if (
            pd.notna(net_income)
            and pd.notna(equity)
            and equity > 0
        ):

            roe = (
                net_income
                / equity
            )

    roic = np.nan

    if sector != 'Financial Services':

        if (
            pd.notna(ebit)
            and pd.notna(total_assets)
            and pd.notna(current_liabilities)
            and pd.notna(cash)
        ):

            tax_rate = 0.21

            if (
                pd.notna(tax_provision)
                and pd.notna(pretax_income)
                and pretax_income > 0
            ):

                calculated_tax_rate = (
                    tax_provision
                    / pretax_income
                )

                if (
                    calculated_tax_rate >= 0
                    and calculated_tax_rate <= 0.60
                ):

                    tax_rate = np.clip(
                        calculated_tax_rate,
                        0.10,
                        0.40
                    )

            nopat = (
                ebit
                * (1 - tax_rate)
            )

            invested_capital = (
                total_assets
                - current_liabilities
                - cash
            )

            if invested_capital > 0:

                calculated_roic = (
                    nopat
                    / invested_capital
                )

                if (
                    calculated_roic >= -1
                    and calculated_roic <= 3
                ):

                    roic = calculated_roic

    operating_cf = get_first_valid_row(
        cashflow,
        [
            'Operating Cash Flow',
            'Total Cash From Operating Activities',
            'Cash Flow From Continuing Operating Activities',
            'OperatingCashFlow'
        ]
    )

    capex = get_first_valid_row(
        cashflow,
        [
            'Capital Expenditure',
            'Capital Expenditures',
            'CapitalExpenditures',
            'CapEx',
            'Purchase Of Property And Equipment'
        ]
    )

    fcf = get_free_cash_flow(
        cashflow
    )

    fcf_yield = np.nan
    fcf_margin = np.nan

    if (
        pd.notna(fcf)
        and pd.notna(market_cap)
        and market_cap > 0
    ):

        fcf_yield = (
            fcf
            / market_cap
        )

    if (
        pd.notna(fcf)
        and pd.notna(revenue)
        and revenue > 0
    ):

        fcf_margin = (
            fcf
            / revenue
        )

    net_debt_ebitda = np.nan

    if sector != 'Financial Services':

        if (
            pd.notna(total_debt)
            and pd.notna(cash)
            and pd.notna(ebitda)
            and ebitda > 0
        ):

            net_debt = (
                total_debt
                - cash
            )

            net_debt_ebitda = (
                net_debt
                / ebitda
            )

    debt_to_equity = clean_positive(
        safe_get(
            info,
            'debtToEquity'
        )
    )

    if pd.isna(debt_to_equity):

        if (
            pd.notna(total_debt)
            and pd.notna(equity)
            and equity > 0
        ):

            debt_to_equity = (
                total_debt
                / equity
                * 100
            )

    cash_to_debt = np.nan

    if (
        pd.notna(cash)
        and pd.notna(total_debt)
        and total_debt > 0
    ):

        cash_to_debt = (
            cash
            / total_debt
        )

    current_ratio = clean_positive(
        safe_get(
            info,
            'currentRatio'
        )
    )

    if pd.isna(current_ratio):

        if (
            pd.notna(current_assets)
            and pd.notna(current_liabilities)
            and current_liabilities > 0
        ):

            current_ratio = (
                current_assets
                / current_liabilities
            )

    annualized_volatility = np.nan
    max_drawdown = np.nan

    try:

        close_for_risk = pd.to_numeric(
            hist['Close'],
            errors='coerce'
        ).dropna()

        if len(close_for_risk) >= 30:

            daily_returns = (
                close_for_risk
                .pct_change()
                .dropna()
            )

            if len(daily_returns) >= 20:

                annualized_volatility = (
                    daily_returns.std(ddof=1)
                    * np.sqrt(252)
                )

            running_max = (
                close_for_risk
                .cummax()
            )

            drawdown_series = (
                close_for_risk
                / running_max
                - 1.0
            )

            if not drawdown_series.empty:
                max_drawdown = float(
                    drawdown_series.min()
                )

    except Exception:
        pass

    historical_revenue = get_numeric_series(
        financials,
        REVENUE_KEYS
    )

    historical_net_income = get_numeric_series(
        financials,
        NET_INCOME_KEYS
    )

    revenue_growth_volatility = calculate_growth_volatility(
        historical_revenue
    )

    earnings_growth_volatility = calculate_growth_volatility(
        historical_net_income
    )

    analyst_target_mean = extract_analyst_target(
        analyst_targets,
        'mean'
    )

    analyst_target_median = extract_analyst_target(
        analyst_targets,
        'median'
    )

    analyst_target_low = extract_analyst_target(
        analyst_targets,
        'low'
    )

    analyst_target_high = extract_analyst_target(
        analyst_targets,
        'high'
    )

    if pd.isna(analyst_target_mean):
        analyst_target_mean = clean_positive(
            safe_get(
                info,
                'targetMeanPrice'
            )
        )

    # yfinance does not always provide a mean target even when a median
    # analyst target exists. The median is a valid fallback for Fair Value.
    if pd.isna(analyst_target_mean) and pd.notna(analyst_target_median):
        analyst_target_mean = analyst_target_median

    if pd.isna(analyst_target_median):
        analyst_target_median = clean_positive(
            safe_get(
                info,
                'targetMedianPrice'
            )
        )

    if pd.isna(analyst_target_low):
        analyst_target_low = clean_positive(
            safe_get(
                info,
                'targetLowPrice'
            )
        )

    if pd.isna(analyst_target_high):
        analyst_target_high = clean_positive(
            safe_get(
                info,
                'targetHighPrice'
            )
        )

    growth_inputs = []

    if pd.notna(revenue_growth):
        growth_inputs.append(
            (revenue_growth, 0.60)
        )

    if pd.notna(earnings_growth):
        growth_inputs.append(
            (earnings_growth, 0.40)
        )

    if growth_inputs:

        weighted_growth = (
            sum(
                value * weight
                for value, weight in growth_inputs
            )
            / sum(
                weight
                for _, weight in growth_inputs
            )
        )

        dcf_growth = np.clip(
            weighted_growth,
            -0.03,
            0.12
        )

    else:

        dcf_growth = 0.05

    fcf_fair_value = np.nan

    if sector != 'Financial Services':

        fcf_fair_value = calculate_fcf_fair_value(
            fcf,
            market_cap,
            current_price,
            dcf_growth
        )

    fair_value_price, fair_value_score = (
        calculate_fair_value_score(
            current_price,
            analyst_target_mean,
            fcf_fair_value
        )
    )

    analyst_target_upside = np.nan

    if (
        pd.notna(analyst_target_mean)
        and pd.notna(current_price)
        and current_price > 0
    ):

        analyst_target_upside = (
            analyst_target_mean
            / current_price
            - 1
        )

    fair_value_upside = np.nan

    if (
        pd.notna(fair_value_price)
        and pd.notna(current_price)
        and current_price > 0
    ):

        fair_value_upside = (
            fair_value_price
            / current_price
            - 1
        )

    above_sma200 = np.nan
    above_sma50 = np.nan

    perf_1m = np.nan
    perf_6m = np.nan
    perf_12m = np.nan

    sma50 = np.nan
    sma200 = np.nan

    rsi14 = np.nan
    macd = np.nan
    macd_signal = np.nan
    macd_hist = np.nan

    try:

        close = pd.to_numeric(
            hist['Close'],
            errors='coerce'
        ).dropna()

        latest_price = close.iloc[-1]

        if len(close) >= 50:

            sma50 = (
                close
                .rolling(50)
                .mean()
                .iloc[-1]
            )

            if pd.notna(sma50):

                above_sma50 = (
                    latest_price > sma50
                )

        if len(close) >= 200:

            sma200 = (
                close
                .rolling(200)
                .mean()
                .iloc[-1]
            )

            if pd.notna(sma200):

                above_sma200 = (
                    latest_price > sma200
                )

        if len(close) >= 22:

            old_price_1m = close.iloc[-22]

            if (
                pd.notna(old_price_1m)
                and old_price_1m > 0
            ):

                perf_1m = (
                    latest_price
                    / old_price_1m
                    - 1
                )

        if len(close) >= 127:

            old_price_6m = close.iloc[-127]

            if (
                pd.notna(old_price_6m)
                and old_price_6m > 0
            ):

                perf_6m = (
                    latest_price
                    / old_price_6m
                    - 1
                )

        if len(close) >= 253:

            old_price_12m = close.iloc[-253]

            if (
                pd.notna(old_price_12m)
                and old_price_12m > 0
            ):

                perf_12m = (
                    latest_price
                    / old_price_12m
                    - 1
                )

        rsi14 = calculate_rsi(
            close,
            14
        )

        (
            macd,
            macd_signal,
            macd_hist
        ) = calculate_macd(
            close
        )

    except Exception:
        pass

    # ========================================================
    # DATA COMPLETENESS
    # ========================================================

    completeness_blocks = []

    completeness_blocks.append(
        pd.notna(fair_value_score)
    )

    completeness_blocks.append(
        (
            pd.notna(forward_pe)
            or pd.notna(trailing_pe)
            or pd.notna(ps_ratio)
            or pd.notna(pb_ratio)
        )
    )

    if sector == 'Financial Services':

        quality_available = (
            pd.notna(roe)
            or pd.notna(net_margin)
            or pd.notna(earnings_growth)
        )

    else:

        quality_available = (
            pd.notna(roic)
            or pd.notna(gross_margin)
            or pd.notna(fcf_margin)
            or pd.notna(earnings_growth)
        )

    completeness_blocks.append(
        quality_available
    )

    risk_available = (
        pd.notna(annualized_volatility)
        or pd.notna(max_drawdown)
        or pd.notna(revenue_growth_volatility)
        or pd.notna(earnings_growth_volatility)
        or pd.notna(net_debt_ebitda)
        or pd.notna(current_ratio)
        or pd.notna(debt_to_equity)
        or pd.notna(cash_to_debt)
    )

    completeness_blocks.append(
        risk_available
    )

    technical_available = (
        pd.notna(rsi14)
        or pd.notna(macd)
        or pd.notna(perf_6m)
        or pd.notna(above_sma200)
    )

    completeness_blocks.append(
        technical_available
    )

    data_completeness = round(
        sum(completeness_blocks)
        / len(completeness_blocks)
        * 100
    )

    return {

        'ticker': ticker_symbol,
        'name': name,
        'sector': sector,

        'price': current_price,
        'market_cap': market_cap,
        'shares_outstanding': shares_outstanding,

        'revenue': revenue,
        'net_income': net_income,
        'ebit': ebit,
        'ebitda': ebitda,
        'pretax_income': pretax_income,
        'tax_provision': tax_provision,

        'total_assets': total_assets,
        'current_liabilities': current_liabilities,
        'current_assets': current_assets,
        'cash': cash,
        'total_debt': total_debt,
        'equity': equity,

        'trailing_eps': trailing_eps,
        'forward_eps': forward_eps,

        'earnings_growth': earnings_growth,
        'revenue_growth': revenue_growth,
        'long_term_growth': long_term_growth,

        'trailing_pe': trailing_pe,
        'forward_pe': forward_pe,
        'ps_ratio': ps_ratio,
        'pb_ratio': pb_ratio,
        'forward_peg': forward_peg,

        'gross_margin': gross_margin,
        'net_margin': net_margin,
        'roe': roe,
        'roic': roic,

        'operating_cf': operating_cf,
        'fcf': fcf,
        'fcf_yield': fcf_yield,
        'fcf_margin': fcf_margin,

        'net_debt_ebitda': net_debt_ebitda,
        'debt_to_equity': debt_to_equity,
        'cash_to_debt': cash_to_debt,
        'current_ratio': current_ratio,

        'annualized_volatility': annualized_volatility,
        'max_drawdown': max_drawdown,
        'revenue_growth_volatility': revenue_growth_volatility,
        'earnings_growth_volatility': earnings_growth_volatility,

        'analyst_target_mean': analyst_target_mean,
        'analyst_target_median': analyst_target_median,
        'analyst_target_low': analyst_target_low,
        'analyst_target_high': analyst_target_high,
        'analyst_target_upside': analyst_target_upside,

        'fcf_fair_value': fcf_fair_value,
        'fair_value_price': fair_value_price,
        'fair_value_upside': fair_value_upside,
        'fair_value_score': fair_value_score,

        'sma50': sma50,
        'sma200': sma200,
        'above_sma50': above_sma50,
        'above_sma200': above_sma200,

        'perf_1m': perf_1m,
        'perf_6m': perf_6m,
        'perf_12m': perf_12m,

        'rsi14': rsi14,
        'macd': macd,
        'macd_signal': macd_signal,
        'macd_hist': macd_hist,

        'data_completeness': data_completeness
    }



# ============================================================
# V20.0 SCORE ARCHITECTURE
# ============================================================

# Core architecture:
#   Quality + Future -> KPAX
#   Fair Value + Relative Valuation -> KPAX-FV
#   KPAX + KPAX-FV + Risk -> Investment Score
#   Technical remains separate and only gates the final recommendation.

INITIAL_SCORE_WEIGHTS = {
    'kpax': 0.40,
    'kpax_fv': 0.50,
    'risk': 0.10
}

# Recommendation thresholds are deliberately separate from score weights.
INVESTMENT_THRESHOLDS = {
    'strong_buy': 85.0,
    'buy': 80.0,
    'accumulate': 75.0,
    'hold': 68.0,
    'watch': 55.0
}


MIN_VALID_SCORE = 5.0

def clip_score(value):
    if pd.isna(value):
        return np.nan
    return float(np.clip(value, MIN_VALID_SCORE, 100.0))


def weighted_available(components):
    """Weighted average of available (non-NaN) components."""
    available = [
        (float(score), float(weight))
        for score, weight in components
        if pd.notna(score) and weight > 0
    ]

    if not available:
        return np.nan

    total_weight = sum(weight for _, weight in available)
    if total_weight <= 0:
        return np.nan

    return clip_score(
        sum(score * weight for score, weight in available)
        / total_weight
    )


def score_growth(value):
    """Generic growth score: -10% .. +50% -> 10 .. 100."""
    if pd.isna(value):
        return np.nan
    return clip_score(
        np.interp(
            value,
            [-0.10, 0.00, 0.05, 0.15, 0.30, 0.50],
            [10, 35, 55, 75, 90, 100]
        )
    )


def score_long_term_growth(value):
    if pd.isna(value):
        return np.nan
    return clip_score(
        np.interp(
            value,
            [-0.05, 0.00, 0.05, 0.10, 0.20, 0.35],
            [15, 35, 55, 70, 90, 100]
        )
    )


def score_analyst_upside(value):
    """Consensus upside score. Conservative caps avoid extreme targets dominating."""
    if pd.isna(value):
        return np.nan
    return clip_score(
        np.interp(
            value,
            [-0.30, -0.10, 0.00, 0.10, 0.20, 0.40, 0.70],
            [0, 20, 40, 60, 75, 90, 100]
        )
    )


def calculate_quality_score(metrics):
    """Quality = present business quality and profitability."""
    sector = metrics.get('sector', 'Default')
    components = []

    if sector == 'Financial Services':
        roe = metrics.get('roe')
        net_margin = metrics.get('net_margin')

        if pd.notna(roe):
            components.append((
                np.interp(roe, [0.04, 0.08, 0.12, 0.18, 0.25], [20, 45, 70, 90, 100]),
                0.40
            ))

        if pd.notna(net_margin):
            components.append((
                np.interp(net_margin, [0.05, 0.10, 0.20, 0.30, 0.40], [20, 45, 70, 90, 100]),
                0.30
            ))
    else:
        roic = metrics.get('roic')
        gross_margin = metrics.get('gross_margin')
        fcf_margin = metrics.get('fcf_margin')

        if pd.notna(roic):
            components.append((
                np.interp(roic, [0.04, 0.08, 0.12, 0.20, 0.35], [20, 45, 65, 90, 100]),
                0.30
            ))

        if pd.notna(gross_margin):
            components.append((
                np.interp(gross_margin, [0.15, 0.30, 0.45, 0.60, 0.75], [20, 40, 65, 85, 100]),
                0.15
            ))

        if pd.notna(fcf_margin):
            components.append((
                np.interp(fcf_margin, [-0.05, 0.00, 0.05, 0.10, 0.20], [0, 30, 65, 85, 100]),
                0.20
            ))

    earnings_growth = metrics.get('earnings_growth')
    revenue_growth = metrics.get('revenue_growth')

    if pd.notna(earnings_growth):
        components.append((score_growth(earnings_growth), 0.20))

    if pd.notna(revenue_growth):
        components.append((score_growth(revenue_growth), 0.10))

    return weighted_available(components)


def calculate_future_score(metrics):
    """Future = growth potential, long-term outlook and external expectations.

    Current quality is intentionally kept out as far as possible to avoid
    simply duplicating the Quality score.
    """
    earnings_growth = metrics.get('earnings_growth')
    revenue_growth = metrics.get('revenue_growth')
    long_term_growth = metrics.get('long_term_growth')
    analyst_upside = metrics.get('analyst_target_upside')
    fcf_margin = metrics.get('fcf_margin')

    components = []

    if pd.notna(earnings_growth):
        components.append((score_growth(earnings_growth), 0.30))

    if pd.notna(revenue_growth):
        components.append((score_growth(revenue_growth), 0.20))

    if pd.notna(long_term_growth):
        components.append((score_long_term_growth(long_term_growth), 0.15))

    if pd.notna(analyst_upside):
        components.append((score_analyst_upside(analyst_upside), 0.15))

    # FCF margin is used here only as a modest future cash-generation proxy.
    if pd.notna(fcf_margin):
        components.append((
            np.interp(
                fcf_margin,
                [-0.05, 0.00, 0.05, 0.10, 0.20],
                [0, 30, 65, 85, 100]
            ),
            0.10
        ))

    # Growth momentum: rewards current growth running ahead of long-term growth,
    # but only with a small 10% weight so one noisy year cannot dominate KPAX.
    if pd.notna(earnings_growth) and pd.notna(long_term_growth):
        acceleration = earnings_growth - long_term_growth
        acceleration_score = np.interp(
            acceleration,
            [-0.20, -0.05, 0.00, 0.05, 0.15, 0.30],
            [10, 30, 50, 65, 85, 100]
        )
        components.append((acceleration_score, 0.10))

    return weighted_available(components)


def calculate_valuation_score(metrics):
    """Relative valuation score from the existing V11 valuation logic."""
    sector = metrics.get('sector', 'Default')
    components = []

    forward_pe = metrics.get('forward_pe')
    if pd.notna(forward_pe) and forward_pe > 0:
        components.append((
            np.interp(forward_pe, [5, 10, 15, 22, 35, 60], [100, 92, 80, 65, 25, 0]),
            0.40
        ))

    trailing_pe = metrics.get('trailing_pe')
    if pd.notna(trailing_pe) and trailing_pe > 0:
        components.append((
            np.interp(trailing_pe, [5, 10, 15, 22, 35, 60], [100, 92, 80, 65, 25, 0]),
            0.20
        ))

    forward_peg = metrics.get('forward_peg')
    if pd.notna(forward_peg) and forward_peg > 0:
        components.append((
            np.interp(forward_peg, [0.4, 0.8, 1.0, 1.5, 2.5, 4.0], [100, 92, 80, 60, 20, 0]),
            0.20
        ))

    if sector == 'Financial Services':
        pb = metrics.get('pb_ratio')
        if pd.notna(pb) and pb > 0:
            components.append((
                np.interp(pb, [0.5, 0.8, 1.1, 1.6, 2.5, 4.0], [100, 92, 80, 55, 20, 0]),
                0.20
            ))
    else:
        ps = metrics.get('ps_ratio')
        if pd.notna(ps) and ps > 0:
            components.append((
                np.interp(ps, [0.5, 1.5, 3.0, 5.0, 8.0], [100, 85, 60, 30, 0]),
                0.20
            ))

    return weighted_available(components)


def calculate_risk_score(metrics):
    """Risk score where 100 = low risk / robust and 0 = high risk."""
    sector = metrics.get('sector', 'Default')

    financial_components = []
    if sector == 'Financial Services':
        debt_to_equity = metrics.get('debt_to_equity')
        cash_to_debt = metrics.get('cash_to_debt')

        if pd.notna(debt_to_equity):
            financial_components.append((
                np.interp(debt_to_equity, [20, 50, 100, 200, 400], [100, 90, 70, 40, 0]),
                0.60
            ))
        if pd.notna(cash_to_debt):
            financial_components.append((
                np.interp(cash_to_debt, [0.05, 0.20, 0.40, 0.75, 1.50], [10, 35, 60, 85, 100]),
                0.40
            ))
    else:
        net_debt_ebitda = metrics.get('net_debt_ebitda')
        current_ratio = metrics.get('current_ratio')
        debt_to_equity = metrics.get('debt_to_equity')

        if pd.notna(net_debt_ebitda):
            financial_components.append((
                np.interp(net_debt_ebitda, [-1, 0, 1, 2, 3, 4, 5, 6], [100, 100, 85, 70, 50, 30, 10, 0]),
                0.45
            ))
        if pd.notna(current_ratio):
            financial_components.append((
                np.interp(current_ratio, [0.5, 0.8, 1.0, 1.5, 2.0, 2.5], [10, 35, 55, 80, 95, 100]),
                0.20
            ))
        if pd.notna(debt_to_equity):
            financial_components.append((
                np.interp(debt_to_equity, [10, 30, 60, 100, 150, 250, 400], [100, 90, 75, 55, 40, 15, 0]),
                0.35
            ))

    financial_score = weighted_available(financial_components)

    volatility = metrics.get('annualized_volatility')
    volatility_score = np.nan
    if pd.notna(volatility):
        volatility_score = np.interp(
            volatility,
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            [100, 90, 80, 65, 50, 25, 0]
        )

    drawdown = metrics.get('max_drawdown')
    drawdown_score = np.nan
    if pd.notna(drawdown):
        drawdown_score = np.interp(
            abs(drawdown),
            [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70],
            [100, 90, 80, 60, 40, 20, 0]
        )

    market_risk_score = weighted_available([
        (volatility_score, 0.50),
        (drawdown_score, 0.50)
    ])

    revenue_vol = metrics.get('revenue_growth_volatility')
    earnings_vol = metrics.get('earnings_growth_volatility')

    revenue_stability = np.nan
    earnings_stability = np.nan

    if pd.notna(revenue_vol):
        revenue_stability = np.interp(
            revenue_vol,
            [0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
            [100, 85, 70, 50, 25, 0]
        )

    if pd.notna(earnings_vol):
        earnings_stability = np.interp(
            earnings_vol,
            [0.05, 0.10, 0.20, 0.30, 0.50, 0.70],
            [100, 85, 65, 45, 20, 0]
        )

    stability_score = weighted_available([
        (revenue_stability, 0.40),
        (earnings_stability, 0.60)
    ])

    return weighted_available([
        (financial_score, 0.40),
        (market_risk_score, 0.30),
        (stability_score, 0.30)
    ])


def calculate_technical_score(metrics):
    """Technical score remains completely separate from Investment Score."""
    components = []

    above_sma200 = metrics.get('above_sma200')
    if pd.notna(above_sma200):
        components.append((80 if bool(above_sma200) else 30, 0.25))

    perf_6m = metrics.get('perf_6m')
    if pd.notna(perf_6m):
        components.append((
            np.interp(perf_6m, [-0.40, -0.20, 0.00, 0.15, 0.35, 0.60], [0, 20, 45, 70, 90, 100]),
            0.25
        ))

    rsi = metrics.get('rsi14')
    if pd.notna(rsi):
        components.append((
            np.interp(rsi, [20, 30, 40, 50, 60, 70, 80, 90], [20, 35, 55, 68, 78, 85, 70, 50]),
            0.20
        ))

    macd = metrics.get('macd')
    macd_signal = metrics.get('macd_signal')
    macd_hist = metrics.get('macd_hist')
    if pd.notna(macd) and pd.notna(macd_signal):
        macd_score = 70.0 if macd > macd_signal else 35.0
        if pd.notna(macd_hist):
            if macd > macd_signal and macd_hist > 0:
                macd_score += 15
            elif macd <= macd_signal and macd_hist < 0:
                macd_score -= 10
        components.append((np.clip(macd_score, 0, 100), 0.20))

    above_sma50 = metrics.get('above_sma50')
    if pd.notna(above_sma50):
        components.append((80 if bool(above_sma50) else 35, 0.10))

    return weighted_available(components)


def calculate_scores(metrics):
    """Return all V20.0 scores and the fundamental Investment Score."""
    fair_value = metrics.get('fair_value_score')
    valuation = calculate_valuation_score(metrics)
    quality = calculate_quality_score(metrics)
    future = calculate_future_score(metrics)
    risk = calculate_risk_score(metrics)
    technical = calculate_technical_score(metrics)

    kpax = weighted_available([
        (quality, 0.40),
        (future, 0.60)
    ])

    kpax_fv = weighted_available([
        (fair_value, 0.60),
        (valuation, 0.40)
    ])

    # Investment Score: technical is deliberately excluded.
    investment_score = weighted_available([
        (kpax, INITIAL_SCORE_WEIGHTS['kpax']),
        (kpax_fv, INITIAL_SCORE_WEIGHTS['kpax_fv']),
        (risk, INITIAL_SCORE_WEIGHTS['risk'])
    ])

    return investment_score, {
        'quality': quality,
        'future': future,
        'kpax': kpax,
        'fair_value': fair_value,
        'valuation': valuation,
        'kpax_fv': kpax_fv,
        'risk': risk,
        'technical': technical
    }


# ============================================================
# RECOMMENDATION
# ============================================================

def get_recommendation(investment_score, kpax, kpax_fv, technical):
    """Fundamental score sets the level; technical score gates the action."""
    if pd.isna(investment_score):
        return "Keine Bewertung"

    # Fundamental gates first.
    if (
        investment_score >= INVESTMENT_THRESHOLDS['strong_buy']
        and pd.notna(kpax)
        and kpax >= 80
        and pd.notna(kpax_fv)
        and kpax_fv >= 80
    ):
        base = "Strong Buy"
    elif (
        investment_score >= INVESTMENT_THRESHOLDS['buy']
        and pd.notna(kpax)
        and kpax >= 75
    ):
        base = "Buy"
    elif investment_score >= INVESTMENT_THRESHOLDS['accumulate']:
        base = "Accumulate"
    elif investment_score >= INVESTMENT_THRESHOLDS['hold']:
        base = "Hold"
    elif investment_score >= INVESTMENT_THRESHOLDS['watch']:
        base = "Reduce / Watch"
    else:
        base = "Avoid"

    # Technical gate: it can downgrade, but never upgrade, the fundamental level.
    level = {
        'Avoid': 0,
        'Reduce / Watch': 1,
        'Hold': 2,
        'Accumulate': 3,
        'Buy': 4,
        'Strong Buy': 5
    }[base]

    if pd.notna(technical):
        if technical < 30:
            level = min(level, 2)       # max Hold
        elif technical < 45:
            level = min(level, 3)       # max Accumulate
        elif technical < 60:
            level = min(level, 4)       # max Buy
        # >=60: no downgrade

    labels = [
        'Avoid',
        'Reduce / Watch',
        'Hold',
        'Accumulate',
        'Buy',
        'Strong Buy'
    ]

    return labels[level]


def get_recommendation_note(investment_score, technical):
    if pd.isna(investment_score) or pd.isna(technical):
        return ""

    if investment_score >= 85 and technical < 60:
        return "⚠️ Fundamental stark, Technical bremst"
    if investment_score >= 80 and technical < 45:
        return "⚠️ Fundamental stark, technisch schwach"
    if technical < 30:
        return "⚠️ Sehr schwaches technisches Timing"
    if technical >= 75:
        return "✓ Technisches Setup stark"
    if technical >= 60:
        return "✓ Technisches Setup positiv"
    if technical < 45:
        return "⚠️ Technisches Timing schwach"
    return "Technisches Timing neutral"


def get_turnaround_status(metrics):
    # Retained as a descriptive timing label; it is no longer the mathematical
    # modifier of the fundamental score.
    perf_1m = metrics.get('perf_1m')
    perf_6m = metrics.get('perf_6m')
    rsi = metrics.get('rsi14')
    macd = metrics.get('macd')
    macd_signal = metrics.get('macd_signal')

    if pd.notna(perf_1m) and pd.notna(perf_6m):
        if perf_1m > 0 and perf_6m < 0:
            return "Turnaround?"
        if perf_1m > 0.05 and perf_6m > 0:
            return "Momentum"
        if perf_1m < -0.05 and perf_6m < 0:
            return "Abwärtstrend"

    if (
        pd.notna(macd)
        and pd.notna(macd_signal)
        and macd > macd_signal
        and pd.notna(rsi)
        and rsi < 50
    ):
        return "Turnaround?"

    return "Neutral"


# ============================================================
# TABLE COLORING
# ============================================================

def style_results_table(df):
    score_columns = [
        'Investment Score',
        'KPAX',
        'KPAX-FV',
        'Fair Value Score',
        'Relative Valuation',
        'Quality',
        'Future',
        'Risk',
        'Technical'
    ]

    completeness_column = ['Datenvollständigkeit (%)']

    styler = (
        df.style
        .background_gradient(cmap='RdYlGn', subset=score_columns, vmin=0, vmax=100)
        .background_gradient(cmap='RdYlGn', subset=completeness_column, vmin=0, vmax=100)
        .format({
            'Investment Score': '{:.1f}',
            'KPAX': '{:.0f}',
            'KPAX-FV': '{:.0f}',
            'Fair Value Score': '{:.0f}',
            'Relative Valuation': '{:.0f}',
            'Quality': '{:.0f}',
            'Future': '{:.0f}',
            'Risk': '{:.0f}',
            'Technical': '{:.0f}',
            'Forward-KGV': '{:.1f}',
            'Datenvollständigkeit (%)': '{:.0f}%'
        }, na_rep='—')
    )
    return styler


# ============================================================
# HEADER
# ============================================================

st.title("📊 Quant-Aktien-Screener V20.0")
st.caption(
    "KPAX-basierter Aktien-Screener mit getrennter Unternehmensqualität, Zukunftspotenzial, "
    "Preisattraktivität, Risiko und technischem Timing"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Einstellungen")

ticker_input = st.sidebar.text_input(
    "Ticker",
    value=(
        "BMW.DE, NVDA, MSFT, AAPL, "
        "GOOGL, AMZN, TTE.PA, ING, "
        "PFE, KO, NKE"
    )
)

min_score = st.sidebar.slider("Minimaler Investment Score", 0, 100, 0, 1)

if st.sidebar.button("🗑️ Cache löschen"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ V20.0 Gewichtung")
st.sidebar.caption(
    "Diese Gewichtung betrifft ausschließlich den Investment Score. "
    "Technical bleibt bewusst außerhalb des Investment Scores."
)

weight_labels = {
    'kpax': 'KPAX',
    'kpax_fv': 'KPAX-FV',
    'risk': 'Risk'
}

if 'v20_weights' not in st.session_state:
    st.session_state.v20_weights = INITIAL_SCORE_WEIGHTS.copy()

for key, label in weight_labels.items():
    current_value = st.session_state.v20_weights[key] * 100
    new_value = st.sidebar.number_input(
        label,
        min_value=0.0,
        max_value=100.0,
        value=float(current_value),
        step=1.0,
        key=f"v20_weight_{key}"
    )
    st.session_state.v20_weights[key] = new_value / 100.0

weight_sum = sum(st.session_state.v20_weights.values())
if weight_sum <= 0:
    st.sidebar.error("Mindestens eine Gewichtung muss > 0 sein.")
    active_weights = INITIAL_SCORE_WEIGHTS.copy()
else:
    active_weights = {
        key: value / weight_sum
        for key, value in st.session_state.v20_weights.items()
    }
    st.sidebar.success(f"Summe: {weight_sum * 100:.0f}% → normalisiert")

if st.sidebar.button("🔄 Reset", use_container_width=True):
    st.session_state.v20_weights = INITIAL_SCORE_WEIGHTS.copy()
    for key, value in INITIAL_SCORE_WEIGHTS.items():
        st.session_state[f"v20_weight_{key}"] = value * 100
    st.rerun()


# ============================================================
# ANALYSE
# ============================================================

if st.button("🚀 Aktien analysieren"):
    tickers = [x.strip().upper() for x in ticker_input.split(',') if x.strip()]

    if not tickers:
        st.warning("Bitte mindestens einen Ticker eingeben.")
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []

    for i, ticker_symbol in enumerate(tickers):
        status_text.write(f"Analysiere {ticker_symbol} ...")
        data = fetch_stock_data(ticker_symbol)

        if data is None:
            progress_bar.progress((i + 1) / len(tickers))
            continue

        metrics = extract_metrics(data)
        investment_score, scores = calculate_scores(metrics)

        # Apply current user-adjusted V20 weights.
        investment_score = weighted_available([
            (scores.get('kpax'), active_weights['kpax']),
            (scores.get('kpax_fv'), active_weights['kpax_fv']),
            (scores.get('risk'), active_weights['risk'])
        ])

        if pd.notna(investment_score) and investment_score < min_score:
            progress_bar.progress((i + 1) / len(tickers))
            continue

        recommendation = get_recommendation(
            investment_score,
            scores.get('kpax'),
            scores.get('kpax_fv'),
            scores.get('technical')
        )

        results.append({
            'Ticker': metrics['ticker'],
            'Name': metrics['name'],
            'Sektor': metrics['sector'],
            'Investment Score': investment_score,
            'KPAX': scores.get('kpax', np.nan),
            'KPAX-FV': scores.get('kpax_fv', np.nan),
            'Quality': scores.get('quality', np.nan),
            'Future': scores.get('future', np.nan),
            'Fair Value Score': scores.get('fair_value', np.nan),
            'Relative Valuation': scores.get('valuation', np.nan),
            'Risk': scores.get('risk', np.nan),
            'Technical': scores.get('technical', np.nan),
            'Empfehlung': recommendation,
            'Timing': get_recommendation_note(investment_score, scores.get('technical')),
            'Turnaround Status': get_turnaround_status(metrics),
            'Forward-KGV': metrics.get('forward_pe', np.nan),
            'Datenvollständigkeit (%)': metrics['data_completeness']
        })

        progress_bar.progress((i + 1) / len(tickers))

    progress_bar.empty()
    status_text.empty()

    if not results:
        st.warning("Keine Aktien konnten mit den aktuellen Einstellungen bewertet werden.")
        st.stop()

    results_df = pd.DataFrame(results)

    numeric_columns = [
        'Investment Score', 'KPAX', 'KPAX-FV', 'Quality', 'Future',
        'Fair Value Score', 'Relative Valuation', 'Risk', 'Technical',
        'Forward-KGV', 'Datenvollständigkeit (%)'
    ]

    for col in numeric_columns:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

    results_df = results_df.sort_values('Investment Score', ascending=False, na_position='last')

    st.subheader("📋 Ergebnisse")
    st.caption("Score 5 = echter, aber sehr schwacher Score. — = Daten/Berechnung nicht verfügbar.")
    st.dataframe(style_results_table(results_df), use_container_width=True, hide_index=True)

    # ========================================================
    # MATHEMATIK / MODELLLOGIK
    # ========================================================

    st.subheader("🧮 Mathematik & Berechnungslogik des Modells")
    st.caption(
        "Dieser Bereich ersetzt die bisherige Score-Analyse. Er zeigt die tatsächlich im V20.0-Code "
        "verwendeten Formeln, Gewichtungen, Interpolationen und Empfehlungsschwellen."
    )

    with st.expander("1. Gesamtarchitektur des Modells", expanded=True):
        st.markdown("""
### Hierarchie

**KPAX** bündelt Unternehmensqualität und Zukunftspotenzial:

\[
KPAX = 0{,}40 \cdot Quality + 0{,}60 \cdot Future
\]

**KPAX-FV** bündelt absolute und relative Preisbewertung:

\[
KPAX\text{-}FV = 0{,}60 \cdot Fair\ Value + 0{,}40 \cdot Relative\ Valuation
\]

Der **Investment Score** enthält bewusst **keinen Technical Score**:

\[
Investment = w_{KPAX}\cdot KPAX + w_{KPAX-FV}\cdot KPAX\text{-}FV + w_{Risk}\cdot Risk
\]

Die aktuell eingestellten Gewichte werden in der Sidebar festgelegt und anschließend auf 100 % normalisiert.
        """)
        st.code(
            f"Aktuelle normalisierte Gewichte:\n"
            f"KPAX       = {active_weights['kpax'] * 100:.1f}%\n"
            f"KPAX-FV    = {active_weights['kpax_fv'] * 100:.1f}%\n"
            f"Risk       = {active_weights['risk'] * 100:.1f}%\n\n"
            "Technical = separat; wird nicht in den Investment Score eingerechnet.",
            language=None
        )

    with st.expander("2. Quality Score – Unternehmensqualität"):
        st.markdown("""
### Nicht-Finanzunternehmen

Die verfügbaren Komponenten werden **nur mit ihrem jeweiligen Gewicht berücksichtigt**. Fehlt eine Kennzahl, wird deren Gewicht auf die vorhandenen Komponenten verteilt.

\[
Quality = weighted\_average(ROIC, GrossMargin, FCFMargin, EarningsGrowth, RevenueGrowth)
\]

Gewichte:

| Komponente | Gewicht | Score-Transformation |
|---|---:|---|
| ROIC | 30 % | 4 / 8 / 12 / 20 / 35 % → 20 / 45 / 65 / 90 / 100 |
| Bruttomarge | 15 % | 15 / 30 / 45 / 60 / 75 % → 20 / 40 / 65 / 85 / 100 |
| FCF-Marge | 20 % | −5 / 0 / 5 / 10 / 20 % → 0 / 30 / 65 / 85 / 100 |
| Gewinnwachstum | 20 % | −10 / 0 / 5 / 15 / 30 / 50 % → 10 / 35 / 55 / 75 / 90 / 100 |
| Umsatzwachstum | 10 % | −10 / 0 / 5 / 15 / 30 / 50 % → 10 / 35 / 55 / 75 / 90 / 100 |

Die Interpolation zwischen den Stützpunkten ist linear.

### Financial Services

Bei Banken/Finanzdienstleistern wird ROIC durch ROE ersetzt:

| Komponente | Gewicht |
|---|---:|
| ROE | 40 % |
| Nettomarge | 30 % |
| Gewinnwachstum | 20 % |
| Umsatzwachstum | 10 % |
| 

ROE: 4 / 8 / 12 / 18 / 25 % → 20 / 45 / 70 / 90 / 100.
Nettomarge: 5 / 10 / 20 / 30 / 40 % → 20 / 45 / 70 / 90 / 100.
        """)

    with st.expander("3. Future Score – Zukunftspotenzial"):
        st.markdown("""
\[
Future = weighted\_average(EarningsGrowth, RevenueGrowth, LongTermGrowth,
AnalystUpside, FCFMargin, GrowthAcceleration)
\]

| Komponente | Gewicht | Bewertungslogik |
|---|---:|---|
| Gewinnwachstum | 30 % | −10 / 0 / 5 / 15 / 30 / 50 % → 10 / 35 / 55 / 75 / 90 / 100 |
| Umsatzwachstum | 20 % | gleiche Transformation |
| Langfristiges Wachstum | 15 % | −5 / 0 / 5 / 10 / 20 / 35 % → 15 / 35 / 55 / 70 / 90 / 100 |
| Analysten-Upside | 15 % | −30 / −10 / 0 / 10 / 20 / 40 / 70 % → 0 / 20 / 40 / 60 / 75 / 90 / 100 |
| FCF-Marge | 10 % | −5 / 0 / 5 / 10 / 20 % → 0 / 30 / 65 / 85 / 100 |
| Wachstumsbeschleunigung | 10 % | \(EarningsGrowth - LongTermGrowth\) |

Für die Wachstumsbeschleunigung gilt:

\[
Acceleration = EarningsGrowth - LongTermGrowth
\]

und die lineare Score-Transformation:

−20 / −5 / 0 / 5 / 15 / 30 % → 10 / 30 / 50 / 65 / 85 / 100.
        """)

    with st.expander("4. Fair Value Score – absolute Preisattraktivität"):
        st.markdown("""
Der Fair-Value-Score wird aus dem bereits berechneten Fair Value relativ zum aktuellen Aktienkurs abgeleitet.

\[
Upside = \frac{FairValue}{CurrentPrice}-1
\]

Danach erfolgt eine lineare Interpolation:

| Upside | Score |
|---:|---:|
| −40 % | 0 → intern auf mindestens 5 begrenzt |
| −20 % | 15 |
| 0 % | 40 |
| +10 % | 58 |
| +25 % | 75 |
| +50 % | 90 |
| +75 % | 100 |

**Wichtig:** Ein berechneter, aber extrem schlechter Wert ist damit ein echter niedriger Score. Fehlen dagegen sämtliche Fair-Value-Daten, bleibt der Wert `None/—`.

Die eigentliche V20.0-Fair-Value-Berechnung verwendet aktuell zwei mögliche Quellen:

\[
FairValue = 0{,}75\cdot AnalystTarget + 0{,}25\cdot FCF\text{-}FairValue
\]

wenn beide Quellen vorhanden sind. Ist nur eine Quelle vorhanden, wird diese allein verwendet.

### FCF-DCF

Zunächst:

\[
FCFYield = \frac{FCF}{MarketCap}
\]

\[
FCF/share = FCFYield\cdot CurrentPrice
\]

Das FCF wird fünf Jahre projiziert. Das jährliche Wachstum wird dabei linear vom Startwachstum zum Terminalwachstum von 2,5 % zurückgeführt.

\[
FCF_t = FCF_{t-1}\cdot(1+g_t)
\]

\[
PV(FCF_t)=\frac{FCF_t}{(1+r)^t}
\]

Terminal Value:

\[
TV=\frac{FCF_5\cdot(1+g_{terminal})}{r-g_{terminal}}
\]

\[
FairValue_{DCF}=\sum_{t=1}^{5}PV(FCF_t)+PV(TV)
\]

Aktuelle Modellparameter:
- Diskontsatz \(r = 9\%\)
- Terminal Growth \(g_{terminal}=2{,}5\%\)
- Prognosezeitraum = 5 Jahre
- Startwachstum wird auf −3 % bis +12 % begrenzt.

Zusätzlich verwirft der aktuelle Code DCF-Werte außerhalb von **60 % bis 175 % des aktuellen Aktienkurses**. Das ist eine Plausibilitätsbremse und keine mathematische Eigenschaft eines DCF.
        """)

    with st.expander("5. Relative Valuation Score"):
        st.markdown("""
Die relative Bewertung besteht aus mehreren Multiples. Auch hier werden nur tatsächlich vorhandene Kennzahlen gewichtet.

| Kennzahl | Gewicht | Score-Stützpunkte |
|---|---:|---|
| Forward-KGV | 40 % | 5 / 10 / 15 / 22 / 35 / 60 → 100 / 92 / 80 / 65 / 25 / 0 |
| Trailing-KGV | 20 % | 5 / 10 / 15 / 22 / 35 / 60 → 100 / 92 / 80 / 65 / 25 / 0 |
| Forward-PEG | 20 % | 0,4 / 0,8 / 1,0 / 1,5 / 2,5 / 4,0 → 100 / 92 / 80 / 60 / 20 / 0 |
| P/S bzw. P/B | 20 % | sektorspezifisch |

Für Financial Services wird P/B verwendet:

0,5 / 0,8 / 1,1 / 1,6 / 2,5 / 4,0 → 100 / 92 / 80 / 55 / 20 / 0.

Für andere Unternehmen wird P/S verwendet:

0,5 / 1,5 / 3 / 5 / 8 → 100 / 85 / 60 / 30 / 0.
        """)

    with st.expander("6. Risk Score – Robustheit und Risiko"):
        st.markdown("""
Der Risk Score ist so definiert, dass **100 = geringes Risiko** und **5 ≈ sehr hohes Risiko** bedeutet.

\[
Risk = 40\%\cdot FinancialRisk + 30\%\cdot MarketRisk + 30\%\cdot StabilityRisk
\]

### Financial Risk – Nicht-Finanzunternehmen

\[
FinancialRisk = 45\%\cdot NetDebt/EBITDA + 20\%\cdot CurrentRatio + 35\%\cdot Debt/Equity
\]

### Market Risk

\[
MarketRisk = 50\%\cdot VolatilityScore + 50\%\cdot DrawdownScore
\]

### Stability

\[
StabilityRisk = 40\%\cdot RevenueStability + 60\%\cdot EarningsStability
\]

Die einzelnen Kennzahlen werden jeweils über lineare Interpolation auf eine 0–100-Skala transformiert. Fehlende Komponenten werden nicht mit 0 bestraft; ihr Gewicht wird auf die verfügbaren Komponenten umgelegt.

Bei Financial Services werden statt Net Debt/EBITDA und Current Ratio die dort sinnvolleren Größen Debt/Equity und Cash/Debt verwendet.
        """)

    with st.expander("7. Technical Score – separates Timing-Modell"):
        st.markdown("""
Der Technical Score fließt **nicht** in den Investment Score ein. Er beeinflusst ausschließlich die finale Empfehlung als nachgelagerter Gatekeeper.

\[
Technical = weighted\_average(SMA200, Performance6M, RSI, MACD, SMA50)
\]

Gewichte:

| Signal | Gewicht | Kernlogik |
|---|---:|---|
| Kurs über SMA200 | 25 % | Ja = 80, Nein = 30 |
| 6-Monats-Performance | 25 % | −40 / −20 / 0 / 15 / 35 / 60 % → 0 / 20 / 45 / 70 / 90 / 100 |
| RSI(14) | 20 % | 20 / 30 / 40 / 50 / 60 / 70 / 80 / 90 → 20 / 35 / 55 / 68 / 78 / 85 / 70 / 50 |
| MACD | 20 % | Basis 70 bei MACD > Signal, sonst 35; Histogramm ±15/−10 |
| Kurs über SMA50 | 10 % | Ja = 80, Nein = 35 |

RSI, MACD und gleitende Durchschnitte werden aus den historischen Kursdaten berechnet.
        """)

    with st.expander("8. Datenverfügbarkeit, fehlende Werte und Score-Minimum"):
        st.markdown("""
### Fehlende Daten

Das Modell verwendet grundsätzlich:

\[
weighted\_available = \frac{\sum(score_i\cdot weight_i)}{\sum weight_i}
\]

wobei nur vorhandene Scores in die Berechnung eingehen.

Damit wird ein fehlender Wert **nicht automatisch als 0** behandelt.

### Mindestwert

Der interne Score wird auf folgende Grenzen begrenzt:

\[
Score = clip(Score, 5, 100)
\]

Daher gilt:

- **5–100:** echter berechneter Score
- **— / None:** benötigte Daten bzw. Berechnung nicht verfügbar

Das verhindert, dass eine Datenlücke fälschlich wie ein extrem schlechter Score aussieht.
        """)

    with st.expander("9. Finale Empfehlung – Score + Technical Gate"):
        st.markdown("""
Zuerst wird aus dem Investment Score die fundamentale Empfehlung bestimmt:

| Investment Score | Basis-Empfehlung |
|---:|---|
| ≥ 85 und KPAX ≥ 80 und KPAX-FV ≥ 80 | **Strong Buy** |
| ≥ 80 und KPAX ≥ 75 | **Buy** |
| ≥ 75 | **Accumulate** |
| ≥ 68 | **Hold** |
| ≥ 55 | **Reduce / Watch** |
| < 55 | **Avoid** |

Danach kann Technical die Empfehlung **nur herabstufen, niemals verbessern**:

| Technical Score | Maximale Empfehlung |
|---:|---|
| < 30 | Hold |
| 30–44,9 | Accumulate |
| 45–59,9 | Buy |
| ≥ 60 | keine technische Herabstufung |

Damit bleibt die fundamentale Investmentbewertung unabhängig von kurzfristiger Markttechnik, während schlechtes technisches Timing eine zu aggressive Handlungsempfehlung verhindert.
        """)

    st.info(
        "Hinweis: Dieser Bereich dokumentiert bewusst die Mathematik des aktuellen V20.0-Modells. "
        "Er verändert keine Scores und keine Datenbeschaffung. Die Fair-Value-Methodik selbst können wir "
        "im nächsten Schritt separat mathematisch verbessern."
    )

else:
    st.info("Ticker eingeben und „🚀 Aktien analysieren“ klicken.")
