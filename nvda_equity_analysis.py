import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime

# NOTE: scikit-learn and TensorFlow are imported inside Section 13 only, so the
# rest of the analysis runs even if TensorFlow is not installed (e.g. on very
# new Python versions that TensorFlow does not yet support).


# === 1. CAPM regression — summary table (Python output) =====================
start = "2015-01-01"
end   = "2026-06-29"

nvda = yf.download("NVDA", start=start, end=end, auto_adjust=True)["Close"]
mkt  = yf.download("^GSPC", start=start, end=end, auto_adjust=True)["Close"]

nvda_ret = nvda.pct_change().dropna()
mkt_ret  = mkt.pct_change().dropna()

df = pd.concat([nvda_ret, mkt_ret], axis=1).dropna()
df.columns = ["nvda_ret", "mkt_ret"]

df["rf"] = 0.0

df["nvda_excess"] = df["nvda_ret"]
df["mkt_excess"]  = df["mkt_ret"]

X = sm.add_constant(df["mkt_excess"])
y = df["nvda_excess"]

capm_model = sm.OLS(y, X).fit()

alpha = float(capm_model.params["const"])
beta  = float(capm_model.params[df["mkt_excess"].name])

table_text = f"""Table X: CAPM regression summary (Python output).
Alpha (Intercept): {alpha:.4f}
Beta (Market): {beta:.2f}
R-squared: {capm_model.rsquared:.3f}
p-value (Beta): {capm_model.pvalues.iloc[1]:.3g}
Number of observations (N): {int(capm_model.nobs)}
Standard error (Beta): {capm_model.bse.iloc[1]:.3f}
"""

print(table_text)
print(capm_model.summary())


# Plot settings

# === 2. Setup, data download & daily returns ===============================
plt.style.use("default")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True

# Parameters
TICKER = "NVDA"      # NVIDIA
BENCHMARK = "^GSPC"  # S&P 500 (you can change to ^NDX etc.)
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

print("Settings OK:", TICKER, BENCHMARK, START_DATE, END_DATE)

nvda = yf.download(TICKER, start=START_DATE, end=END_DATE)
bench = yf.download(BENCHMARK, start=START_DATE, end=END_DATE)

print("NVDA columns:", nvda.columns.tolist())
print("Benchmark columns:", bench.columns.tolist())
nvda.head()

def get_price_column(df):
    """Return the best available price column name."""
    for col in ["Adj Close", "Adj_Close", "Close", "close"]:
        if col in df.columns:
            return col
    raise KeyError("No suitable price column found (Adj Close / Close).")

price_col_nvda = get_price_column(nvda)
price_col_bench = get_price_column(bench)

print("Using NVDA price column:", price_col_nvda)
print("Using benchmark price column:", price_col_bench)

# Daily returns
nvda["Return"] = nvda[price_col_nvda].pct_change()
bench["Return"] = bench[price_col_bench].pct_change()

# Combined returns DataFrame
returns = pd.concat(
    [nvda["Return"], bench["Return"]],
    axis=1,
    keys=[TICKER, "Market"]
).dropna()

returns.head()


# === 3. Technical analysis — moving averages ===============================
nvda["MA50"] = nvda[price_col_nvda].rolling(window=50).mean()
nvda["MA200"] = nvda[price_col_nvda].rolling(window=200).mean()

fig, ax = plt.subplots()
ax.plot(nvda.index, nvda[price_col_nvda], label=f"{TICKER} price")
ax.plot(nvda.index, nvda["MA50"], label="50-day MA")
ax.plot(nvda.index, nvda["MA200"], label="200-day MA")

ax.set_title(
    f"{TICKER} Price with 50- and 200-day Moving Averages\n"
    f"Source: Yahoo Finance, currency: USD, period: {START_DATE}–{END_DATE}"
)
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
plt.tight_layout()
plt.show()

# Define window for Bollinger Bands

# === 4. Technical analysis — Bollinger Bands ===============================
window = 20
nvda["MB"] = nvda[price_col_nvda].rolling(window=window).mean()   # Middle band
nvda["STD"] = nvda[price_col_nvda].rolling(window=window).std()
nvda["UB"] = nvda["MB"] + 2 * nvda["STD"]                         # Upper band
nvda["LB"] = nvda["MB"] - 2 * nvda["STD"]                         # Lower band

fig, ax = plt.subplots()
ax.plot(nvda.index, nvda[price_col_nvda], label=f"{TICKER} price")
ax.plot(nvda.index, nvda["MB"], label="Middle Band (20-day MA)")
ax.plot(nvda.index, nvda["UB"], label="Upper Band (+2σ)")
ax.plot(nvda.index, nvda["LB"], label="Lower Band (-2σ)")

ax.set_title(
    f"{TICKER} Bollinger Bands (20 days)\n"
    f"Source: Yahoo Finance, currency: USD, period: {START_DATE}–{END_DATE}"
)
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
plt.tight_layout()
plt.show()


# === 5. Technical analysis — Relative Strength Index (RSI) =================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

nvda["RSI14"] = compute_rsi(nvda[price_col_nvda], period=14)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

ax1.plot(nvda.index, nvda[price_col_nvda], label=f"{TICKER} price")
ax1.set_ylabel("Price (USD)")
ax1.legend()

ax2.plot(nvda.index, nvda["RSI14"], label="RSI (14 days)")
ax2.axhline(70, linestyle="--")
ax2.axhline(30, linestyle="--")
ax2.set_ylabel("RSI")
ax2.set_xlabel("Date")
ax2.legend()

fig.suptitle(
    f"{TICKER} Price and 14-day RSI\n"
    f"Source: Yahoo Finance, currency: USD, period: {START_DATE}–{END_DATE}"
)
plt.tight_layout()
plt.show()


# === 6. CAPM regression — scatter plot & fitted line =======================
annual_rf = 0.04
daily_rf = (1 + annual_rf) ** (1/252) - 1

returns["Rf"] = daily_rf
returns["Excess_NVDA"] = returns[TICKER] - returns["Rf"]
returns["Excess_Market"] = returns["Market"] - returns["Rf"]

returns.head()

capm_data = returns[["Excess_NVDA", "Excess_Market"]].dropna()

Y = capm_data["Excess_NVDA"]
X = sm.add_constant(capm_data["Excess_Market"])

capm_model = sm.OLS(Y, X).fit()
print(capm_model.summary())
alpha = capm_model.params["const"]
beta = capm_model.params["Excess_Market"]

fig, ax = plt.subplots()
ax.scatter(
    capm_data["Excess_Market"], capm_data["Excess_NVDA"],
    s=5, alpha=0.5, label="Daily observations"
)

x_vals = np.linspace(capm_data["Excess_Market"].min(),
                     capm_data["Excess_Market"].max(), 100)
y_vals = alpha + beta * x_vals
ax.plot(x_vals, y_vals, label=f"CAPM line (alpha={alpha:.4f}, beta={beta:.2f})")

ax.set_title(
    f"CAPM Regression: {TICKER} vs {BENCHMARK}\n"
    f"Source: Yahoo Finance, daily excess returns, period: {START_DATE}–{END_DATE}"
)
ax.set_xlabel("Market excess return")
ax.set_ylabel(f"{TICKER} excess return")
ax.legend()
plt.tight_layout()
plt.show()


# === 7. Cumulative returns — NVDA vs market ================================
cum_nvda = (1 + returns[TICKER]).cumprod()
cum_market = (1 + returns["Market"]).cumprod()

fig, ax = plt.subplots()
ax.plot(cum_nvda.index, cum_nvda, label=TICKER)
ax.plot(cum_market.index, cum_market, label="Market")

ax.set_title(
    f"Cumulative Returns: {TICKER} vs {BENCHMARK}\n"
    f"Source: Yahoo Finance, currency: USD, period: {START_DATE}–{END_DATE}"
)
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1 investment")
ax.legend()
plt.tight_layout()
plt.show()


# === 8. Fundamental analysis — financial statements & ratios ===============
nvda_ticker = yf.Ticker("NVDA")

fin = nvda_ticker.financials
bs = nvda_ticker.balance_sheet
cf = nvda_ticker.cashflow

print("Income statement rows:", fin.index.tolist())
print("Balance sheet rows:", bs.index.tolist())
print("Cash flow rows:", cf.index.tolist())
print("Columns (fiscal period ends):", fin.columns)

def get_row(df, row_name):
    if row_name in df.index:
        return df.loc[row_name]
    else:
        return pd.Series(index=df.columns, dtype="float64")

fund_raw = pd.DataFrame({
    "Revenue":          get_row(fin, "Total Revenue"),
    "NetIncome":        get_row(fin, "Net Income"),
    "OperatingIncome":  get_row(fin, "Operating Income"),
    "InterestExpense":  get_row(fin, "Interest Expense"),
    "TotalAssets":      get_row(bs, "Total Assets"),
    "TotalEquity":      get_row(bs, "Total Stockholder Equity"),
    "TotalDebt":        get_row(bs, "Total Debt"),
    "CurrentAssets":    get_row(bs, "Total Current Assets"),
    "CurrentLiabilities": get_row(bs, "Total Current Liabilities"),
    "OperatingCashFlow":  get_row(cf, "Operating Cash Flow")
})

fund = fund_raw.T.copy()
fund.columns = [c.year for c in fund.columns]
fund = fund.T.sort_index()

fund["NetMargin"] = fund["NetIncome"] / fund["Revenue"]
fund["ROA"] = fund["NetIncome"] / fund["TotalAssets"]
fund["ROE"] = fund["NetIncome"] / fund["TotalEquity"]

fund["DebtToEquity"] = fund["TotalDebt"] / fund["TotalEquity"]
fund["DebtToAssets"] = fund["TotalDebt"] / fund["TotalAssets"]
fund["InterestCoverage"] = fund["OperatingIncome"] / (-fund["InterestExpense"].replace(0,np.nan))

fund["CurrentRatio"] = fund["CurrentAssets"] / fund["CurrentLiabilities"]
fund["OCF_to_NI"] = fund["OperatingCashFlow"] / fund["NetIncome"]

fund["RevenueGrowth"] = fund["Revenue"].pct_change()
fund["NetIncomeGrowth"] = fund["NetIncome"].pct_change()

fund

fig, ax = plt.subplots()

ax.plot(fund.index, fund["NetMargin"] * 100, marker="o", label="Net margin (%)")
ax.plot(fund.index, fund["ROA"] * 100, marker="o", label="ROA (%)")
ax.plot(fund.index, fund["ROE"] * 100, marker="o", label="ROE (%)")

ax.set_title(
    "NVIDIA Profitability Ratios\n"
    "Source: Yahoo Finance (NVDA financials), currency: USD"
)
ax.set_xlabel("Fiscal year")
ax.set_ylabel("Ratio (%)")
ax.legend()
plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots()

ax1.plot(fund.index, fund["Revenue"] / 1e9, marker="o", label="Revenue (USD bn)")
ax1.plot(fund.index, fund["NetIncome"] / 1e9, marker="o", label="Net income (USD bn)")

ax1.set_title(
    "NVIDIA Revenue and Net Income Growth\n"
    "Source: Yahoo Finance (NVDA financials), currency: USD"
)
ax1.set_xlabel("Fiscal year")
ax1.set_ylabel("USD billions")
ax1.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()

ax.plot(fund.index, fund["DebtToEquity"], marker="o", label="Debt/Equity")
ax.plot(fund.index, fund["DebtToAssets"], marker="o", label="Debt/Assets")
ax.plot(fund.index, fund["InterestCoverage"], marker="o", label="Interest coverage (x)")

ax.set_title(
    "NVIDIA Capital Structure and Solvency Ratios\n"
    "Source: Yahoo Finance (NVDA financials & balance sheet), currency: USD"
)
ax.set_xlabel("Fiscal year")
ax.set_ylabel("Ratio")
ax.legend()
plt.tight_layout()
plt.show()
fig, ax = plt.subplots()

ax.plot(fund.index, fund["CurrentRatio"], marker="o", label="Current ratio")
ax.plot(fund.index, fund["OCF_to_NI"], marker="o", label="Operating cash flow / Net income")

ax.set_title(
    "NVIDIA Liquidity and Cash-Flow Quality\n"
    "Source: Yahoo Finance (NVDA financials, balance sheet, cash flow), currency: USD"
)
ax.set_xlabel("Fiscal year")
ax.set_ylabel("Ratio")
ax.legend()
plt.tight_layout()
plt.show()


# === 9. Peer comparison — valuation multiples (live) =======================
peer_tickers = ["NVDA", "AMD", "AVGO"]
peers = {t: yf.Ticker(t) for t in peer_tickers}

data = []
for t, tk in peers.items():
    info = tk.info  # may be slow; needs internet
    data.append({
        "Ticker": t,
        "PE (TTM)": info.get("trailingPE", np.nan),
        "Forward P/E": info.get("forwardPE", np.nan),
        "P/S (TTM)": info.get("priceToSalesTrailing12Months", np.nan),
        "P/B": info.get("priceToBook", np.nan),
        "EV/EBITDA": info.get("enterpriseToEbitda", np.nan)
    })

val_df = pd.DataFrame(data).set_index("Ticker")
val_df
fig, ax = plt.subplots()

val_df[["PE (TTM)", "EV/EBITDA"]].plot(kind="bar", ax=ax)

ax.set_title(
    "Valuation Multiples: NVIDIA vs Peers (AMD, AVGO)\n"
    "Source: Yahoo Finance, currency: USD"
)
ax.set_xlabel("Ticker")
ax.set_ylabel("Multiple")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# === 10. Peer comparison — market cap, P/E, ROE (live) =====================
tickers = ["NVDA", "AMD", "AVGO"]

rows = []
for t in tickers:
    info = yf.Ticker(t).info
    rows.append({
        "Ticker": t,
        "MarketCap (USD bn)": info.get("marketCap", np.nan) / 1e9,
        "P/E (TTM)": info.get("trailingPE", np.nan),
        "ROE (%)": (info.get("returnOnEquity", np.nan) or np.nan) * 100
    })

comp_df = pd.DataFrame(rows).set_index("Ticker")
comp_df
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

comp_df["MarketCap (USD bn)"].plot(kind="bar", ax=axes[0])
axes[0].set_title("Market capitalisation")
axes[0].set_ylabel("USD billions")

comp_df["P/E (TTM)"].plot(kind="bar", ax=axes[1])
axes[1].set_title("P/E ratio (TTM)")

comp_df["ROE (%)"].plot(kind="bar", ax=axes[2])
axes[2].set_title("Return on equity")
axes[2].set_ylabel("%")

for ax in axes:
    ax.set_xlabel("Ticker")
    ax.tick_params(axis="x", rotation=0)

fig.suptitle(
    "NVIDIA vs peers: Market cap, P/E and ROE\n"
    "Source: Yahoo Finance, currency: USD",
    y=1.05
)
plt.tight_layout()
plt.show()


# === 11. Peer comparison — snapshot used in the report (offline) ===========
# example – change numbers to yours if you enter them manually
data = {
    "Ticker": ["NVDA", "AMD", "AVGO"],
    "MarketCap (USD bn)": [325.97, 360.81, 188.90],
    "P/E (TTM)": [36.8, 46.8, 106.4],
    "ROE (%)": [31.4, 105.3, 28.7]
}

comp_df = pd.DataFrame(data).set_index("Ticker")
metrics = ["MarketCap (USD bn)", "P/E (TTM)", "ROE (%)"]

ax = comp_df[metrics].T.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_title("Comparison of NVDA, AMD and AVGO\nby Market cap, P/E and ROE")
ax.set_xlabel("Metric")
ax.set_ylabel("Market cap (USD bn) / Ratios")
ax.legend(title="Ticker")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# === 12. Revenue by segment ================================================
years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]


data_center = [1.932, 2.932, 2.983, 6.696, 10.613, 15.005, 47.525, 115.186, 191.661]
gaming      = [5.513, 6.246, 5.518, 7.759, 12.462, 9.067, 10.447, 11.350, 16.355]
auto        = [0.558, 0.641, 0.700, 0.536, 0.566, 0.903, 1.091, 1.694, 2.392]
pro_viz     = [0.934, 1.060, 1.053, 1.053, 1.211, 1.544, 0.0, 0.0, 0.0]

df = pd.DataFrame({
    "Year": years,
    "Data Center": data_center,
    "Gaming": gaming,
    "Automotive": auto,

})

df
plt.figure(figsize=(10, 6))

plt.plot(df["Year"], df["Data Center"], marker="o", label="Data Center")
plt.plot(df["Year"], df["Gaming"], marker="o", label="Gaming")
plt.plot(df["Year"], df["Automotive"], marker="o", label="Automotive")

plt.title(
    "NVIDIA Revenue by Segment\n"
    "Source: Bloomberg, currency: USD (billions)"
)
plt.xlabel("Fiscal year")
plt.ylabel("Revenue (USD billions)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# === 13. Machine learning — LSTM price forecast ============================
#
# This section needs TensorFlow + scikit-learn. They are imported here (not
# at the top) and wrapped in try/except so the rest of the script still runs
# if TensorFlow is unavailable for your Python version.
try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    np.random.seed(42)

    ticker = "NVDA"

    data = yf.download(ticker, start="2020-01-01", auto_adjust=False)

    if "Adj Close" in data.columns:
        price_col_name = "Adj Close"
    else:
        price_col_name = "Close"

    df = data[[price_col_name]].copy()
    df.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(df[[price_col_name]])

    def create_sequences(series, seq_len=60):
        X, y = [], []
        for i in range(seq_len, len(series)):
            X.append(series[i-seq_len:i, 0])
            y.append(series[i, 0])
        return np.array(X), np.array(y)

    seq_len = 60
    X, y = create_sequences(scaled_prices, seq_len=seq_len)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(seq_len, 1)))
    model.add(LSTM(64))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[es],
        verbose=1
    )

    all_pred_scaled = model.predict(X)
    all_pred = scaler.inverse_transform(all_pred_scaled.reshape(-1, 1))

    pred_index = df.index[seq_len:]

    df_pred = pd.DataFrame(
        {
            "Actual": df[price_col_name].squeeze().values[seq_len:],
            "LSTM_pred": all_pred.flatten()
        },
        index=pred_index
    )

    plt.figure(figsize=(12, 5))
    plt.plot(df_pred.index, df_pred["Actual"], label="Actual price")
    plt.plot(df_pred.index, df_pred["LSTM_pred"], label="LSTM fitted", alpha=0.8)
    plt.title("NVDA – LSTM fitted vs actual (2020–present)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    future_steps = 378

    last_sequence = scaled_prices[-seq_len:].reshape(1, seq_len, 1)
    future_scaled = []

    for _ in range(future_steps):
        next_scaled = model.predict(last_sequence, verbose=0)[0, 0]
        future_scaled.append(next_scaled)

        new_seq = np.append(last_sequence[0, 1:, 0], next_scaled)
        last_sequence = new_seq.reshape(1, seq_len, 1)

    future_scaled = np.array(future_scaled).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_scaled).flatten()

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=future_steps,
                                 freq="B")

    df_future = pd.DataFrame({"Forecast_price": future_prices}, index=future_dates)

    history_window = 252 * 2
    hist_prices = df[price_col_name].squeeze().iloc[-history_window:]
    hist_dates = df.index[-history_window:]

    plt.figure(figsize=(12, 5))
    plt.plot(hist_dates, hist_prices, label="Historical price (last ~2 years)")
    plt.plot(df_future.index, df_future["Forecast_price"], "r--",
             label="LSTM forecast (next ~1.5 years)")

    plt.title("NVDA – LSTM share price prediction\n(illustrative machine-learning model)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(
        f"Predicted NVDA price on {df_future.index[-1].date()}: "
        f"${df_future['Forecast_price'].iloc[-1]:.2f}"
    )

except ImportError:
    print(
        "\n[Section 13 skipped] TensorFlow / scikit-learn is not installed, "
        "so the LSTM forecast was not run.\n"
        "All other sections (fundamentals, technical analysis, CAPM) ran fine.\n"
        "To run the LSTM, install them on a supported Python version (3.10-3.13):\n"
        "    pip install tensorflow scikit-learn"
    )
