

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_stock_data(ticker, start, end):
    """Download daily close prices via yfinance."""
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, progress=False)
    df = raw[["Close"]].copy()
    df.columns = ["price"]
    df.index.name = "date"
    return df


def generate_demo_data(n_days=1200, seed=42):
    """Synthetic price series with trend + weekly wiggle + noise.
    Lets you run the whole pipeline with zero internet access."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    t = np.arange(n_days)

    trend = 100 + 0.05 * t + 15 * np.sin(t / 250)      # slow drift + long cycle
    weekly_effect = 2 * np.sin(2 * np.pi * t / 5)        # day-of-week wiggle
    noise = np.cumsum(rng.normal(0, 1, n_days))          # random-walk noise
    price = np.maximum(trend + weekly_effect + noise, 1)

    df = pd.DataFrame({"price": price}, index=dates)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# 2. Exploratory analysis
# ---------------------------------------------------------------------------
def plot_price_overview(df, ticker, outdir="."):
    df["MA50"] = df["price"].rolling(50).mean()
    df["MA200"] = df["price"].rolling(200).mean()
    df["log_return"] = np.log(df["price"]).diff()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df.index, df["price"], label="Close", linewidth=1)
    axes[0].plot(df.index, df["MA50"], label="50-day MA", linewidth=1)
    axes[0].plot(df.index, df["MA200"], label="200-day MA", linewidth=1)
    axes[0].set_title(f"{ticker} Price with Moving Averages")
    axes[0].legend()
    axes[0].set_ylabel("Price")

    axes[1].plot(df.index, df["log_return"], linewidth=0.7, color="grey")
    axes[1].set_title("Daily Log Returns (a proxy for volatility)")
    axes[1].set_ylabel("Log return")

    plt.tight_layout()
    fig.savefig(f"{outdir}/{ticker}_price_overview.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Trend / seasonal / residual decomposition (STL)
# ---------------------------------------------------------------------------
def decompose_trend(df, ticker, period=5, outdir="."):
    """STL decomposition. period=5 treats the trading week as the cycle."""
    stl = STL(df["price"], period=period, robust=True)
    result = stl.fit()

    fig = result.plot()
    fig.set_size_inches(10, 8)
    fig.suptitle(f"{ticker} STL Decomposition (trend / seasonal / residual)", y=1.02)
    fig.savefig(f"{outdir}/{ticker}_stl_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return result


# ---------------------------------------------------------------------------
# 4. ARIMA — model short-term dynamics around the trend
# ---------------------------------------------------------------------------
def check_stationarity(series, label="series"):
    result = adfuller(series.dropna())
    print(f"ADF test on {label}: statistic={result[0]:.3f}, p-value={result[1]:.4f}")
    if result[1] < 0.05:
        print("  -> Likely stationary (reject the unit-root null)")
    else:
        print("  -> Likely non-stationary (typical for raw prices)")
    return result[1]


def fit_arima(df, ticker, forecast_days=15, outdir="."):
    price = df["price"]

    print("\n--- Stationarity check ---")
    check_stationarity(price, "raw price")
    check_stationarity(price.diff(), "first difference")

    # order=(p,d,q); d=1 handles the unit root that raw prices usually have
    model = ARIMA(price, order=(5, 1, 0))
    fitted = model.fit()
    print(fitted.summary())

    forecast = fitted.get_forecast(steps=forecast_days)
    forecast_index = pd.bdate_range(price.index[-1], periods=forecast_days + 1)[1:]
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(price.index[-150:], price.values[-150:], label="Observed")
    ax.plot(forecast_index, mean_forecast, label="ARIMA extrapolation", color="orange")
    ax.fill_between(
        forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
        color="orange", alpha=0.2, label="95% CI",
    )
    ax.set_title(
        f"{ticker} ARIMA(5,1,0) — short-horizon extrapolation\n"
        "(illustrative only, note how fast the interval widens — not a trading signal)"
    )
    ax.legend()
    fig.savefig(f"{outdir}/{ticker}_arima_forecast.png", dpi=150)
    plt.close(fig)

    return fitted


# ---------------------------------------------------------------------------
# 5. Prophet — trend + seasonality with built-in uncertainty bands
# ---------------------------------------------------------------------------
def fit_prophet(df, ticker, forecast_days=30, outdir="."):
    from prophet import Prophet

    prophet_df = df.reset_index()[["date", "price"]].rename(
        columns={"date": "ds", "price": "y"}
    )

    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)

    fig1 = m.plot(forecast)
    fig1.gca().set_title(f"{ticker} Prophet Forecast")
    fig1.savefig(f"{outdir}/{ticker}_prophet_forecast.png", dpi=150)
    plt.close(fig1)

    fig2 = m.plot_components(forecast)
    fig2.savefig(f"{outdir}/{ticker}_prophet_components.png", dpi=150)
    plt.close(fig2)

    return forecast


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stock price trend analysis")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--demo", action="store_true", help="use synthetic data, no internet needed")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--skip-prophet", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.demo:
        print("Using synthetic demo data (no internet required)...")
        df = generate_demo_data()
        ticker = "DEMO"
    else:
        print(f"Downloading {args.ticker} data...")
        df = load_stock_data(args.ticker, args.start, args.end)
        ticker = args.ticker

    print("\n1) Plotting price overview (price, moving averages, returns)...")
    plot_price_overview(df, ticker, args.outdir)

    print("2) Running STL decomposition (trend / seasonal / residual)...")
    decompose_trend(df, ticker, outdir=args.outdir)

    print("3) Fitting ARIMA model...")
    fit_arima(df, ticker, outdir=args.outdir)

    if not args.skip_prophet:
        print("4) Fitting Prophet model...")
        try:
            fit_prophet(df, ticker, outdir=args.outdir)
        except ImportError:
            print("   Prophet not installed. Run `pip install prophet` or pass --skip-prophet.")

    print(f"\nDone! Plots saved to '{args.outdir}/'")


if __name__ == "__main__":
    main()
