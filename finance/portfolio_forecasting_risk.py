# %%
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import data_visualization as dv
import polars as pl
from src.dataloader import get_stock_df
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.ion()
dv.set_plot_style("cashaback_dark.mplstyle")

# %% Get some stock data
symbols = [
    "MSFT",
]
df = get_stock_df(symbols=symbols, start_date="2019-01-01")
total_days = len(df.filter(pl.col("Ticker") == symbols[0]))

# %% plot it
fig, ax = plt.subplots()
for ticker in symbols:
    ticker_data = df.filter(pl.col("Ticker") == ticker)
    ax.plot(ticker_data["Date"], ticker_data["Close"], label=ticker)
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
plt.show()

# %% Check Stationaryity with ADF test
# This checks the null hypothesis that the data is not stationary
# So, if p > 0.05, then it's likely not stationary
# If p < 0.05, then we reject the null, and can assume it's stationary

adf_result = adfuller(df["Close"])
assert len(adf_result) >= 6  # so type checker doesn't complain
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value}")

# %% Visual inspection of stationarity with rolling mean and rolling std
rolling_mean = df["Close"].rolling_mean(window_size=100)
rolling_std = df["Close"].rolling_std(window_size=100)
fig, ax = plt.subplots()
ax.plot(df["Close"], label="original")
ax.plot(rolling_mean, label="rolling_mean")
ax.plot(rolling_std, label="rolling_std")
ax.legend()
plt.show()

# %% Get differenced data to make stationary
# Differenced data w/o log10 seems to have variance grow over time, so taking log
difference_data = df["Close"].log10().diff().drop_nulls().to_numpy()
# Looks pretty good, except for two big spikes
fig, ax = plt.subplots()
ax.plot(difference_data)
plt.show()


# %% AutoRegressive Moving Average (ARMA) Model
def AR(data, i, p):
    """AutoRgressive Function

    Args:
        data (np.ndarray): Time series data
        i (int): current timepoint
        p (int): number of timepoints to look back on to predict next i
    """

    # FOr now, phi are gonna be 0.5 for all i
    phis = 0.5
    return np.sum(phis * data[i - p : i])


def MA(errors, q, cov):
    thetas = 0.1
    noise = np.random.normal(0, cov, q)
    return np.sum(thetas * noise)


def loglike(n, sigma, residuals):
    ll = (
        -n / 2 * np.log(2 * np.pi)
        - n / 2 * np.log(sigma**2)
        - 1 / (2 * sigma**2) * np.sum(residuals**2)
    )
    return ll


## Plot ACF and PACF to establish p and q respectively
plot_acf(difference_data)
plot_pacf(difference_data)

# %%
p = 1  # ACF not really there, so just gonan use 1
q = 2  # PACF drops to 0 after 2
ar_values = np.empty(len(difference_data) - p)
for i in range(p, len(difference_data)):
    ar_values[i - p] = AR(difference_data, i, p)


cutoff_date = datetime.strptime("2023-01-01", "%Y-%m-%d")
train_df = df.filter(pl.col("Date") <= cutoff_date)
test_df = df.filter(pl.col("Date") > cutoff_date)
n_train = len(train_df)
n_test = len(test_df)

# [TODO] Figure out how to fit the parameters using MLE
