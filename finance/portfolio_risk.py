# %%
import numpy as np
import matplotlib.pyplot as plt
import data_visualization as dv
import polars as pl
import yfinance as yf

plt.ion()
dv.set_plot_style("cashaback_dark.mplstyle")
# %% Import data
symbols = [
    "MSFT",
    "AAPL",
    "NVDA",
    # "META",
    # "GOOG",
    # "AMZN",
    # "LLY",
    # "T",
    # "VZ",
    # "KEY",
    # "HBAN",
]
og_df = yf.download(symbols, "2019-01-01")  # Download all tickers
if og_df is None:
    raise ValueError("No data downloaded from yfinance.")

# %% Reorganize dataframe from multi-index to long format
# Stack to long format
og_dff = og_df.stack(level=1).reset_index()
print(og_dff.head())
df = pl.DataFrame(og_dff)
total_days = len(df.filter(pl.col("Ticker") == symbols[0]))

# %%
fig, ax = plt.subplots()
for ticker in symbols:
    ticker_data = df.filter(pl.col("Ticker") == ticker)
    ax.plot(ticker_data["Date"], ticker_data["Close"], label=ticker)
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
plt.show()

# %% Calculate daily returns, volatility, and covariance btwn tickers
# Daily returns is just the percentage change in close price from one day to the next
df = df.with_columns(
    daily_return=pl.col("Close").pct_change().over("Ticker"),
).drop_nulls()  # Drop first row that has null daily return
# We want volatitlity, so first get the std of daily returns
df = df.with_columns(
    std_daily_return=pl.col("daily_return").std().over("Ticker"),
)
# Covariance matrix of daily returns
cov_matrix = np.cov(
    df.select(pl.col("Ticker", "daily_return", "Date"))
    # Rows are daily_return, columns are tickers
    .pivot(values="daily_return", index="Date", on="Ticker")
    # Get rid of date column so I can do the covariance on the floats, over time
    .drop("Date")
    .to_numpy(),
    # rows are NOT variables, columns are, so set rowvar to False
    rowvar=False,
)
print("Covariance matrix of daily returns:")
print(cov_matrix)
# Average daily returns and volatitily per ticker
summary_df = df.group_by(pl.col("Ticker"), maintain_order=True).agg(
    pl.col("daily_return").mean().alias("avg_daily_return"),
    pl.col("std_daily_return").mean().alias("volatility"),
)
# Calculate predicted annual return and annual volatility
summary_df = summary_df.with_columns(
    annual_return=pl.col("avg_daily_return").add(1).pow(252).sub(1),
    annual_volatility=pl.col("volatility").mul(np.sqrt(252)),
)
print("Summary statistics per ticker:")
print(summary_df)


# %%% Monte Carlo optimization of portfolio weights
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
weights = np.zeros((num_portfolios, len(symbols)))
for i in range(num_portfolios):
    # Randomly assign weights to each ticker
    weights[i] = np.random.random(len(symbols))
    weights[i] /= np.sum(weights[i])  # Normalize to sum to 1

    # Calculate portfolio return and volatility
    portfolio_return = np.sum(
        weights[i] * summary_df.select("avg_daily_return").to_numpy().flatten()
    )
    # Double sum of w_i * w_j * cov(i,j) is just
    portfolio_volatility = np.sqrt(
        np.dot(
            weights[i].T,
            np.dot(
                cov_matrix * 252,  # Annualize covariance matrix
                weights[i],
            ),
        )
    )
    # Store results
    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = results[0, i] / results[1, i]  # Sharpe ratio
max_sharpe_idx = np.argmax(results[2, :])
max_return_idx = np.argmax(results[0, :])
min_volatility_idx = np.argmin(results[1, :])
# %% Plot results
# Create scatter plot of volatility vs return
fig, ax = plt.subplots()
sc = ax.scatter(
    results[1, :],
    results[0, :],
    c=results[2, :],
    cmap="viridis",
)
# Max retrun
ax.scatter(
    results[1, max_return_idx],
    results[0, max_return_idx],
    label="max return",
    marker="x",
    color="white",
)
# Min volatility
ax.scatter(
    results[1, min_volatility_idx],
    results[0, min_volatility_idx],
    label="min volatility",
    marker="x",
    color="white",
)
# Max sharpe
ax.scatter(
    results[1, max_sharpe_idx],
    results[0, max_sharpe_idx],
    label="min volatility",
    marker="x",
    color="white",
)

ax.set_xlabel("Volatility")
ax.set_ylabel("Return")
plt.colorbar(sc, label="Sharpe Ratio")
plt.show()

# %% Show optimal portfolio
max_sharpe_weights = weights[max_sharpe_idx]
max_return_weights = weights[max_return_idx]
min_volatility_weight = weights[min_volatility_idx]
fig = dv.AutoFigure("a;b;c")
axes = list(fig.axes.values())
(ax1, ax2, ax3) = axes
ax1.bar(symbols, max_return_weights)
ax2.bar(symbols, max_sharpe_weights)
ax3.bar(symbols, min_volatility_weight)

ax1.text(
    0.02, 1, f"Return : {results[0, max_return_idx].round(8)}", transform=ax1.transAxes
)
ax1.text(
    0.02, 0.88, f"Risk : {results[1, max_return_idx].round(8)}", transform=ax1.transAxes
)
ax2.text(
    0.02, 1, f"Return : {results[0, max_sharpe_idx].round(8)}", transform=ax2.transAxes
)
ax2.text(
    0.02, 0.88, f"Risk : {results[1, max_sharpe_idx].round(8)}", transform=ax2.transAxes
)
ax3.text(
    0.02,
    1,
    f"Return : {results[0, min_volatility_idx].round(8)}",
    transform=ax3.transAxes,
)
ax3.text(
    0.02,
    0.88,
    f"Risk : {results[1, min_volatility_idx].round(8)}",
    transform=ax3.transAxes,
)
ax1.set_title("Max Return Weights")
ax2.set_title("Max Sharpe Weights")
ax3.set_title("Min Volatility Weights")
[ax.set_ylabel("Weight") for ax in axes]
[ax.set_ylim(0, 0.5) for ax in axes]

plt.show()


# %% Create modeldf, one row per ticker
model_df = (
    df.drop_nulls()
    .group_by(pl.col("Ticker"), maintain_order=True)
    .agg(
        [
            pl.col("daily_return").mean().mul(252).alias("drift"),
            pl.col("std_daily_return").mean().mul(np.sqrt(252)).alias("volatility"),
        ]
    )
)
print(model_df)


# %% DDM model of the price
def dXdt(current_price, drift, volatility):
    """
    Differential equation of the drift-diffusion model
    dX/dt = mu + sigma*N(0,1)
    """

    return current_price * np.exp(drift + volatility * np.random.normal())


dt = 1
timesteps = int(total_days / dt)
company = "T"
X0 = df.filter(pl.col("Ticker") == company).select("Close")[0].item()
volatility = model_df.filter(pl.col("Ticker") == company).select("volatility").item()
drift = model_df.filter(pl.col("Ticker") == company).select("drift").item()
ddm_prediction = np.zeros(timesteps)
ddm_prediction[0] = X0
for i in range(total_days - 1):
    print(ddm_prediction[i])
    ddm_prediction[i + 1] = ddm_prediction[i] + dt * dXdt(drift, volatility)

# %% Plot model with true values
fig, ax = plt.subplots()
dff = df.filter(pl.col("Ticker") == company)
model_dff = model_df.filter(pl.col("Ticker") == company)
xvals = np.arange(0, total_days, 1)
model_price = model_dff["drift"] * xvals
ax.plot(ddm_prediction)
# ax.plot(xvals, model_price)
ax.plot(xvals, dff["Close"])
plt.show()
