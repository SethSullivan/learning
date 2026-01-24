# %%
import numpy as np
import polars as pl
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %%
# Import data
df = pl.read_csv("./jp_quant_research/data/Nat_Gas.csv")

# %%
# display(df)

# %% Create model
# Linear portion
num_points = df.height
x = np.arange(num_points)
y = df["Prices"].to_numpy()
# Fit linear model to data
linear_coefficients = np.polyfit(x, y, 1)
polynomial = np.poly1d(linear_coefficients)
# Generate fitted values with projected prices
projected_x = np.arange(0, num_points + 12)
projected_prices = polynomial(projected_x)
print("Linear model linear_coefficients:", linear_coefficients)


# Fit sin model
def loss(params, x, y):
    A, B = params
    prediction = A * np.sin(x / B)
    loss = np.sqrt(np.mean((y - prediction) ** 2))
    return loss


def linear_plus_sin(x, A, B, M, C):
    return A * np.sin(x / B) + M * x + C


out = curve_fit(
    linear_plus_sin,
    xdata=x,
    ydata=y,
    full_output=True,
    p0=[1, 2, linear_coefficients[0], linear_coefficients[1]],
)
print(out[0])
# Combine linear and sin model

combined_model = linear_plus_sin(x, *out[0])

# %%
fig, ax = plt.subplots(dpi=300)
ax.plot(df["Dates"], df["Prices"])
ax.plot(x, combined_model, color="red")
ax.set_xticks(df["Dates"])
ax.set_xticklabels(df["Dates"], rotation=90, fontsize=6)

plt.show()
