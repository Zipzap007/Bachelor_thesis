import numpy as np
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.linear_model import LinearRegression

# Parameters
T = 1.0          # Total time (1 year)
N = 50         # Number of time steps
dt = T / N       # Time pr. step
r = 0.06      # Interest rate
S = 36         # Stock prise (t=0)
sigma = 0.2      # Volatility of returns
Omega = 100000    # Number of stock paths
K = 40          #Strike

#Here we simulate stock prices
data = np.zeros((Omega, N+1))  # Rows: Stocks, Columns: times

for omega in range(Omega):
    # Simulate the Brownian Motion
    Z = np.random.normal(0, np.sqrt(dt), size=N)
    x = np.zeros(N+1)
    x[0] = S
    for i in range(1, N+1):
        x[i] = x[i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * Z[i-1])
    data[omega, :] = x  # Store the stock's price in the data matrix

print(data)