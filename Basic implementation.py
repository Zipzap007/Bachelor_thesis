import numpy as np
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.linear_model import LinearRegression

# Parameters
T = 1         # Total time (1 year)
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

data = data / K #Normalize data
CFM = np.maximum(1 - data, 0)
CFM[:,0] = 0 #Not excersizable at time 0, so the value here will be 0
#print('CFM:', CFM)
#print('data:', data)

A_CFM = CFM

for i in range(N, 1, -1):
    Y = np.exp(-r * dt) * A_CFM[:, i][CFM[:, i-1] > 0]
    X = data[:, i-1][CFM[:, i-1] > 0]
    #print('Y:', Y)
    #print('X:', X)

    # Skip iteration if no valid samples are available
    if Y.size == 0:
        print(f"Skipping iteration {i}: No valid samples.")
        continue

    #Define basis to do regression on
    L_0 = np.exp(-X/2)
    L_1 = np.exp(-X/2) * (1 - X)
    L_2 = np.exp(-X/2) * (1 - 2*X + X**2/2)
    Z = np.column_stack((L_0, L_1, L_2))

    #print('Z:', Z)

    #Regression
    model = LinearRegression()
    model.fit(Z, Y)
    a = model.intercept_
    b, c, d = model.coef_
    #print(f'Iteration {i}: a={a}, b={b}, c={c}, d={d}')

    Q = data[:, i-1]

    L_0_D = np.exp(-Q / 2)
    L_1_D = np.exp(-Q / 2) * (1 - Q)
    L_2_D = np.exp(-Q / 2) * (1 - 2 * Q + Q ** 2 / 2)

    E = a + b*L_0_D + c*L_1_D + d*L_2_D

    for j in range(Omega):
        if CFM[j, i-1] < E[j]:
            CFM[j, i-1] = 0
            A_CFM[j, i - 1] = CFM[j, i] * np.exp(-r * dt)


    #print('CFM:', CFM)
    #print('E:', E)
    #print('------------------------------------------')

CFM = A_CFM
CFM_ = np.zeros_like(CFM)
CFM__ = []

#This will modify CFM array st. there only is one cashflow for all omega in Omega
for omega in range(Omega):
    non_zero_indices = np.nonzero(CFM[omega])[0]
    if non_zero_indices.size > 0:
        first_non_zero_idx = non_zero_indices[0]
        CFM_[omega, first_non_zero_idx] = CFM[omega, first_non_zero_idx]

#We can also get a Matrix, which gives us the first cashflow and timepoint for
for i in range(Omega):
    non_zero_indices = np.nonzero(CFM[i])[0]
    if non_zero_indices.size > 0:
        first_non_zero_idx = non_zero_indices[0]
        CFM__.append([CFM[i, first_non_zero_idx], first_non_zero_idx])
    else:
        continue

#Check to see if data is correct
#print(CFM)
#print(CFM_)
#print(CFM__)

x = 0
for i in range(len(CFM__)):
    v = CFM__[i][0] * K
    time = CFM__[i][1]
    x += np.exp(-r * dt * time) * v / Omega


print('American Option price:', x)
