import numpy as np
from array import array
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.linear_model import LinearRegression
from itertools import product


def BM(T, N, r, S, sigma, Omega):
    data = np.zeros((Omega, N + 1))
    dt = T / N
    for omega in range(Omega):
        # Simulate the Brownian Motion
        Z = np.random.normal(0, np.sqrt(dt), size=N)
        x = np.zeros(N + 1)
        x[0] = S
        for i in range(1, N + 1):
            x[i] = x[i - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * Z[i - 1])
        data[omega, :] = x  # Store the stock's price in the data matrix
    return data

def Lag_Pol(X):
    L_0 = np.exp(-X / 2)
    L_1 = np.exp(-X / 2) * (1 - X)
    L_2 = np.exp(-X / 2) * (1 - 2 * X + X ** 2 / 2)
    return np.column_stack((L_0, L_1, L_2))

def LSM_put(T, N, r, S, sigma, Omega, K):
    dt = T / N
    data = BM(T, N, r, S, sigma, Omega) / K  # Normalize data
    CFM = np.maximum(1 - data, 0)
    CFM[:, 0] = 0
    A_CFM = CFM.copy()

    for i in range(N, 1, -1):
        Y = np.exp(-r * dt) * A_CFM[:, i][CFM[:, i - 1] > 0]
        X = data[:, i - 1][CFM[:, i - 1] > 0]

        if Y.size == 0:
            continue

        Z = Lag_Pol(X)

        model = LinearRegression()
        model.fit(Z, Y)
        a = model.intercept_
        b, c, d = model.coef_

        Q = data[:, i - 1]

        L_0_Q = np.exp(-Q / 2)
        L_1_Q = np.exp(-Q / 2) * (1 - Q)
        L_2_Q = np.exp(-Q / 2) * (1 - 2 * Q + Q ** 2 / 2)

        E = a + b * L_0_Q + c * L_1_Q + d * L_2_Q

        for omega in range(Omega):
            if CFM[omega, i - 1] < E[omega]:
                CFM[omega, i - 1] = 0
                A_CFM[omega, i - 1] = A_CFM[omega, i] * np.exp(-r * dt)

    CFM__ = []
    for omega in range(Omega):
        non_zero_indices = np.nonzero(CFM[omega])[0]
        if non_zero_indices.size > 0:
            first_non_zero_idx = non_zero_indices[0]
            CFM__.append([CFM[omega, first_non_zero_idx], first_non_zero_idx])

    x = []
    for entry in CFM__:
        v, time = entry
        x.append(np.exp(-r * dt * time) * v * K)

    price = sum(x) / Omega
    se = np.sqrt(sum((x - price) ** 2)) / Omega

    return (price, se)

def CRR_put(T, N, r, S, sigma, K):
    dt = T/N
    u = np.exp( sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    R = np.exp(r*dt)
    p = (R - d) / (u - d)
    #print (u,d, R)

    X = np.array([max(0, K - (S * (d ** k * u ** (N-k)))) for k in range(N+1)])

    XX = X.copy()

    X = np.delete(X, -1)
    XX = np.delete(XX, 0)


    Y = np.array([max(0, K - (S * (d ** k * u ** (N-k)))) for k in range(N)])

    for i in range(N, 0, -1):

        ev = Y
        bdv = R ** -1 * (p * X + (1-p) * XX)
        val = np.maximum(bdv, ev)

        X_temp = X.copy()
        X = np.delete(val.copy(), -1)
        XX = np.delete(val.copy(), 0)
        Y = np.delete(X_temp,0)
    return(val[0])


S_values = [36, 38, 40, 42, 44]
sigma_values = [0.2, 0.4]
T_values = [1, 2]
K = 40
r = 0.06
N = 50
Omega = 100000

combinations = product(S_values, sigma_values, T_values)
results = []

for S, sigma, T in combinations:
    LSM = LSM_put(T, N, r, S, sigma, Omega, K)
    CRR = CRR_put(T, N, r, S, sigma, K)
    conf_int = ["%.4f"%(LSM[0]-1.96*LSM[1]), "%.4f"%(LSM[0]+1.96*LSM[1])]

    results.append((S, sigma, T, "%.2f"%LSM[0], "%.4f"%LSM[1], "%.2f"%conf_int, CRR))

df = pd.DataFrame(results, columns=['S', 'sigma', 'T', 'LSM price', 's.e.', '95% confidence interval','CRR price'])
