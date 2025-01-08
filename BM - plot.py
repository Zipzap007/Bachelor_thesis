import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap


T = 1
N = 5000
r = 0.06
S = 40
sigma = 0.2
Omega = 5

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

data = np.array(BM(T,N,r,S,sigma,Omega))
x = np.linspace(0,T,N+1)
y = np.array(30*np.exp(x/1.8))

cmap = colormaps.get_cmap('cubehelix')


for i in range(Omega):
    clr = cmap(i/Omega)
    arr = data[i]
    indices = np.where(arr <= y)

    if indices[0].size > 0:
        index = indices[0][0]
    else:
        continue

    first_part_y = arr[:index + 1]
    first_part_x = x[:index + 1]
    second_part_y = arr[index + 1:]
    second_part_x = x[index + 1:]

    plt.plot(first_part_x, first_part_y, color=clr)
    plt.plot(second_part_x, second_part_y, color=clr, alpha=0.30)

plt.plot(x,30*np.exp(x/1.8))

plt.xlabel('t')
plt.ylabel('S')
plt.title('---')
plt.show()