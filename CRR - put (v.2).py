import time

import numpy as np
from array import array
import matplotlib.pyplot as plt

t0 = time.time()
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


#CRR(1,2,0.06,36, 0.2,40)

print(CRR(1,10000,0.06,36, 0.2,40))
t1 = time.time()
print(t1-t0)

nn = 10000
mm = int(nn / 500)

X_points = np.array(np.linspace(1, nn , int(nn / mm)))
Y_points = np.array([CRR(1, (i + 1) * mm, 0.06, 36, 0.2, 40) for i in range(int(nn / mm))])

plt.plot(X_points, Y_points)
plt.show()