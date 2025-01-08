import numpy as np
from array import array
import matplotlib.pyplot as plt
import time

t0 = time.time()

def CRR(T, N, r, S, sigma, K):
    dt = T/N
    u = np.exp( sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    R = np.exp(r*dt)
    p = (R - d) / (u - d)
    #print (u,d, R)

    X = []
    for k in range(N+1):
        v = max(0, K - (S * (d ** k * u ** (N-k))))
        X.append(v)
    #print(f"XX={X}")
    for i in range(N, 0, -1):
        Y = []
        #print(f"i={i}")
        for j in range(i):
            bdv = 1 / R * (p * X[j] + (1 - p) * X[j+1])
            ev = max( 0, (K - S * (d ** j * u ** (i-j))) )

            Y.append(max(bdv, ev))
            #print(f"bdv={bdv}, ev={ev}, Y= {Y}")
        X = Y
        #print(f"X={X}")
    return(X)

CRR(1,10000,0.06,36, 0.2,40)
t1 = time.time()

print(t1-t0)

#print(CRR(1,100,0.06,36, 0.2,40))

#nn = 1000
#mm = int(nn / 20)

#Y_points = np.array([CRR(1, (i + 1) * mm, 0.06, 36, 0.2, 40)[0] for i in range(int(nn / mm))])
#Y_points = Y_points.ravel()
#X_points = np.array(np.linspace(1, nn , int(nn / mm)))

#print(Y_points, X_points)
#plt.show()

#make a graf where we see how i converges for N -> infty