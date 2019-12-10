import numpy as np

load = np.array([150, 200, 250, 180])  # MW each hour
p_g = np.array([20, 25, 30, 22])  # $/MWh each hour
p_r = p_g + np.array([2, 3, 4, 2])  # $/MW each hour
p_n = p_g + np.array([1, 2, 3, 3])  # $/MW each hour

ramprate = np.array([120, 60, 60])  # MW/h each unit
quickstart = np.array([0, 20, 20])  # MW each unit
minOn = np.array([2, 2, 1])  # Minimum on time in hours
minOff = np.array([2, 1, 1])  # Minimum off time in hours
initialState = np.array([1, 0, 0])  # (1 means it starts on)
initialHour = np.array([4, 2, 1])  # hours it has been in the initial state
state = np.array([])
Pgmin = np.array([50, 20, 20])  # MW each unit
Pgmax = np.array([200, 100, 100])  # MW each unit
p_fuel = np.array([2, 2, 2.5])  # $/MBtu each unit
startup = p_fuel*np.array([200, 25, 25])  # $ to start the unit
shutdown = np.array([0, 0, 0])  # $ to shutdown unit
a = p_fuel*np.array([400, 25, 25])  # each unit
b = p_fuel*np.array([8, 10, 10])  # each unit
c = p_fuel*np.array([0.01, 0.025, 0.02])  # each unit
P0 = np.array([
    [0, 0, 0, 0],
    [30, 30, 30, 30],
    [0, 0, 0, 0]])  # MW each hour
k = 1  # markup percent for bilateral contract
# Contract price for gen2
f_bl = k*(np.array([a]).T*np.array([[0], [1], [0]]) + np.array([b]).T*P0 + np.array([c]).T*P0**2)
lg = 0
lr = 0
ln = 0
nhours = 4
cost_on = np.zeros(nhours)
cost_off = np.zeros(nhours)
i = 2 - 1
r = np.zeros(nhours)
n = np.zeros(nhours)
p = np.zeros(nhours)
L = np.zeros(nhours)
n_off = np.zeros(nhours)
for t in range(nhours):
    # Unit On
    if p_r[t] > p_n[t]:
        Rmin = 0
        Rmax = ramprate[i] / 60 * 10  # ten minute ramp capacity
        r[t] = (p_r[t] - lr - b[i]) / (2 * c[i]) - Pgmin[i]
        if r[t] > Rmax:
            r[t] = Rmax

        if r[t] < Rmin:
            r[t] = Rmin

        Nmin = 0
        Nmax = ramprate[i] / 60 * 10 - r[t]
        n[t] = (p_n[t] - lr - b[i]) / (2 * c[i]) - Pgmin[i] - r[t]
        if n[t] > Nmax:
            n[t] = Nmax

        if n[t] < Nmin:
            n[t] = Nmin

    if p_n[t] > p_r[t]:
        Nmin = 0
        Nmax = ramprate[i] / 60 * 10
        n[t] = (p_n[t] - lr - b[i]) / (2 * c[i]) - Pgmin[i]
        if n[t] > Nmax:
            n[t] = Nmax

        if n[t] < Nmin:
            n[t] = Nmin

        Rmin = 0
        Rmax = ramprate[i] / 60 * 10 - n[t]
        r[t] = (p_r[t] - lr - b[i]) / (2 * c[i]) - Pgmin[i] - n[t]
        if r[t] > Rmax:
            r[t] = Rmax

        if r[t] < Rmin:
            r[t] = Rmin

    Pmin = Pgmin[i]
    Pmax = Pgmax[i] - n[t] - r[t]
    p[t] = (p_g[t] - lr - b[i]) / (2 * c[i]) - r[t] - n[t]
    if p[t] > Pmax:
        p[t] = Pmax

    if p[t] < Pmin:
        p[t] = Pmin

    Cost = a[i] + b[i] * (p[t] + n[t] + r[t]) + c[i] * (p[t] + n[t] + r[t]) ** 2 - f_bl[i, t]
    L[t] = (-p_g[t] * (p[t] - P0[i, t]) + lg * p[t] - p_r[t] * r[t] + lr * r[t] - p_n[t] * n[t] + ln * n[t]) + Cost
    cost_on[t] = L[t]

    # Unit Off
    Noffmin = 0
    Noffmax = quickstart[i]
    n_off[t] = (p_n[t] - lr - b[i]) / (2 * c[i])
    if n_off[t] > Noffmax:
        n_off[t] = Noffmax

    if n_off[t] < Noffmin:
        n_off[t] = Noffmin

    if n_off[t] == 0:
        B = P0[i, t]
        cost_off[t] = -f_bl[i, t] + p_g[t] * B
    else:
        B = P0[i, t]
        Cost = a[i] + b[i] * (n_off[t]) + c[i] * (n_off[t]) ** 2 + p_g[t] * B
        cost_off[t] = - p_n[t] * n_off[t] + ln * n_off[t] - f_bl[i, t] + Cost

print('L: ', L)
print('r: ', r)
print('n: ', n)
print('p: ', p)
print('n_off: ', n_off)
"""
L:  [ 182.5   77.5 -102.5  142.5]
r:  [ 0. 10. 10.  0.]
n:  [0. 0. 0. 0.]
p:  [20. 20. 40. 20.]
n_off:  [ 0. 20. 20.  0.]
"""