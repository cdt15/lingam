import numpy as np
import pandas as pd

n = 5
T = 1000
random_state = None

np.random.seed(random_state)

T_spurious = 20
expon = 1.5

value = np.random.uniform(low=0.05, high=0.5, size=(n, n))
sign = np.random.choice([-1, 1], size=(n, n))
B0 = np.multiply(value, sign)
B0 = np.multiply(B0, np.random.binomial(1, 0.4, size=(n, n)))
B0 = np.tril(B0, k=-1)

value = np.random.uniform(low=0.05, high=0.5, size=(n, n))
sign = np.random.choice([-1, 1], size=(n, n))
B1 = np.multiply(value, sign)
B1 = np.multiply(B1, np.random.binomial(1, 0.4, size=(n, n)))
M1 = np.dot(np.linalg.inv(np.eye(n) - B0), B1)

ee = np.empty((n, T + T_spurious))
for i in range(n):
    ee[i, :] = np.random.normal(size=(1, T + T_spurious))
    ee[i, :] = np.multiply(np.sign(ee[i, :]), abs(ee[i, :]) ** expon)
    ee[i, :] = ee[i, :] - np.mean(ee[i, :])
    ee[i, :] = ee[i, :] / np.std(ee[i, :])

std_e = np.random.uniform(size=(n,)) + 0.5
nn = np.dot(np.dot(np.linalg.inv(np.eye(n) - B0), np.diag(std_e)), ee)

xx = np.zeros((n, T + T_spurious))
xx[:, 0] = np.random.normal(size=(n,))

for t in range(1, T + T_spurious):
    xx[:, t] = np.dot(M1, xx[:, t - 1]) + nn[:, t]

data = xx[:, T_spurious + 1 : T_spurious + T]


df = pd.DataFrame(data.T, columns=["x0", "x1", "x2", "x3", "x4"])

df.to_csv("5vars_var.csv", index=False)
