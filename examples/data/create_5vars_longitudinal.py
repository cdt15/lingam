import numpy as np
import pandas as pd

np.random.seed(0)

n_features = 5
n_samples = 200
n_lags = 1
n_timepoints = 3

causal_orders = []
B_t_true = np.empty((n_timepoints, n_features, n_features))
B_tau_true = np.empty((n_timepoints, n_lags, n_features, n_features))
X_t = np.empty((n_timepoints, n_samples, n_features))

# B(0,0)
B_t_true[0] = np.array(
    [
        [0.0, 0.5, -0.3, 0.0, 0.0],
        [0.0, 0.0, -0.3, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, -0.7, 0.0, 0.0, 0.0],
    ]
)
causal_orders.append([3, 2, 1, 0, 4])

# B(1,1)
B_t_true[1] = np.array(
    [
        [0.0, 0.2, -0.1, 0.0, -0.5],
        [0.0, 0.0, 0.0, 0.4, 0.0],
        [0.0, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.4, 0.0, 0.0, 0.0],
    ]
)
causal_orders.append([3, 1, 2, 4, 0])

# B(2,2)
B_t_true[2] = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.7, 0.0, 0.5],
        [0.2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.4, 0.0, 0.0],
        [0.3, 0.0, 0.0, 0.0, 0.0],
    ]
)
causal_orders.append([0, 2, 4, 3, 1])

# create B(t,t-τ) and X
for t in range(n_timepoints):
    # external influence
    expon = 0.1
    ext = np.empty((n_features, n_samples))
    for i in range(n_features):
        ext[i, :] = np.random.normal(size=(1, n_samples))
        ext[i, :] = np.multiply(np.sign(ext[i, :]), abs(ext[i, :]) ** expon)
        ext[i, :] = ext[i, :] - np.mean(ext[i, :])
        ext[i, :] = ext[i, :] / np.std(ext[i, :])

    # create B(t,t-τ)
    for tau in range(n_lags):
        value = np.random.uniform(low=0.01, high=0.5, size=(n_features, n_features))
        sign = np.random.choice([-1, 1], size=(n_features, n_features))
        B_tau_true[t, tau] = np.multiply(value, sign)

    # create X(t)
    X = np.zeros((n_features, n_samples))
    for co in causal_orders[t]:
        X[co] = np.dot(B_t_true[t][co, :], X) + ext[co]
        if t > 0:
            for tau in range(n_lags):
                X[co] = X[co] + np.dot(B_tau_true[t, tau][co, :], X_t[t - (tau + 1)].T)

    X_t[t] = X.T

for t in range(n_timepoints):
    df = pd.DataFrame(X_t[t], columns=["x0", "x1", "x2", "x3", "x4"])
    df.to_csv(f"5vars_longitudinal_t{t}.csv", index=False)
