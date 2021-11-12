import numpy as np
import pandas as pd


def randnetbalanced(dims, samples, indegree, parminmax, errminmax):

    # First, generate errstd
    errstd = np.random.uniform(
        low=errminmax["min"], high=errminmax["max"], size=(dims, 1)
    )

    # Initializations
    X = np.empty(shape=[dims, samples])
    B = np.zeros([dims, dims])

    # Go trough each node in turn
    for i in range(dims):

        # If indegree is finite, randomly pick that many parents,
        # else, all previous variables are parents
        if indegree == float("inf"):
            if i <= indegree:
                par = np.arange(i)
            else:
                par = np.random.permutation(i)[:indegree]
        else:
            par = np.arange(i)

        if len(par) == 0:
            # if node has no parents
            # Increase errstd to get it to roughly same variance
            parent_std = np.random.uniform(low=parminmax["min"], high=parminmax["max"])
            errstd[i] = np.sqrt(errstd[i] ** 2 + parent_std ** 2)

            # Set data matrix to empty
            X[i] = np.zeros(samples)
        else:
            # If node has parents, do the following
            w = np.random.normal(size=[1, len(par)])

            # Randomly pick weights
            wfull = np.zeros([1, i])
            wfull[0, par] = w

            # Calculate contribution of parents
            X[i] = np.dot(wfull, X[:i, :])

            # Randomly select a 'parents std'
            parstd = np.random.uniform(low=parminmax["min"], high=parminmax["max"])

            # Scale w so that the combination of parents has 'parstd' std
            scaling = parstd / np.sqrt(np.mean(X[i] ** 2))
            w = w * scaling

            # Recalculate contribution of parents
            wfull = np.zeros([1, i])
            wfull[0, par] = w
            X[i] = np.dot(wfull, X[:i, :])

            # Fill in B
            B[i, par] = w

        # Update data matrix
        X[i] = X[i] + np.random.normal(size=samples) * errstd[i]

    return B, errstd


n = 5
T = 500
head = 100
T = T + head

# psi0
indegree = float("inf")
psi0, _ = randnetbalanced(
    n, n, indegree, {"min": 0.05, "max": 0.5}, {"min": 0.05, "max": 0.5}
)
permutation = np.random.permutation(n)
psi0 = psi0[permutation][:, permutation]

# causal order
causal_order = np.empty(len(permutation))
causal_order[permutation] = np.arange(len(permutation))
causal_order = causal_order.astype(int)

# phi1
value = np.random.uniform(low=0.01, high=0.5, size=(n, n))
sign = np.random.choice([-1, 1], size=(n, n))
phi1 = np.multiply(value, sign)

# theta1
value = np.random.uniform(low=0.01, high=0.5, size=(n, n))
sign = np.random.choice([-1, 1], size=(n, n))
theta1 = np.multiply(value, sign)

# psi1, omega1
psi1 = np.dot(np.eye(n) - psi0, phi1)
omega1 = np.dot(np.eye(n) - psi0, theta1, np.linalg.inv(np.eye(n) - psi0))

# external influence
expon = 0.1
ext = np.empty((n, T))
for i in range(n):
    ext[i, :] = np.random.normal(size=(1, T))
    ext[i, :] = np.multiply(np.sign(ext[i, :]), abs(ext[i, :]) ** expon)
    ext[i, :] = ext[i, :] - np.mean(ext[i, :])
    ext[i, :] = ext[i, :] / np.std(ext[i, :])

# observed signals y
y = np.zeros((n, T))
y[:, 0] = np.random.normal(loc=0.1, scale=1, size=(n,)) * np.random.choice(
    [-1, 1], size=(n,)
)
for t in range(1, T):
    for i in causal_order:
        y[i, t] = (
            np.dot(psi0[i, :], y[:, t])
            + np.dot(psi1[i, :], y[:, t - 1])
            + ext[i, t]
            + np.dot(omega1[i, :], ext[:, t - 1])
        )

df = pd.DataFrame(y[:, head:].T, columns=["x0", "x1", "x2", "x3", "x4"])

df.to_csv("5vars_varma.csv", index=False)
