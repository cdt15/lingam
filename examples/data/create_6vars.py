import numpy as np
import pandas as pd

N_SAMPLES = 1000

np.random.seed(23)

x3 = np.random.uniform(size=N_SAMPLES)
x0 = 3.0 * x3 + np.random.uniform(size=N_SAMPLES)
x2 = 6.0 * x3 + np.random.uniform(size=N_SAMPLES)
x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=N_SAMPLES)
x5 = 4.0 * x0 + np.random.uniform(size=N_SAMPLES)
x4 = 8.0 * x0 - 1.0 * x2 + np.random.uniform(size=N_SAMPLES)
df = pd.DataFrame(
    np.array([x0, x1, x2, x3, x4, x5]).T, columns=["x0", "x1", "x2", "x3", "x4", "x5"]
)
df.to_csv("6vars.csv", index=False)
