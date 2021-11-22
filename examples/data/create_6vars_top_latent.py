import numpy as np
import pandas as pd

N_SAMPLES = 1000

np.random.seed(1000)

x6 = np.random.uniform(size=N_SAMPLES)
x3 = 2.0 * x6 + np.random.uniform(size=N_SAMPLES)
x0 = 0.5 * x3 + np.random.uniform(size=N_SAMPLES)
x2 = 2.0 * x6 + np.random.uniform(size=N_SAMPLES)
x1 = 0.5 * x0 + 0.5 * x2 + np.random.uniform(size=N_SAMPLES)
x5 = 0.5 * x0 + np.random.uniform(size=N_SAMPLES)
x4 = 0.5 * x0 - 0.5 * x2 + np.random.uniform(size=N_SAMPLES)

# The latent variable x6 is not included.
df = pd.DataFrame(
    np.array([x0, x1, x2, x3, x4, x5]).T, columns=["x0", "x1", "x2", "x3", "x4", "x5"]
)

df.to_csv("6vars_top_latent.csv", index=False)
