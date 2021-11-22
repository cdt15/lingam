import numpy as np
import pandas as pd

N_SAMPLES = 300

np.random.seed(0)

get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3

x5 = get_external_effect(N_SAMPLES)
x6 = get_external_effect(N_SAMPLES)
x1 = 0.6 * x5 + get_external_effect(N_SAMPLES)
x3 = 0.5 * x5 + get_external_effect(N_SAMPLES)
x0 = 1.0 * x1 + 1.0 * x3 + get_external_effect(N_SAMPLES)
x2 = 0.8 * x0 - 0.6 * x6 + get_external_effect(N_SAMPLES)
x4 = 1.0 * x0 - 0.5 * x6 + get_external_effect(N_SAMPLES)

# The latent variable x6 is not included.
df = pd.DataFrame(
    np.array([x0, x1, x2, x3, x4, x5]).T, columns=["x0", "x1", "x2", "x3", "x4", "x5"]
)

df.to_csv("6vars_mid_latent.csv", index=False)
