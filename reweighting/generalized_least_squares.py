"""
Positive weight calibration on a 50x50_000 toy data set -- compact and
readable.
"""

import numpy as np
from numpy.linalg import solve
from scipy.optimize import root

# Problem size and random seed -------------------------------------------
Q = 50            # number of targets
N = 50_000        # number of households
rng = np.random.default_rng(0)

# Metric matrix M (shape Q x N) ------------------------------------------
M = rng.normal(size=(Q, N))

# Simulate a target vector y ---------------------------------------------
y = np.random.lognormal(mean=10, sigma=2, size=50).round()

# The minimum-norm (possibly negative) solution based on the Moore-Penrose inverse
w_mp = np.linalg.pinv(M) @ y
print(w_mp)
print((M @ w_mp)[:5])
print(y[:5])
print("yup")

# Calibrate positive weights using solver --------------------------------
starting_w = rng.lognormal(mean=4.0, sigma=1.2, size=N)  # 1 to a few 1000

def g(theta):
    """Solve non-linear equation g(theta) = 0."""
    return M @ (starting_w * np.exp(M.T @ theta)) - y

theta_hat = root(g, x0=np.zeros(Q), method="hybr", tol=1e-10).x
w_cal = starting_w * np.exp(M.T @ theta_hat)
print(w_cal)
print((M @ w_cal)[:5])
print(y[:5])
print("yup")

# A second positive solution (null-space shift) --------------------------
# Note that this solution uses the w_cal solution we found above
z = rng.normal(size=N)                  # random direction
coef = solve(M @ M.T, M @ z)            # 50 x 50 system
z -= M.T @ coef                         # project into null space (M z ~ 0)

neg = z < 0
eps = 0.5 * np.min(w_cal[neg] / (-z[neg])) if np.any(neg) else 1.0
w_alt = w_cal + eps * z                 # stays strictly positive

print(w_alt)
print((M @ w_alt)[:5])
print(y[:5])
print("yup")
