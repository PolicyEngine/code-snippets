# The goal was to see how well QRF did with a heavily zero-inflated distribution
# The answer is: pretty well!

import pandas as pd
import numpy as np
from scipy import stats

from policyengine_us_data.utils import QRF

# What is the proportion of zeros before things fall apart?
zero_proportion = .95

# Other data generating process parameters
n_samples = 1000  # will be used for both training and holdout sets
lognormal_mu_intercept = 10
lognormal_mu_coef_x1 = 1
lognormal_mu_coef_x2 = .5
lognormal_sigma = .5

X1 = np.random.normal(loc=0, scale=1, size=n_samples)
X2 = np.random.normal(loc=0, scale=1, size=n_samples)

is_zero = np.random.binomial(n=1, p=zero_proportion, size=n_samples).astype(bool)
n_nonzero = n_samples - np.sum(is_zero)
y_nonzero = np.zeros(n_nonzero)

if n_nonzero > 0:
    X1_nonzero = X1[~is_zero]
    X2_nonzero = X2[~is_zero]

    mu_log_nonzero = (lognormal_mu_intercept +
                      lognormal_mu_coef_x1 * X1_nonzero +
                      lognormal_mu_coef_x2 * X2_nonzero)

    y_nonzero = stats.lognorm.rvs(s=lognormal_sigma,
                                  scale=np.exp(mu_log_nonzero),
                                  size=n_nonzero)

y = np.zeros(n_samples)
y[~is_zero] = y_nonzero

df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})

X_train = df[['X1', 'X2']]
y_train = df[['y']]

model = QRF()
model.fit(
    X_train, 
    df[['y']]
)

y_pred = model.predict(X_train)

# Out of sample
X1_a = np.random.normal(loc=0, scale=1, size=n_samples)
X2_a = np.random.normal(loc=0, scale=1, size=n_samples)
X_train_a = pd.DataFrame({'X1': X1_a, 'X2': X2_a})
y_pred_a = model.predict(X_train_a)


print(f"\n\n---- Analysis of zeros imputed---------\n")
print(f"Zeros in actual values make up {100 * np.mean(y_train < 1E-6)}%")
print(f"Zeros in in-sample predicted values make up {100 * np.mean(y_pred < 1E-6)}%")
print(f"Zeros in out-of-sample predicted values make up {100 * np.mean(y_pred_a < 1E-6):.1f}%")

