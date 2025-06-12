import numpy as np
import pandas as pd


def simulate_contradictory_data(
    T: float,  # official grand total
    k: int,   # number of strata
    c: float,  # contradiction factor. Strata totals fall short by floor(c * T)
    N: int,  # Population size
    n: int,  # Sample size, over all strata
    dirichlet_alpha = 5.0,  # low value creates uneven divisions
    gamma_shape: float = 2.0,  
    gamma_scale: float = 2.0
):

    """Returns tuple containing the sample DataFrame and the official totals dict."""
    # 1. Simulate stratum proportions (thetas)
    alphas = np.full(k, dirichlet_alpha)
    thetas = np.random.dirichlet(alphas)

    # 2. Define "latent" subtotals S_i
    S_i = thetas * T

    # 3. Calculate and distribute the contradiction 'delta'
    delta = np.floor(c * T)
    p = np.full(k, 1/k)
    delta_i = np.random.multinomial(n=int(delta), pvals=p)

    # 4. Calculate final, contradictory official subtotals S_i^*
    S_i_star = S_i - delta_i

    # 5. Allocate population size N into strata sizes N_i
    N_i_float = thetas * N
    N_i = N_i_float.astype(int)
    remainder_N = N - N_i.sum()
    if remainder_N > 0:
        top_indices_N = np.argsort(N_i_float - N_i)[-remainder_N:]
        N_i[top_indices_N] += 1
    N_i[N_i == 0] = 1
    if N_i.sum() != N:
      N_i[0] -= (N_i.sum() - N)

    # 6. Allocate sample size n into strata sample sizes n_i
    n_i_float = thetas * n
    n_i = n_i_float.astype(int)
    remainder_n = n - n_i.sum()
    if remainder_n > 0:
        top_indices_n = np.argsort(n_i_float - n_i)[-remainder_n:]
        n_i[top_indices_n] += 1
    n_i = np.maximum(1, np.minimum(n_i, N_i))
    if n_i.sum() != n:
      diff = n_i.sum() - n
      while diff > 0:
          largest_idx = np.argmax(n_i)
          if n_i[largest_idx] > 1:
              n_i[largest_idx] -= 1
              diff -= 1
          else:
              break

    # 7. Simulate population microdata, draw sample, and calculate weights
    all_samples = []
    for i in range(k):
        population_y = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=N_i[i])
        sample_y = np.random.choice(population_y, size=n_i[i], replace=False)
        weight = N_i[i] / n_i[i]
        stratum_sample = pd.DataFrame({
            'stratum_id': i + 1,
            'y_ij': sample_y,
            'weight': weight
        })
        all_samples.append(stratum_sample)

    # 8. Combine and return
    final_sample_df = pd.concat(all_samples, ignore_index=True)
    official_totals = {'T_official': T, 'S_star_official': S_i_star}
    return final_sample_df, official_totals


sample_df, totals = simulate_contradictory_data(
   T = 1000000,
   k = 4,
   c = .1,
   N = 1000,
   n = 100,
   dirichlet_alpha = .4,
   gamma_shape = 2,
   gamma_scale = 2
)
