import numpy as np
import pandas as pd


def simulate_contradictory_data(
    T: float,  # official grand total
    k: int,   # number of strata
    c: float,  # contradiction factor. Strata totals fall short by floor(c * T)
    n: int,  # Sample size, over all strata
    dirichlet_alpha = 5.0,  # low value creates uneven divisions
    gamma_shape: float = 2.0,  
    gamma_scale: float = 2.0
):

    """Returns tuple containing the sample DataFrame and the official totals dict."""
    # Simulate stratum proportions (thetas)
    alphas = np.full(k, dirichlet_alpha)
    thetas = np.random.dirichlet(alphas)

    # Define Stratum subtotals S_i
    S_i = thetas * T

    # Calculate and distribute the contradiction 'delta'
    delta = np.floor(c * T)
    p = np.full(k, 1/k)
    delta_i = np.random.multinomial(n=int(delta), pvals=p)

    # Calculate final, contradictory official subtotals S_i^*
    S_i_star = S_i - delta_i

    # Allocate sample size n into strata sample sizes n_i
    n_i_float = thetas * n  # Assuming sample size in proportion to stratum total
    n_i = n_i_float.astype(int)
    remainder_n = n - n_i.sum()
    if remainder_n > 0:
        top_indices_n = np.argsort(n_i_float - n_i)[-remainder_n:]
        n_i[top_indices_n] += 1
    if n_i.sum() != n:
      diff = n_i.sum() - n
      while diff > 0:
          largest_idx = np.argmax(n_i)
          if n_i[largest_idx] > 1:
              n_i[largest_idx] -= 1
              diff -= 1
          else:
              break

    # Simulate population microdata, draw sample, and calculate weights
    all_samples = []
    for i in range(k):
        sample_y = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=n_i[i])
        weight = np.full(n_i[i], S_i_star[i] / np.sum(sample_y))  # baseline
        stratum_sample = pd.DataFrame({
            'stratum_id': i + 1,
            'y_ij': sample_y,
            'w_ij': weight
        })
        all_samples.append(stratum_sample)

    # Combine and return
    final_sample_df = pd.concat(all_samples, ignore_index=True)
    official_totals = {'T_official': T, 'S_star_official': S_i_star}
    return final_sample_df, official_totals


sample_df, totals = simulate_contradictory_data(
   T = 6000,
   k = 3,
   c = .1,
   n = 30
)


import torch
num_strata = sample_df['stratum_id'].nunique()
num_obs = len(sample_df)
M = torch.zeros((num_strata + 1, num_obs), dtype=torch.float32)

M[0, :] = torch.tensor(sample_df['y_ij'].values, dtype=torch.float32)
for i in range(num_strata):
    stratum_mask = (sample_df['stratum_id'] == (i + 1))
    M[i + 1, :] = torch.tensor(np.where(stratum_mask, sample_df['y_ij'], 0), dtype=torch.float32)

t_values = [totals['T_official']] + list(totals['S_star_official'])
t = torch.tensor(t_values, dtype=torch.float32)

log_w = torch.tensor(np.log(sample_df['w_ij'].values), dtype=torch.float32, requires_grad=True)

def relative_loss(log_weights):
    """
    Calculates the squared relative error, split into two components:
    1. The loss for the grand total.
    2. The sum of losses for the stratum totals.
    """
    weights = torch.exp(log_weights)
    estimated_totals = torch.matmul(M, weights)
    grand_total_loss = ((estimated_totals[0] - t[0]) / t[0]) ** 2
    stratum_totals_loss = torch.sum(((estimated_totals[1:] - t[1:]) / t[1:]) ** 2)
    return grand_total_loss + stratum_totals_loss

# Setup the Adam optimizer to adjust the log-weights
optimizer = torch.optim.Adam([log_w], lr=0.1)
n_epochs = 100

print("--- Starting Optimization ---")
# Optimization loop
for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = relative_loss(log_w)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.8f}")
print("--- Optimization Finished ---\n")


# Results and Verification ---
final_weights = torch.exp(log_w).detach()
final_estimates = torch.matmul(M, final_weights)

print("--- Comparison of Totals ---")
print(f"{'Metric':<25} | {'Official Target':>15} | {'Final Estimate':>15} | {'Relative Error (%)':>20}")
print("-" * 85)

grand_total_official = t[0].item()
grand_total_estimate = final_estimates[0].item()
rel_err_grand_total = (grand_total_estimate - grand_total_official) / grand_total_official * 100
print(f"{'Grand Total (T)':<25} | {grand_total_official:>15.2f} | {grand_total_estimate:>15.2f} | {rel_err_grand_total:>19.4f}%")

for i in range(num_strata):
    stratum_official = t[i + 1].item()
    stratum_estimate = final_estimates[i + 1].item()
    rel_err_stratum = (stratum_estimate - stratum_official) / stratum_official * 100
    print(f"{f'Stratum {i+1} Total':<25} | {stratum_official:>15.2f} | {stratum_estimate:>15.2f} | {rel_err_stratum:>19.4f}%")

print("\nFinal optimized weights (first 5):")
print(final_weights.numpy()[:5])

