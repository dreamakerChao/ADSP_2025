import numpy as np

# --- constants -------------------------------------------------
N = 1500           # input sequence length
M = 10            # FIR filter length
L_max = 10000      # search upper bound for L

# --- search range ---------------------------------------------
L = np.arange(1, L_max + 1)      # 1 … L_max (inclusive)

# --- cost function --------------------------------------------
cost = (N / L) * 3 * (L + M - 1) * (np.log2(L + M - 1) + 1)

# --- find minimum ---------------------------------------------
idx_min = np.argmin(cost)        # index of minimal cost
L0 = L[idx_min]                  # corresponding L
min_cost = cost[idx_min]

print(f"min_cost = {min_cost:.6f}")
print(f"L0       = {L0}")
