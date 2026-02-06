"""
MIQP for binary latent factors:  X = B Z + eps

- Generates synthetic data (B_true, Z_true, X)
- Builds and solves the MIQP in Gurobi using the big-M linearization:
    U[i,l,j] = B[i,l] * Z[l,j]
    S[i,j]   = sum_l U[i,l,j]
    minimize sum_{i,j} (X[i,j] - S[i,j])^2

Requirements:
  pip install numpy gurobipy
  (and a working Gurobi license)

Note:
  This joint (B,Z) MIQP is hard; start with small p,k,n.
"""

import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
import itertools

import os

current_dir = os.path.dirname(os.path.abspath(__file__))

np.set_printoptions(linewidth=200)

def generate_synthetic(p=20, k=3, n=40, dense=True, b_prob=0.2, noise_std=0.1, seed=0,
                       L_val=-1.0, U_val=1.0, z_prob=0.3):
    """
    Generate:
      B_true in [L, U]^{p x k}
      Z_true in {0,1}^{k x n}
      X = B_true @ Z_true + eps
    Using simple constant bounds (L_val, U_val) for all (i,l).
    """
    rng = np.random.default_rng(seed)

    L = np.full((p, k), L_val, dtype=float)
    U = np.full((p, k), U_val, dtype=float)

    B_true = rng.uniform(low=L, high=U)  # elementwise bounds
    if not dense:
        mask = (rng.uniform(size=(p, k)) < b_prob).astype(float)
        B_true = B_true * mask
    Z_true = (rng.uniform(size=(k, n)) < z_prob).astype(int)

    eps = rng.normal(loc=0.0, scale=noise_std, size=(p, n))
    X = B_true @ Z_true + eps
    return X, B_true, Z_true, L, U


def solve_miqp_binary_relaxed(X, L, U, n1,
                             time_limit=300, mip_gap=1e-4, threads=0, verbose=True):
    """
    Solve MIQP with:
      - Z_{l,j} binary for j=0,...,n1-1  (identifiability block)
      - Z_{l,j} continuous in [0,1] for j=n1,...,n-1 (accuracy block)

    Returns:
      B_hat: (p,k) float
      Z_hat: (k,n) float   (first n1 columns ~{0,1}, rest in [0,1])
      S_hat: (p,n) float
      info: dict
    """
    p, n = X.shape
    p2, k = L.shape
    assert p2 == p and U.shape == (p, k)
    assert 0 <= n1 <= n

    m = gp.Model("binary_relaxed_latent_miqp")
    if not verbose:
        m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    if threads is not None and threads > 0:
        m.Params.Threads = threads

    # B in [L,U]
    B = m.addMVar((p, k), lb=-1, ub=1, vtype=GRB.CONTINUOUS, name="B")

    # Z split
    Zb = m.addVars(k, n1, vtype=GRB.BINARY, name="Zb")
    Zc = m.addVars(k, n - n1, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="Zc")

    def Z(l, j):
        return Zb[l, j] if j < n1 else Zc[l, j - n1]

    # w = B * Z (McCormick)
    w = m.addVars(p, k, n, lb=-1, vtype=GRB.CONTINUOUS, name="U")

    # S = sum_l w
    S = m.addVars(p, n, lb=float(np.min(L)) * n, vtype=GRB.CONTINUOUS, name="S")

    m.addConstrs(
        (S[i, j] == gp.quicksum(w[i, l, j] for l in range(k))
         for i in range(p) for j in range(n)),
        name="S_def"
    )

    # McCormick / big-M: valid for Z in [0,1] too
    m.addConstrs(
        (w[i, l, j] <= float(U[i, l]) * Z(l, j)
         for i in range(p) for l in range(k) for j in range(n)),
        name="mc1"
    )
    m.addConstrs(
        (w[i, l, j] >= float(L[i, l]) * Z(l, j)
         for i in range(p) for l in range(k) for j in range(n)),
        name="mc2"
    )
    m.addConstrs(
        (w[i, l, j] <= B[i, l] - float(L[i, l]) * (1 - Z(l, j))
         for i in range(p) for l in range(k) for j in range(n)),
        name="mc3"
    )
    m.addConstrs(
        (w[i, l, j] >= B[i, l] - float(U[i, l]) * (1 - Z(l, j))
         for i in range(p) for l in range(k) for j in range(n)),
        name="mc4"
    )

    obj = gp.quicksum((float(X[i, j]) - S[i, j]) * (float(X[i, j]) - S[i, j])
                      for i in range(p) for j in range(n))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi ended with status {m.Status}")

    B_hat = np.array([[B[i, l].X for l in range(k)] for i in range(p)], dtype=float)

    Z_hat = np.zeros((k, n), dtype=float)
    for l in range(k):
        for j in range(n1):
            Z_hat[l, j] = Zb[l, j].X
        for j in range(n1, n):
            Z_hat[l, j] = Zc[l, j - n1].X

    S_hat = np.array([[S[i, j].X for j in range(n)] for i in range(p)], dtype=float)

    info = {
        "status": m.Status,
        "obj_val": m.ObjVal if m.SolCount > 0 else None,
        "runtime_sec": m.Runtime,
        "mip_gap": m.MIPGap if hasattr(m, "MIPGap") else None,
        "node_count": m.NodeCount,
    }
    return B_hat, Z_hat, S_hat, info




def best_permutation_align_B(B_true: np.ndarray,
                             B_hat: np.ndarray,
                             metric: str = "fro"):
    """
    Align columns of B_hat to B_true by finding the best column permutation.

    Args:
        B_true: (p, k)
        B_hat:  (p, k)
        metric: "fro" (min Frobenius norm) or "corr" (max sum of abs corr)

    Returns:
        perm: tuple of length k, where B_hat_aligned[:, t] = B_hat[:, perm[t]]
        B_hat_aligned: (p, k)
        score: objective value (lower is better for "fro", higher is better for "corr")
    """
    p, k = B_true.shape
    assert B_hat.shape == (p, k)

    # Precompute things for speed
    Bt = B_true

    best_perm = None
    best_score = None

    if metric == "fro":
        # minimize ||B_true - B_hat[:, perm]||_F^2
        for perm in itertools.permutations(range(k)):
            Bhp = B_hat[:, perm]
            score = np.sum((Bt - Bhp) ** 2)
            if (best_score is None) or (score < best_score):
                best_score = score
                best_perm = perm

        B_hat_aligned = B_hat[:, best_perm]
        return best_perm, B_hat_aligned, best_score

    elif metric == "corr":
        # maximize sum_t |corr(B_true[:,t], B_hat[:,perm[t]])|
        # (abs corr is common because some problems allow sign flips; if you
        # truly don't want sign invariance, remove abs)
        def col_corr(a, b):
            a0 = a - a.mean()
            b0 = b - b.mean()
            denom = (np.linalg.norm(a0) * np.linalg.norm(b0))
            if denom == 0:
                return 0.0
            return float(a0 @ b0 / denom)

        # Build correlation matrix C[t, l] = |corr(B_true[:,t], B_hat[:,l])|
        C = np.zeros((k, k))
        for t in range(k):
            for l in range(k):
                C[t, l] = abs(col_corr(B_true[:, t], B_hat[:, l]))

        for perm in itertools.permutations(range(k)):
            score = sum(C[t, perm[t]] for t in range(k))
            if (best_score is None) or (score > best_score):
                best_score = score
                best_perm = perm

        B_hat_aligned = B_hat[:, best_perm]
        return best_perm, B_hat_aligned, best_score

    else:
        raise ValueError("metric must be 'fro' or 'corr'")


def apply_perm_to_Z(Z_hat: np.ndarray, perm):
    """
    If you permute columns of B_hat as B_hat[:, t] = B_hat[:, perm[t]],
    you must permute rows of Z_hat the same way: Z_hat[t, :] = Z_hat[perm[t], :].
    """
    return Z_hat[list(perm), :]


def eval_B(B_true, B_hat_aligned):
    """
    A few simple evaluation numbers after alignment.
    """
    fro = np.linalg.norm(B_true - B_hat_aligned, ord="fro")
    rel = fro / (np.linalg.norm(B_true, ord="fro") + 1e-12)

    # column-wise cosine similarity
    def cos(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        return float((a @ b) / denom)

    col_cos = np.array([cos(B_true[:, t], B_hat_aligned[:, t]) for t in range(B_true.shape[1])])
    return {
        "fro_error": fro,
        "rel_fro_error": rel,
        "col_cosine_mean": float(col_cos.mean()),
        "col_cosine_min": float(col_cos.min()),
        "col_cosine": col_cos,
    }


def classification_metrics(Z_hat, Z_true):
    Z_hat = Z_hat.astype(int).ravel()
    Z_true = Z_true.astype(int).ravel()

    TP = np.sum((Z_hat == 1) & (Z_true == 1))
    FP = np.sum((Z_hat == 1) & (Z_true == 0))
    TN = np.sum((Z_hat == 0) & (Z_true == 0))
    FN = np.sum((Z_hat == 0) & (Z_true == 1))

    return TP, FP, TN, FN

def tpr_fpr_f1(Z_hat, Z_true, eps=1e-12):
    TP, FP, TN, FN = classification_metrics(Z_hat, Z_true)

    TPR = TP / (TP + FN + eps)          # Recall / Sensitivity
    FPR = FP / (FP + TN + eps)
    Precision = TP / (TP + FP + eps)
    F1 = 2 * Precision * TPR / (Precision + TPR + eps)

    return {
        "TPR": TPR,
        "FPR": FPR,
        "Precision": Precision,
        "F1": F1,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN
    }

def tpr_fpr_f1_first_block(Z_hat_aligned, Z_true, n1, eps=1e-12):
    Zh = np.rint(Z_hat_aligned[:, :n1]).astype(int)   # should already be near 0/1
    Zt = Z_true[:, :n1].astype(int)
    return tpr_fpr_f1(Zh, Zt, eps=eps)




results = []
for p in [50]:
# for p in [20]:
    for k in [1, 2, 3]:
    # for k in [2]:
        for n in [50]:
        # for n in [8*k]:
            for n1 in [4*k, 6*k, 8*k, 10*k, n]:
                # p, k, n = 50, 1, 10
                X, B_true, Z_true, L, U = generate_synthetic(
                    p=p, k=k, n=n,
                    dense=False,
                    noise_std=0.10,
                    seed=1,
                    L_val=-1.0, U_val=1.0,
                    z_prob=0.3
                )
                B_hat, Z_hat, S_hat, info = solve_miqp_binary_relaxed(
                    X, L, U, n1=n1, time_limit=600, mip_gap=1e-4, threads=8, verbose=True
                )

                perm, B_hat_aligned, score = best_permutation_align_B(B_true, B_hat, metric="fro")
                Z_hat_aligned = apply_perm_to_Z(Z_hat, perm)

                metrics = eval_B(B_true, B_hat_aligned)
                print("best perm:", perm)
                print("alignment score:", score)
                # print(metrics)

                # metrics for Z
                Z_metrics = tpr_fpr_f1_first_block(Z_hat_aligned, Z_true, n1=n1)
                print(Z_metrics)

                # Basic diagnostics (note: B,Z identifiable only up to column permutations in general)
                recon_err = np.linalg.norm(X - S_hat, ord="fro")
                print("\n=== Solve info ===")
                print(info)
                print(f"Frobenius reconstruction ||X - BZ||_F/(n*p) (via S_hat): {recon_err/(n*p):.6f}")

                # If you just want to see raw comparisons (not permutation-aligned):
                print("\nB_true (first 3 rows):\n", B_true[:3])
                print("\nB_hat  (first 3 rows):\n", B_hat_aligned[:3])
                print("\nZ_true (first 3 rows):\n", Z_true)
                print("\nZ_hat  (first 3 rows):\n", Z_hat_aligned[:3])
                results.append([p, k, n, n1, info['mip_gap'], info['runtime_sec'], recon_err/(n*p), Z_metrics['TPR'], Z_metrics['FPR']])
                df = pd.DataFrame(results, columns=['p', 'k', 'n', 'n1', 'rgap', 'time', '||X - BZ||_F/(n*p)', 'TPR', 'FPR'])
                print(df)
                df.to_csv(f"{current_dir}/../experiment_results/split_sample_results.csv")


