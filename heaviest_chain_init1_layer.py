#!/usr/bin/env python3

import math
import numpy as np
from collections import defaultdict
from read_from_excel import ExcelReader
import argparse

# ---------- Fenwick Tree for prefix MAX (value, index) ----------
class FenwickMax:
    def __init__(self, n):
        self.n = n
        # store (best_value, best_index)
        self.bit_val = np.zeros(n + 1, dtype=float)
        self.bit_idx = np.full(n + 1, -1, dtype=int)

    # runtime: O(log M), Where M is the num of uniq ys valves
    def update(self, i, value, idx):
        n = self.n
        while i <= n:
            if value > self.bit_val[i]:
                self.bit_val[i] = value
                self.bit_idx[i] = idx
            i += i & -i

    # runtime: O(log M), Where M is the num of uniq ys valves
    def query(self, i):
        best_v = 0.0
        best_prev = -1
        while i > 0:
            if self.bit_val[i] > best_v:
                best_v = self.bit_val[i]
                best_prev = self.bit_idx[i]
            i -= i & -i
        return best_v, best_prev


# ---------- oracle implemented (to find the heaviest chain) ----------
def oracle(w, xs, y_rank, ids, M, n):
        fenw = FenwickMax(M)
        dp = np.zeros(n, dtype=float)
        parent = np.full(n, -1, dtype=int)

        i = 0
        best_end = -1
        best_sum = 0.0

        while i < n:
            x_val = xs[ids[i]]

            group = []
            while i < n and xs[ids[i]] == x_val:
                group.append(ids[i])
                i += 1

            tmp = []
            for pid in group:
                r = y_rank[pid]
                best_v, best_prev = fenw.query(r - 1)
                dp[pid] = w[pid] + best_v
                parent[pid] = best_prev
                tmp.append(pid)

                if dp[pid] > best_sum:
                    best_sum = dp[pid]
                    best_end = pid

            for pid in tmp:
                fenw.update(y_rank[pid], dp[pid], pid)

        chain = []
        cur = best_end
        while cur != -1:
            chain.append(cur)
            cur = parent[cur]
        chain.reverse()
        return chain, best_sum


# ---------- Pretty progress bar (no external deps) ----------
def print_progress(t, T, best_score, S, score, delta, bar_width=32):
    """
    Lightweight progress bar + metrics.
    Prints in-place (no new line) except at the end.
    I/O only: does not affect optimization.
    """
    frac = t / T
    filled = int(bar_width * frac)
    bar = "█" * filled + "░" * (bar_width - filled)
    msg = (f"\r[{bar}] {100*frac:6.2f}% | "
           f"iter {t}/{T} | "
           f"S={S:9.6f} | "
           f"score={score:12.8f} | "
           f"best={best_score:12.8f} | "
           f"delta={delta: .3e}")
    end = "\n" if t == T else ""
    print(msg, end=end, flush=True)


# ---------- Init #1 helpers: depth-layers feasible ----------
def compute_depth_lengths(xs, y_rank, ids, M, n):
    """
    Compute depth[i] = length of the longest chain ending at i (strict dominance).
    Strictness is enforced by:
      - delayed updates within equal-x groups
      - query(r-1) for strict y
    Runtime: O(n log M)
    """
    fenw = FenwickMax(M)
    dp_len = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        x_val = xs[ids[i]]

        group = []
        while i < n and xs[ids[i]] == x_val:
            group.append(ids[i])
            i += 1

        tmp = []
        for pid in group:
            r = y_rank[pid]
            best_v, _ = fenw.query(r - 1)
            dp_len[pid] = best_v + 1.0
            tmp.append(pid)

        for pid in tmp:
            fenw.update(y_rank[pid], dp_len[pid], pid)

    depth = dp_len.astype(int)
    L = int(dp_len.max()) if n > 0 else 0
    return depth, L


def init_weights_layer(depth, L, n, beta=1.0):
    """
    Feasible-by-construction initialization:
      - A_k = {i : depth(i) = k}
      - alpha_k = (|A_k| + beta) / (n + beta*L)
      - w_i = alpha_{depth(i)}

    Any chain has at most one point per depth k, so chain weight <= sum_k alpha_k = 1.
    """
    counts = np.bincount(depth, minlength=L + 1).astype(float)
    denom = n + beta * L

    alpha = np.zeros(L + 1, dtype=float)
    for k in range(1, L + 1):
        alpha[k] = (counts[k] + beta) / denom

    w_init = np.array([alpha[d] for d in depth], dtype=float)
    return w_init


def solve(points, max_iter=10000, tol=1e-6):
    n = len(points)
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    ys_sorted_unique = sorted(set(ys))
    y_to_rank = {y: i + 1 for i, y in enumerate(ys_sorted_unique)}
    y_rank = np.array([y_to_rank[y] for y in ys], dtype=int)
    M = len(ys_sorted_unique)

    ids = list(range(n))
    ids.sort(key=lambda i: (xs[i], ys[i]))

    eps = 1e-4
    eta0 = 1e-2

    # --- Hyperparameters for the two fixes (tuned for stability on n up to ~10k) ---
    noise_scale = 1e-12   # tiny noise for oracle tie-breaking (very small on purpose)
    bias_factor = 0.05    # bias = bias_factor * eta (eta = eta0/sqrt(t))
    rng = np.random.default_rng(12345)  # deterministic noise for reproducibility

    # ---------- Init1: depth-layers feasible initialization ----------
    depth, L = compute_depth_lengths(xs, y_rank, ids, M, n)
    w_init = init_weights_layer(depth, L, n, beta=1.0)

    # Keep algorithm structure identical: w = 1/s
    s = 1.0 / np.maximum(w_init, eps)

    best_score = float("inf")
    best_w = None

    # Print progress ~200 times total
    print_every = max(1, max_iter // 200)

    for t in range(1, max_iter + 1):
        w = 1.0 / s

        # ---- Fix #2: oracle diversification via tiny multiplicative noise ----
        # This is only to break ties / avoid repeating the exact same chain.
        w_noisy = w * (1.0 + noise_scale * rng.standard_normal(n))
        # keep weights non-negative (noise is tiny; this is just a safety clamp)
        w_noisy = np.maximum(w_noisy, 0.0)

        chain, S = oracle(w_noisy, xs, y_rank, ids, M, n)

        scale = max(S, 1.0)
        w_feas = w / scale  # objective computed on true current w (not noisy)

        score = -(np.log2(np.maximum(w_feas, eps)).sum() / n)

        # ---- Fix #1: tie-breaking push (bias) so we keep moving even when S==1 ----
        eta = eta0 / math.sqrt(t)
        bias = bias_factor * eta
        delta = eta0 * (S - 1.0) + bias

        # --- Progress output (I/O only) ---
        if (t == 1) or (t % print_every == 0) or (t == max_iter):
            print_progress(t, max_iter, best_score, S, score, delta)

        if score < best_score and S <= 1 + tol:
            best_score = score
            best_w = w_feas.copy()

        for pid in chain:
            s[pid] += delta

        # Safety: keep s positive to avoid division issues (rare but important long-run)
        if np.any(s <= 0):
            s = np.maximum(s, eps)

    return best_w, best_score


def main():
    parser = argparse.ArgumentParser(description="Heaviest chain optimizer (Init1 + bias + noise)")

    parser.add_argument("-p", "--points", type=str, default="points_1000.xlsx",
                        help="Excel file containing point list (x,y)")

    parser.add_argument("-i", "--iters", type=int, default=100000,
                        help="Maximum number of iterations")

    parser.add_argument("-t", "--tol", type=float, default=1e-6,
                        help="Tolerance value (default: 1e-6)")

    args = parser.parse_args()

    pts = ExcelReader.read_points(args.points)

    w, score = solve(pts, args.iters, tol=args.tol)

    print("score:", score)
    return score


if __name__ == "__main__":
    main()
