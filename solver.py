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

            #TODO maybe sort groups in the beginig?
            group = []
            while i < n and xs[ids[i]] == x_val:
                group.append(ids[i])
                i += 1

            # why we need tmp?
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
def print_progress(t, T, best_score, S, score, bar_width=32):
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
           f"best={best_score:12.8f}")
    end = "\n" if t == T else ""
    print(msg, end=end, flush=True)


def solve(points,max_iter=10000,tol=1e-6):
    n = len(points)
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    ys_sorted_unique = sorted(set(ys))
    y_to_rank = {y: i+1 for i, y in enumerate(ys_sorted_unique)}
    y_rank = np.array([y_to_rank[y] for y in ys], dtype=int)
    M = len(ys_sorted_unique)

    ids = list(range(n))
    ids.sort(key=lambda i: (xs[i], ys[i]))

    eps = 1e-4
    eta0 = 1e-2

   # Initialize weights to 1 (i.e., w = 1/s starts as ones)
    s = np.ones(n, dtype=float)

    # Optional: keep these only if you want an initial S for logging/debug
    w_init = np.ones(n, dtype=float)
    chain, S = oracle(w_init, xs, y_rank, ids, M, n)


    best_score = float("inf")
    best_w = None

    # Print progress ~200 times total (I/O only)
    print_every = max(1, max_iter // 200)

    for t in range(1, max_iter + 1):
        w = 1.0 / s

        chain, S = oracle(w, xs, y_rank, ids, M, n)

        scale = max(S, 1.0)
        w_feas = w / scale

        score = -(np.log2(np.maximum(w_feas, eps)).sum() / n)

        # --- Progress output (I/O only) ---
        if (t == 1) or (t % print_every == 0) or (t == max_iter):
            print_progress(t, max_iter, best_score, S, score)

        if score < best_score:
            best_score = score
            best_w = w_feas.copy()

        # TODO lerenig rate to high?
        eta = eta0 / math.sqrt(t)
        delta = eta0 * (S - 1.0)
        for pid in chain:
            s[pid] += delta

    return best_w, best_score


def main():
    parser = argparse.ArgumentParser(description="Heaviest chain optimizer")

    parser.add_argument("-p", "--points", type=str, default="points_10000_xy.xlsx",
                        help="Excel file containing point list (x,y)")

    parser.add_argument("-i", "--iters", type=int,default=10000,
                        help="Maximum number of iterations")

    parser.add_argument("-t", "--tol", type=float, default=1e-6,
                        help="Tolerance value (default: 1e-6)")

    args = parser.parse_args()

    # --- Load points ---
    pts = ExcelReader.read_points(args.points)


    # --- Run solver ---
    w, score = solve(pts, args.iters, tol=args.tol)

    # --- Output ---
    print("score:", score)
    return score




if __name__ == "__main__":
    main()
