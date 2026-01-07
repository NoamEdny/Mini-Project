#!/usr/bin/env python3

import math
import numpy as np
from read_from_excel import ExcelReader
import argparse


# ---------- Fenwick Tree for prefix MAX (value, index) ----------
class FenwickMax:
    def __init__(self, size):
        self.size = size
        self.best_value = np.zeros(size + 1, dtype=float)
        self.best_index = np.full(size + 1, -1, dtype=int)

    def update(self, i, value, idx):
        while i <= self.size:
            if value > self.best_value[i]:
                self.best_value[i] = value
                self.best_index[i] = idx
            i += i & -i

    def query(self, i):
        max_value = 0.0
        max_index = -1
        while i > 0:
            if self.best_value[i] > max_value:
                max_value = self.best_value[i]
                max_index = self.best_index[i]
            i -= i & -i
        return max_value, max_index


# ---------- Oracle: find heaviest increasing chain ----------
def oracle(point_weight, xs, y_rank, sorted_ids, num_y_levels, n):
    fenwick = FenwickMax(num_y_levels)

    best_chain_weight = np.zeros(n, dtype=float)
    prev_point = np.full(n, -1, dtype=int)

    i = 0
    best_chain_end = -1
    max_chain_weight = 0.0

    while i < n:
        x_val = xs[sorted_ids[i]]

        same_x_group = []
        while i < n and xs[sorted_ids[i]] == x_val:
            same_x_group.append(sorted_ids[i])
            i += 1

        pending_updates = []
        for pt in same_x_group:
            y_idx = y_rank[pt]
            prev_weight, prev_pt = fenwick.query(y_idx - 1)

            best_chain_weight[pt] = point_weight[pt] + prev_weight
            prev_point[pt] = prev_pt
            pending_updates.append(pt)

            if best_chain_weight[pt] > max_chain_weight:
                max_chain_weight = best_chain_weight[pt]
                best_chain_end = pt

        for pt in pending_updates:
            fenwick.update(y_rank[pt], best_chain_weight[pt], pt)

    chain = []
    cur = best_chain_end
    while cur != -1:
        chain.append(cur)
        cur = prev_point[cur]

    chain.reverse()
    return chain, max_chain_weight


def solve(points, max_iter=10000, tol=1e-6):
    # ================= Hyper-parameters =================
    base_lr = 5      # learning rate
    influence_scale = 1.1 # multiplier for initial influenceFactor
    lr_func = math.sqrt # learning rate decay function: lr = base_lr / lr_func(t)
    # ====================================================

    n = len(points)

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    # Rank y-coordinates
    unique_ys = sorted(set(ys))
    y_to_rank = {y: i + 1 for i, y in enumerate(unique_ys)}
    y_rank = np.array([y_to_rank[y] for y in ys], dtype=int)
    num_y_levels = len(unique_ys)

    sorted_ids = list(range(n))
    sorted_ids.sort(key=lambda i: (xs[i], ys[i]))

    # ----- Initialization -----
    init_weight = np.ones(n, dtype=float)
    chain, chain_weight = oracle(
        init_weight, xs, y_rank, sorted_ids, num_y_levels, n
    )

    influenceFactor = np.ones(n, dtype=float) * chain_weight * influence_scale

    best_score = float("inf")
    best_weights = None

    grad_accum = np.zeros(n)
    eps = 1e-8

    influenceFactor = np.ones(n, dtype=float)

    # ----- Optimization loop -----
    for t in range(1, max_iter + 1):
        point_weight = 1.0 / influenceFactor

        chain, chain_weight = oracle(
            point_weight, xs, y_rank, sorted_ids, num_y_levels, n
        )

        scale = max(chain_weight, 1.0)
        feasible_weight = point_weight / scale

        score = -(np.log2(feasible_weight)).sum() / n

        if score < best_score:
            best_score = score
            best_weights = feasible_weight.copy()

        if(t%50 == 0) or t < 50:
            msg = (
           f"iter {t}/{max_iter} | "
           f"S={chain_weight:9.6f} | "
           f"score={score:12.8f} | "
           f"best={best_score:12.8f}")
            print(f"{msg}")
            #print(f"feasible_weight= {feasible_weight}")

        lr = base_lr / lr_func(t)
        delta = base_lr * (chain_weight - 1.0)

        for pt in chain:
            grad_accum[pt] += delta**2
            adjusted_delta = delta / math.sqrt(grad_accum[pt] + eps)
            influenceFactor[pt] += adjusted_delta


    return best_weights, best_score


def main():
    parser = argparse.ArgumentParser(description="Heaviest chain optimizer")

    parser.add_argument("-p", "--points", type=str, default="points_1000.xlsx",
                        help="Excel file containing point list (x,y)")

    parser.add_argument("-i", "--iters", type=int, default=100000,
                        help="Maximum number of iterations")

    parser.add_argument("-t", "--tol", type=float, default=1e-6,
                        help="Tolerance value")

    args = parser.parse_args()

    points = ExcelReader.read_points(args.points)
    #points = [[0,0],[0.1,1],[0.5,0.5],[1,0.9]] #sb

    weights, score = solve(points, args.iters, tol=args.tol)

    print("score:", score)
    return score


if __name__ == "__main__":
    main()
