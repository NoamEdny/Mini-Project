import math
from collections import defaultdict

# TODO: maybe use numpy to do it faster??

# ---------- Fenwick Tree for prefix MAX (value, index) ----------
class FenwickMax:
    def __init__(self, n):
        self.n = n
        # store (best_value, best_index)
        self.bit_val = [0.0] * (n + 1)
        self.bit_idx = [-1] * (n + 1)

    def update(self, i, value, idx):
        # set position i to max(current, value)
        n = self.n
        while i <= n:
            if value > self.bit_val[i]:
                self.bit_val[i] = value
                self.bit_idx[i] = idx
            i += i & -i

    def query(self, i):
        # max over [1..i]
        best_s = 0.0
        best_prev = -1
        while i > 0:
            if self.bit_val[i] > best_s:
                best_s = self.bit_val[i]
                best_prev = self.bit_idx[i]
            i -= i & -i
        return best_s, best_prev

# ---------- Heaviest chain oracle: returns (chain_indices, sum) ----------
def heaviest_chain(points_sorted, y_rank, w):
    """
    points_sorted: list of point ids in increasing x (and y)
    y_rank[id] -> 1..M
    w[id] -> weight
    Returns: (chain_list_of_ids, best_sum)
    """
    n = len(points_sorted)
    M = max(y_rank)  # because y_rank is list aligned with ids
    fenw = FenwickMax(M)

    dp = [0.0] * n      # dp over point ids? we use arrays by id in caller usually.
    parent = [-1] * n   # parent[id] = previous id in best chain

    # We'll process by groups with same x. So we need x per id.
    # Assume we have global arrays x, y. We'll pass x via closure or store in points tuple.
    # Better: points_sorted is list of (x, id). We'll implement that below in solve().

    raise NotImplementedError

# ---------- oracle implemented (to find the heaviest chain) ----------
def oracle(w, xs, y_rank, ids):
        fenw = FenwickMax(M)
        dp = [0.0] * n # dp[i] holds the heaviest chain thets end in the i'ts point
        parent = [-1] * n # parent[i] holds a pointer to the prev point thet leds me to the heaviest chain

        i = 0
        best_end = -1
        best_sum = 0.0

        while i < n:
            # group with same x value
            x_val = xs[ids[i]]
            # TODO: maybe we can do it faster?
            group = [] # alll the ids of the same x_val
            while i < n and xs[ids[i]] == x_val:
                group.append(ids[i])
                i += 1

            # compute dp for group WITHOUT updating fenwick yet
            tmp = []
            for pid in group:
                r = y_rank[pid]
                best_s, best_prev = fenw.query(r - 1)  # strict y < current y
                dp[pid] = w[pid] + best_v
                parent[pid] = best_prev
                tmp.append(pid)

                if dp[pid] > best_sum:
                    best_sum = dp[pid]
                    best_end = pid

            # now update fenwick with this group
            for pid in tmp:
                fenw.update(y_rank[pid], dp[pid], pid)

        # reconstruct chain
        chain = []
        cur = best_end
        while cur != -1:
            chain.append(cur)
            cur = parent[cur]
        chain.reverse()
        return chain, best_sum

def solve(points):
    """
    points: list of (x, y) in input order
    Returns best feasible weights in input order, and score
    """
    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # --- compress y ---
    ys_sorted_unique = sorted(set(ys))
    y_to_rank = {y: i+1 for i, y in enumerate(ys_sorted_unique)}  # 1..M
    y_rank = [y_to_rank[y] for y in ys]
    M = len(ys_sorted_unique)

    # --- sort point ids by x (then y) ---
    ids = list(range(n))
    ids.sort(key=lambda i: (xs[i], ys[i]))

    # --- main loop parameters ---
    eps = 1e-12
    tol = 1e-6
    eta0 = 1e-2
    max_iter = 800

    s = [eps] * n

    best_score = float("inf")
    best_w = None

    for t in range(1, max_iter + 1):
        # weights from penalties
        w = [1.0 / si for si in s]

        chain, S = oracle(w)

        # make feasible by scaling
        scale = max(S, 1.0)
        w_feas = [wi / scale for wi in w]

        # score
        # (use eps to avoid log(0) if extreme)
        score = -(sum(math.log2(max(wi, eps)) for wi in w_feas) / n)

        if score < best_score:
            best_score = score
            best_w = w_feas

        # stop if already essentially feasible
        if S <= 1.0 + tol:
            break

        # punish points in the heaviest chain
        eta = eta0 / math.sqrt(t)  # stable steps
        delta = eta * (S - 1.0)
        for pid in chain:
            s[pid] += delta

    return best_w, best_score


if __name__ == "__main__":
    pts = [(3,5), (2,1), (4,6), (5,7)]
    w, score = solve(pts)
    print("weights:", w)
    print("score:", score)
