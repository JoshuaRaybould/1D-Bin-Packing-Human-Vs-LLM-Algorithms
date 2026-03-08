import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.time()
    deadline = start + max(0.0, float(time_limit))

    # ---- Helpers ----
    def now() -> float:
        return time.time()

    def feasible_bin_score(item_idx: int, item_w: int, bin_items: List[int], bin_rem: int,
                           tau: List[List[float]], alpha: float, beta: float) -> float:
        """Score of placing item into an existing bin."""
        # Heuristic: prefer tighter fill (smaller remainder after placement)
        rem_after = bin_rem - item_w
        # Use (1 / (1+rem_after))^beta, integer-safe.
        eta = 1.0 / (1.0 + rem_after)

        # Pheromone: average attraction to items already in bin
        if bin_items:
            s = 0.0
            for j in bin_items:
                s += tau[item_idx][j]
            tau_ij = s / len(bin_items)
        else:
            tau_ij = 1.0

        return (tau_ij ** alpha) * (eta ** beta)

    def new_bin_score(item_w: int, tau_new: float, beta: float) -> float:
        # Small heuristic preference for opening a new bin: neutral, slightly penalize waste.
        rem_after = bin_capacity - item_w
        eta = 1.0 / (1.0 + rem_after)
        return tau_new * (eta ** (0.25 * beta))

    def construct_solution(order: List[int], tau: List[List[float]],
                           alpha: float, beta: float,
                           q0: float, tau_new: float,
                           check_every: int = 32) -> Tuple[List[List[int]], List[int]]:
        """Construct a packing using probabilistic decisions guided by pheromone and heuristic."""
        bins: List[List[int]] = []
        bin_weights: List[int] = []
        bin_rems: List[int] = []

        for t, idx in enumerate(order):
            if (t & (check_every - 1)) == 0 and now() >= deadline:
                break

            w = weights[idx]

            # Collect feasible existing bins
            feasible = []  # list of (bin_id, score)
            for b in range(len(bins)):
                if bin_rems[b] >= w:
                    sc = feasible_bin_score(idx, w, bins[b], bin_rems[b], tau, alpha, beta)
                    feasible.append((b, sc))

            # Option: open new bin
            sc_new = new_bin_score(w, tau_new, beta)

            # If no feasible bin, must open new
            if not feasible:
                bins.append([idx])
                bin_weights.append(w)
                bin_rems.append(bin_capacity - w)
                continue

            # Decision: exploitation vs exploration
            if random.random() < q0:
                # Greedy choice among feasible and new bin
                best_b = max(feasible, key=lambda x: x[1])
                if sc_new > best_b[1]:
                    # open new bin
                    bins.append([idx])
                    bin_weights.append(w)
                    bin_rems.append(bin_capacity - w)
                else:
                    b = best_b[0]
                    bins[b].append(idx)
                    bin_weights[b] += w
                    bin_rems[b] -= w
            else:
                # Roulette wheel over feasible bins + new bin
                total = sc_new
                for _, sc in feasible:
                    total += sc

                r = random.random() * total
                acc = sc_new
                if r <= acc:
                    bins.append([idx])
                    bin_weights.append(w)
                    bin_rems.append(bin_capacity - w)
                else:
                    chosen_b = feasible[-1][0]
                    for b, sc in feasible:
                        acc += sc
                        if r <= acc:
                            chosen_b = b
                            break
                    bins[chosen_b].append(idx)
                    bin_weights[chosen_b] += w
                    bin_rems[chosen_b] -= w

        return bins, bin_weights

    def solution_cost(bins: List[List[int]], bin_w: List[int]) -> Tuple[int, int, int]:
        """Lexicographic cost: (#bins, total waste, max waste) minimize."""
        m = len(bins)
        waste = 0
        maxwaste = 0
        for bw in bin_w:
            w = bin_capacity - bw
            waste += w
            if w > maxwaste:
                maxwaste = w
        return (m, waste, maxwaste)

    def reinforce_pheromone(tau: List[List[float]], bins: List[List[int]],
                            delta: float, tau_min: float, tau_max: float) -> None:
        # Reinforce pairwise co-membership in bins
        for b in bins:
            k = len(b)
            if k <= 1:
                continue
            for i in range(k):
                a = b[i]
                row = tau[a]
                for j in range(i + 1, k):
                    c = b[j]
                    nv = row[c] + delta
                    if nv > tau_max:
                        nv = tau_max
                    row[c] = nv
                    # symmetric
                    nv2 = tau[c][a] + delta
                    if nv2 > tau_max:
                        nv2 = tau_max
                    tau[c][a] = nv2

        # Clamp all pheromone lightly (only where updated we already clamp, but keep safe)
        # Avoid O(n^2) clamp each time.

    # ---- Preprocessing ----
    # Sort items descending by weight; ACO will decide bin placements.
    order = list(range(n))
    order.sort(key=lambda i: weights[i], reverse=True)

    # Basic feasibility check: if any item exceeds capacity, put it alone (still infeasible in classic sense,
    # but we will place it in its own bin). However standard instances should not have this.
    # We'll handle by forcing new bin even if overweight (bin weight may exceed capacity).
    # (Not expected; no special objective handling.)

    # Initial solution via First-Fit Decreasing (for baseline upper bound)
    def ffd_solution() -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        bin_w: List[int] = []
        rem: List[int] = []
        for idx in order:
            w = weights[idx]
            placed = False
            for b in range(len(bins)):
                if rem[b] >= w:
                    bins[b].append(idx)
                    bin_w[b] += w
                    rem[b] -= w
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_w.append(w)
                rem.append(bin_capacity - w)
        return bins, bin_w

    best_bins, best_binw = ffd_solution()
    best_cost = solution_cost(best_bins, best_binw)

    # ---- ACO parameters ----
    # Iterations fixed (but may exit early due to time). Choose moderate default.
    # Also scale ants with problem size.
    max_iters = 400 if n <= 400 else 250
    n_ants = 18 if n <= 200 else 12

    alpha = 1.2
    beta = 3.5
    rho = 0.12  # evaporation
    q0 = 0.65   # exploitation probability

    # Pheromone matrix on item-item pairs.
    # For memory: n up to a few thousand may be heavy (n^2). We will use a sparse structure if large.
    # But requirement says standard library only; implement sparse as dict-of-dict when needed.
    # We'll choose dense up to 1200; above, sparse.
    dense_limit = 1200

    use_dense = n <= dense_limit

    if use_dense:
        # Initialize with small uniform pheromone
        tau0 = 1.0
        tau = [[tau0] * n for _ in range(n)]
    else:
        # Sparse: only store reinforced edges; default tau0
        tau0 = 1.0
        tau = None  # type: ignore

        class SparseTau:
            __slots__ = ("d",)

            def __init__(self):
                self.d = {}  # i -> {j: val}

            def get(self, i: int, j: int) -> float:
                row = self.d.get(i)
                if row is None:
                    return tau0
                return row.get(j, tau0)

            def add(self, i: int, j: int, delta: float, tau_max: float) -> None:
                row = self.d.get(i)
                if row is None:
                    row = {}
                    self.d[i] = row
                nv = row.get(j, tau0) + delta
                if nv > tau_max:
                    nv = tau_max
                row[j] = nv

            def evaporate(self, rho: float, tau_min: float) -> None:
                # Evaporate only stored edges; keep implicit default at tau0 (not evaporated).
                # To emulate evaporation for all edges, we keep tau0 constant and only adjust stored ones.
                # This is an approximation suitable for large n.
                d = self.d
                for i, row in list(d.items()):
                    for j, val in list(row.items()):
                        nv = (1.0 - rho) * val
                        if nv < tau_min:
                            # Drop to implicit baseline if below tau_min close to tau0
                            if tau0 <= tau_min:
                                nv = tau_min
                                row[j] = nv
                            else:
                                row.pop(j, None)
                        else:
                            row[j] = nv
                    if not row:
                        d.pop(i, None)

        stau = SparseTau()

    # Bounds and new-bin pheromone
    tau_min = 0.2
    tau_max = 6.0
    tau_new = 1.0

    def tau_get(i: int, j: int) -> float:
        if use_dense:
            return tau[i][j]  # type: ignore
        return stau.get(i, j)  # type: ignore

    # Wrap scoring functions for sparse/dense
    def feasible_bin_score_generic(item_idx: int, item_w: int, bin_items: List[int], bin_rem: int) -> float:
        rem_after = bin_rem - item_w
        eta = 1.0 / (1.0 + rem_after)
        if bin_items:
            s = 0.0
            for j in bin_items:
                s += tau_get(item_idx, j)
            tau_ij = s / len(bin_items)
        else:
            tau_ij = 1.0
        return (tau_ij ** alpha) * (eta ** beta)

    def construct_solution_generic(order_: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        bin_weights: List[int] = []
        bin_rems: List[int] = []
        for t, idx in enumerate(order_):
            if (t & 31) == 0 and now() >= deadline:
                break
            w = weights[idx]
            feasible: List[Tuple[int, float]] = []
            for b in range(len(bins)):
                if bin_rems[b] >= w:
                    feasible.append((b, feasible_bin_score_generic(idx, w, bins[b], bin_rems[b])))
            sc_new = new_bin_score(w, tau_new, beta)
            if not feasible:
                bins.append([idx])
                bin_weights.append(w)
                bin_rems.append(bin_capacity - w)
                continue

            if random.random() < q0:
                best_b, best_sc = max(feasible, key=lambda x: x[1])
                if sc_new > best_sc:
                    bins.append([idx])
                    bin_weights.append(w)
                    bin_rems.append(bin_capacity - w)
                else:
                    bins[best_b].append(idx)
                    bin_weights[best_b] += w
                    bin_rems[best_b] -= w
            else:
                total = sc_new + sum(sc for _, sc in feasible)
                r = random.random() * total
                acc = sc_new
                if r <= acc:
                    bins.append([idx])
                    bin_weights.append(w)
                    bin_rems.append(bin_capacity - w)
                else:
                    chosen = feasible[-1][0]
                    for b, sc in feasible:
                        acc += sc
                        if r <= acc:
                            chosen = b
                            break
                    bins[chosen].append(idx)
                    bin_weights[chosen] += w
                    bin_rems[chosen] -= w
        return bins, bin_weights

    def evaporate() -> None:
        nonlocal tau_new
        tau_new = max(tau_min, (1.0 - rho) * tau_new)
        if use_dense:
            # Evaporate all entries; O(n^2) but acceptable for n<=1200
            T = tau  # type: ignore
            for i in range(n):
                row = T[i]
                for j in range(n):
                    v = (1.0 - rho) * row[j]
                    if v < tau_min:
                        v = tau_min
                    row[j] = v
        else:
            stau.evaporate(rho, tau_min)  # type: ignore

    def reinforce(bins: List[List[int]], bin_w: List[int]) -> None:
        # Stronger reinforcement for fewer bins and less waste
        m, waste, _ = solution_cost(bins, bin_w)
        # delta scales inversely with bins and waste
        delta = 1.0 / (1.0 + (m - 1) + 0.01 * waste)
        # amplify
        delta *= 4.0
        if use_dense:
            reinforce_pheromone(tau, bins, delta, tau_min, tau_max)  # type: ignore
        else:
            # sparse reinforcement
            for b in bins:
                k = len(b)
                if k <= 1:
                    continue
                for i in range(k):
                    a = b[i]
                    for j in range(i + 1, k):
                        c = b[j]
                        stau.add(a, c, delta, tau_max)  # type: ignore
                        stau.add(c, a, delta, tau_max)  # type: ignore

        # also reinforce opening-new-bin attractiveness slightly if solution is good
        nonlocal tau_new
        tau_new = min(tau_max, tau_new + 0.15 * delta)

    # ---- Main loop ----
    it = 0
    while it < max_iters:
        if now() >= deadline:
            break

        iter_best_bins = None
        iter_best_binw = None
        iter_best_cost = None

        # Slightly randomized order occasionally: swap within equal-weight blocks
        # to diversify while maintaining mostly descending order.
        if n <= 2000:
            order2 = order[:]  # copy
            # random perturbations
            swaps = 2 if n < 200 else 4
            for _ in range(swaps):
                a = random.randrange(n)
                b = random.randrange(n)
                if a > b:
                    a, b = b, a
                # only swap if weights are close to preserve structure
                if abs(weights[order2[a]] - weights[order2[b]]) <= 1:
                    order2[a], order2[b] = order2[b], order2[a]
        else:
            order2 = order

        for _k in range(n_ants):
            if now() >= deadline:
                break

            bins_k, binw_k = construct_solution_generic(order2)
            c = solution_cost(bins_k, binw_k)
            if (iter_best_cost is None) or (c < iter_best_cost):
                iter_best_cost = c
                iter_best_bins = bins_k
                iter_best_binw = binw_k

            if c < best_cost:
                best_cost = c
                best_bins = bins_k
                best_binw = binw_k

        # Pheromone update using iteration-best and occasionally global-best
        evaporate()
        if iter_best_bins is not None:
            reinforce(iter_best_bins, iter_best_binw)  # type: ignore

        # With small probability, also reinforce global best (stabilizes)
        if random.random() < 0.25:
            reinforce(best_bins, best_binw)

        it += 1

    # ---- Produce output in required format ----
    # Ensure bin_weights correct and within capacity if feasible.
    packing = [b[:] for b in best_bins]
    bin_weights_out = [0] * len(packing)
    for i, b in enumerate(packing):
        s = 0
        for idx in b:
            s += weights[idx]
        bin_weights_out[i] = s

    return {"packing": packing, "bin_weights": bin_weights_out}
