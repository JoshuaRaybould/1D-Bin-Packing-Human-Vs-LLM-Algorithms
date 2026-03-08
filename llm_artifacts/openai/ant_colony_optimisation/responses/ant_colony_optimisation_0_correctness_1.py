import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.time()
    deadline = start + max(0.0, float(time_limit))

    def now() -> float:
        return time.time()

    # --- Preprocessing ---
    order = list(range(n))
    order.sort(key=lambda i: weights[i], reverse=True)

    def ffd_solution() -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        bin_w: List[int] = []
        rem: List[int] = []
        for idx in order:
            w = weights[idx]
            # If item overweight, force its own bin to keep accounting consistent.
            if w > bin_capacity:
                bins.append([idx])
                bin_w.append(w)
                rem.append(0)
                continue
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

    def solution_cost(bins: List[List[int]], bin_w: List[int]) -> Tuple[int, int, int]:
        m = len(bins)
        waste = 0
        maxwaste = 0
        for bw in bin_w:
            wst = bin_capacity - bw
            if wst < 0:
                wst = 0
            waste += wst
            if wst > maxwaste:
                maxwaste = wst
        return (m, waste, maxwaste)

    best_bins, best_binw = ffd_solution()
    best_cost = solution_cost(best_bins, best_binw)

    # --- ACO parameters ---
    max_iters = 400 if n <= 400 else 250
    n_ants = 18 if n <= 200 else 12

    alpha = 1.2
    beta = 3.5
    rho = 0.12
    q0 = 0.65

    dense_limit = 1200
    use_dense = n <= dense_limit

    tau0 = 1.0
    tau_min = 0.2
    tau_max = 6.0
    tau_new = 1.0

    if use_dense:
        tau = [[tau0] * n for _ in range(n)]
        stau = None

        def tau_get(i: int, j: int) -> float:
            return tau[i][j]

        def tau_add(i: int, j: int, delta: float) -> None:
            v = tau[i][j] + delta
            if v > tau_max:
                v = tau_max
            tau[i][j] = v

        def evaporate_tau() -> None:
            for i in range(n):
                row = tau[i]
                for j in range(n):
                    v = (1.0 - rho) * row[j]
                    if v < tau_min:
                        v = tau_min
                    row[j] = v

    else:
        class SparseTau:
            __slots__ = ("d",)

            def __init__(self):
                self.d = {}  # i -> {j: val}

            def get(self, i: int, j: int) -> float:
                row = self.d.get(i)
                if row is None:
                    return tau0
                return row.get(j, tau0)

            def add(self, i: int, j: int, delta: float) -> None:
                row = self.d.get(i)
                if row is None:
                    row = {}
                    self.d[i] = row
                v = row.get(j, tau0) + delta
                if v > tau_max:
                    v = tau_max
                row[j] = v

            def evaporate(self) -> None:
                d = self.d
                for i, row in list(d.items()):
                    for j, val in list(row.items()):
                        nv = (1.0 - rho) * val
                        if nv < tau_min:
                            # If baseline already >= tau_min, just drop to implicit baseline.
                            if tau0 <= tau_min:
                                row[j] = tau_min
                            else:
                                row.pop(j, None)
                        else:
                            row[j] = nv
                    if not row:
                        d.pop(i, None)

        stau = SparseTau()

        def tau_get(i: int, j: int) -> float:
            return stau.get(i, j)  # type: ignore

        def tau_add(i: int, j: int, delta: float) -> None:
            stau.add(i, j, delta)  # type: ignore

        def evaporate_tau() -> None:
            stau.evaporate()  # type: ignore

    def new_bin_score(item_w: int) -> float:
        # Slightly penalize opening new bins unless it packs tightly.
        if item_w > bin_capacity:
            return 0.0
        rem_after = bin_capacity - item_w
        eta = 1.0 / (1.0 + rem_after)
        return tau_new * (eta ** (0.25 * beta))

    def feasible_bin_score(item_idx: int, item_w: int, bin_items: List[int], bin_rem: int) -> float:
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

    def complete_with_ffd(remaining: List[int], bins: List[List[int]], bin_w: List[int], bin_rem: List[int]) -> None:
        # Deterministically pack remaining items into current bins then new bins.
        for idx in remaining:
            w = weights[idx]
            if w > bin_capacity:
                bins.append([idx])
                bin_w.append(w)
                bin_rem.append(0)
                continue
            placed = False
            for b in range(len(bins)):
                if bin_rem[b] >= w:
                    bins[b].append(idx)
                    bin_w[b] += w
                    bin_rem[b] -= w
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_w.append(w)
                bin_rem.append(bin_capacity - w)

    def construct_solution(order_: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        bin_w: List[int] = []
        bin_rem: List[int] = []

        t = 0
        while t < len(order_):
            # If time is up, complete deterministically so solution is always complete.
            if (t & 31) == 0 and now() >= deadline:
                complete_with_ffd(order_[t:], bins, bin_w, bin_rem)
                return bins, bin_w

            idx = order_[t]
            w = weights[idx]

            if w > bin_capacity:
                bins.append([idx])
                bin_w.append(w)
                bin_rem.append(0)
                t += 1
                continue

            feasible: List[Tuple[int, float]] = []
            for b in range(len(bins)):
                if bin_rem[b] >= w:
                    feasible.append((b, feasible_bin_score(idx, w, bins[b], bin_rem[b])))

            sc_new = new_bin_score(w)

            if not feasible:
                bins.append([idx])
                bin_w.append(w)
                bin_rem.append(bin_capacity - w)
                t += 1
                continue

            if random.random() < q0:
                best_b, best_sc = max(feasible, key=lambda x: x[1])
                if sc_new > best_sc:
                    bins.append([idx])
                    bin_w.append(w)
                    bin_rem.append(bin_capacity - w)
                else:
                    bins[best_b].append(idx)
                    bin_w[best_b] += w
                    bin_rem[best_b] -= w
            else:
                total = sc_new
                for _, sc in feasible:
                    total += sc
                # total should be > 0; but guard anyway.
                if total <= 0.0:
                    # fallback to best feasible
                    best_b, _ = max(feasible, key=lambda x: x[1])
                    bins[best_b].append(idx)
                    bin_w[best_b] += w
                    bin_rem[best_b] -= w
                    t += 1
                    continue

                r = random.random() * total
                acc = sc_new
                if r <= acc:
                    bins.append([idx])
                    bin_w.append(w)
                    bin_rem.append(bin_capacity - w)
                else:
                    chosen = feasible[-1][0]
                    for b, sc in feasible:
                        acc += sc
                        if r <= acc:
                            chosen = b
                            break
                    bins[chosen].append(idx)
                    bin_w[chosen] += w
                    bin_rem[chosen] -= w

            t += 1

        return bins, bin_w

    def evaporate() -> None:
        nonlocal tau_new
        tau_new = max(tau_min, (1.0 - rho) * tau_new)
        evaporate_tau()

    def reinforce(bins: List[List[int]], bin_w: List[int]) -> None:
        nonlocal tau_new
        m, waste, _ = solution_cost(bins, bin_w)
        delta = 1.0 / (1.0 + (m - 1) + 0.01 * waste)
        delta *= 4.0

        for b in bins:
            k = len(b)
            if k <= 1:
                continue
            for i in range(k):
                a = b[i]
                for j in range(i + 1, k):
                    c = b[j]
                    tau_add(a, c, delta)
                    tau_add(c, a, delta)

        tau_new = min(tau_max, tau_new + 0.15 * delta)

    # --- Main loop ---
    it = 0
    while it < max_iters:
        if now() >= deadline:
            break

        iter_best_bins = None
        iter_best_binw = None
        iter_best_cost = None

        # Diversification: mild perturbation of order
        if n <= 2000:
            order2 = order[:]
            swaps = 2 if n < 200 else 4
            for _ in range(swaps):
                a = random.randrange(n)
                b = random.randrange(n)
                if a > b:
                    a, b = b, a
                if abs(weights[order2[a]] - weights[order2[b]]) <= 1:
                    order2[a], order2[b] = order2[b], order2[a]
        else:
            order2 = order

        for _ in range(n_ants):
            if now() >= deadline:
                break

            bins_k, binw_k = construct_solution(order2)
            c = solution_cost(bins_k, binw_k)

            if (iter_best_cost is None) or (c < iter_best_cost):
                iter_best_cost = c
                iter_best_bins = bins_k
                iter_best_binw = binw_k

            if c < best_cost:
                best_cost = c
                best_bins = bins_k
                best_binw = binw_k

        evaporate()
        if iter_best_bins is not None:
            reinforce(iter_best_bins, iter_best_binw)  # type: ignore
        if random.random() < 0.25:
            reinforce(best_bins, best_binw)

        it += 1

    # --- Output (ensure completeness and correct weights) ---
    packing = [b[:] for b in best_bins]

    # Sanity: ensure every item appears exactly once; if not, repair quickly.
    seen = [False] * n
    for b in packing:
        for idx in b:
            if 0 <= idx < n:
                seen[idx] = True
    missing = [i for i, ok in enumerate(seen) if not ok]
    if missing:
        # add missing via FFD into existing bins
        bin_w = [sum(weights[i] for i in b) for b in packing]
        bin_rem = [max(0, bin_capacity - w) for w in bin_w]
        complete_with_ffd(missing, packing, bin_w, bin_rem)

    bin_weights_out = [sum(weights[idx] for idx in b) for b in packing]

    return {"packing": packing, "bin_weights": bin_weights_out}
