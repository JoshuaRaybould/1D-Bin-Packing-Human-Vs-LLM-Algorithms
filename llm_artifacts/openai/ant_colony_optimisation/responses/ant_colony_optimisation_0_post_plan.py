import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start_time = time.time()
    deadline = start_time + max(0.0, float(time_limit))

    def now() -> float:
        return time.time()

    C = int(bin_capacity)
    wts = weights

    # ---- helpers ----
    order_desc = list(range(n))
    order_desc.sort(key=lambda i: wts[i], reverse=True)

    def compute_bin_weights(packing: List[List[int]]) -> List[int]:
        return [sum(wts[i] for i in b) for b in packing]

    def solution_cost(packing: List[List[int]], bin_w: List[int]) -> Tuple[int, int, int, int]:
        # (bins, total_waste, max_waste, almost_full_count)
        m = len(packing)
        total_waste = 0
        max_waste = 0
        almost_full = 0
        for bw in bin_w:
            waste = C - bw
            if waste < 0:
                waste = 0
            total_waste += waste
            if waste > max_waste:
                max_waste = waste
            if waste <= 2:
                almost_full += 1
        return (m, total_waste, max_waste, -almost_full)

    # ---- baseline heuristics (FFD, BFD, randomized FFD) ----
    def ffd_with_order(ord_list: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        bin_w: List[int] = []
        rem: List[int] = []
        for idx in ord_list:
            wi = wts[idx]
            if wi > C:
                bins.append([idx])
                bin_w.append(wi)
                rem.append(0)
                continue
            placed = False
            for b in range(len(bins)):
                if rem[b] >= wi:
                    bins[b].append(idx)
                    bin_w[b] += wi
                    rem[b] -= wi
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_w.append(wi)
                rem.append(C - wi)
        return bins, bin_w

    def bfd_with_order(ord_list: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        bin_w: List[int] = []
        rem: List[int] = []
        for idx in ord_list:
            wi = wts[idx]
            if wi > C:
                bins.append([idx])
                bin_w.append(wi)
                rem.append(0)
                continue
            best_b = -1
            best_rem_after = 10**18
            for b in range(len(bins)):
                rb = rem[b]
                if rb >= wi:
                    ra = rb - wi
                    if ra < best_rem_after:
                        best_rem_after = ra
                        best_b = b
                        if ra == 0:
                            break
            if best_b >= 0:
                bins[best_b].append(idx)
                bin_w[best_b] += wi
                rem[best_b] -= wi
            else:
                bins.append([idx])
                bin_w.append(wi)
                rem.append(C - wi)
        return bins, bin_w

    def randomized_ffd_trials(trials: int) -> Tuple[List[List[int]], List[int]]:
        # Shuffle among equal weights only.
        # Keep it fast: make one pass to group indices by weight.
        by_w: Dict[int, List[int]] = {}
        for i in range(n):
            by_w.setdefault(wts[i], []).append(i)

        # Build canonical weight-sorted list of weights
        uniq_w = sorted(by_w.keys(), reverse=True)

        best_p: Optional[List[List[int]]] = None
        best_bw: Optional[List[int]] = None
        best_c: Optional[Tuple[int, int, int, int]] = None

        for _ in range(trials):
            if now() >= deadline:
                break
            ord_list: List[int] = []
            for ww in uniq_w:
                grp = by_w[ww]
                if len(grp) == 1:
                    ord_list.append(grp[0])
                else:
                    tmp = grp[:]
                    random.shuffle(tmp)
                    ord_list.extend(tmp)
            p, bw = ffd_with_order(ord_list)
            c = solution_cost(p, bw)
            if best_c is None or c < best_c:
                best_c = c
                best_p = p
                best_bw = bw

        if best_p is None:
            return ffd_with_order(order_desc)
        return best_p, best_bw  # type: ignore

    # Baseline set
    base_ffd_p, base_ffd_bw = ffd_with_order(order_desc)
    base_bfd_p, base_bfd_bw = bfd_with_order(order_desc)

    # Try some randomized FFD quickly (time-aware)
    # Keep trials modest; more is handled by ACS.
    trials = 20 if n <= 1200 else 10
    base_r_p, base_r_bw = randomized_ffd_trials(trials)

    best_packing, best_binw = base_ffd_p, base_ffd_bw
    best_cost = solution_cost(best_packing, best_binw)

    for p, bw in [(base_bfd_p, base_bfd_bw), (base_r_p, base_r_bw)]:
        c = solution_cost(p, bw)
        if c < best_cost:
            best_cost = c
            best_packing, best_binw = p, bw

    m0 = max(1, len(best_packing))

    # ---- ACO/ACS parameters (Plan section 8 + adaptive) ----
    alpha = 1.0
    beta = 6.0
    rho = 0.10
    q0_base = 0.80
    phi_base = 0.10

    # Candidate weights window; if capacity huge, keep a reasonable window.
    if C <= 0:
        # Degenerate: each item must be its own bin (or overweight bins)
        packing = [[i] for i in range(n)]
        return {"packing": packing, "bin_weights": [wts[i] for i in range(n)]}

    window = 60
    if C <= 120:
        window = 40
    elif C >= 2000:
        window = 80

    # Candidate list size cap (in addition to window weights)
    K_items_cap = 40 if n <= 2000 else 25

    # Iterations/ants (fixed max_iters, time checks stop early)
    max_iters = 5000
    if n <= 300:
        n_ants = 36
    elif n <= 800:
        n_ants = 28
    elif n <= 2000:
        n_ants = 20
    else:
        n_ants = 14

    # ---- Pheromones: adjacency + start ----
    dense_limit = 1200
    use_dense = n <= dense_limit

    # ACS init tau0 tied to baseline quality
    # Standard-ish: tau0 = 1/(n*m0); scaled and clamped
    tau0 = 1.0 / max(1.0, float(n * m0))

    tau_min = max(0.01 * tau0, 1e-6)
    tau_max = max(10.0 * tau0, tau0 + 1e-6)

    if use_dense:
        tau_adj = [[tau0] * n for _ in range(n)]
        tau_start = [tau0] * n

        def tau_get(i: int, j: int) -> float:
            return tau_adj[i][j]

        def tau_set(i: int, j: int, v: float) -> None:
            if v < tau_min:
                v = tau_min
            elif v > tau_max:
                v = tau_max
            tau_adj[i][j] = v

        def tau_start_get(j: int) -> float:
            return tau_start[j]

        def tau_start_set(j: int, v: float) -> None:
            if v < tau_min:
                v = tau_min
            elif v > tau_max:
                v = tau_max
            tau_start[j] = v

        def evaporate_global() -> None:
            mult = 1.0 - rho
            for i in range(n):
                row = tau_adj[i]
                for j in range(n):
                    v = row[j] * mult
                    if v < tau_min:
                        v = tau_min
                    row[j] = v
            for j in range(n):
                v = tau_start[j] * mult
                if v < tau_min:
                    v = tau_min
                tau_start[j] = v

        def partial_reset(blend: float) -> None:
            # tau <- (1-blend)*tau + blend*tau0
            inv = 1.0 - blend
            for i in range(n):
                row = tau_adj[i]
                for j in range(n):
                    row[j] = row[j] * inv + tau0 * blend
            for j in range(n):
                tau_start[j] = tau_start[j] * inv + tau0 * blend

    else:
        class SparseTauAdj:
            __slots__ = ("d",)

            def __init__(self):
                self.d: Dict[int, Dict[int, float]] = {}

            def get(self, i: int, j: int) -> float:
                row = self.d.get(i)
                if row is None:
                    return tau0
                return row.get(j, tau0)

            def set(self, i: int, j: int, v: float) -> None:
                if v < tau_min:
                    v = tau_min
                elif v > tau_max:
                    v = tau_max
                row = self.d.get(i)
                if row is None:
                    row = {}
                    self.d[i] = row
                # If close to tau0, drop to implicit to save memory
                if abs(v - tau0) <= 1e-15:
                    row.pop(j, None)
                    if not row:
                        self.d.pop(i, None)
                else:
                    row[j] = v

            def evaporate(self) -> None:
                mult = 1.0 - rho
                d = self.d
                for i, row in list(d.items()):
                    for j, val in list(row.items()):
                        nv = val * mult
                        if nv <= tau0 and tau0 > tau_min:
                            # let it fall back to implicit tau0
                            row.pop(j, None)
                        else:
                            if nv < tau_min:
                                nv = tau_min
                            row[j] = nv
                    if not row:
                        d.pop(i, None)

            def partial_reset(self, blend: float) -> None:
                # Only blend stored entries towards tau0.
                inv = 1.0 - blend
                d = self.d
                for i, row in list(d.items()):
                    for j, val in list(row.items()):
                        nv = val * inv + tau0 * blend
                        if abs(nv - tau0) <= 1e-15:
                            row.pop(j, None)
                        else:
                            row[j] = nv
                    if not row:
                        d.pop(i, None)

        class SparseTauStart:
            __slots__ = ("d",)

            def __init__(self):
                self.d: Dict[int, float] = {}

            def get(self, j: int) -> float:
                return self.d.get(j, tau0)

            def set(self, j: int, v: float) -> None:
                if v < tau_min:
                    v = tau_min
                elif v > tau_max:
                    v = tau_max
                if abs(v - tau0) <= 1e-15:
                    self.d.pop(j, None)
                else:
                    self.d[j] = v

            def evaporate(self) -> None:
                mult = 1.0 - rho
                for j, val in list(self.d.items()):
                    nv = val * mult
                    if nv <= tau0 and tau0 > tau_min:
                        self.d.pop(j, None)
                    else:
                        if nv < tau_min:
                            nv = tau_min
                        self.d[j] = nv

            def partial_reset(self, blend: float) -> None:
                inv = 1.0 - blend
                for j, val in list(self.d.items()):
                    nv = val * inv + tau0 * blend
                    if abs(nv - tau0) <= 1e-15:
                        self.d.pop(j, None)
                    else:
                        self.d[j] = nv

        stau_adj = SparseTauAdj()
        stau_start = SparseTauStart()

        def tau_get(i: int, j: int) -> float:
            return stau_adj.get(i, j)

        def tau_set(i: int, j: int, v: float) -> None:
            stau_adj.set(i, j, v)

        def tau_start_get(j: int) -> float:
            return stau_start.get(j)

        def tau_start_set(j: int, v: float) -> None:
            stau_start.set(j, v)

        def evaporate_global() -> None:
            stau_adj.evaporate()
            stau_start.evaporate()

        def partial_reset(blend: float) -> None:
            stau_adj.partial_reset(blend)
            stau_start.partial_reset(blend)

    # ---- Buckets for fast candidate generation ----
    max_w = max(wts) if wts else 0
    # Clamp bucket size; if weights huge, we still only need up to C.
    bucket_cap = max(0, min(max_w, C))
    buckets: List[List[int]] = [[] for _ in range(bucket_cap + 1)]

    # Items with weight > C are "overweight" and always forced to own bin.
    overweight_items: List[int] = []

    for i in range(n):
        wi = wts[i]
        if wi > C:
            overweight_items.append(i)
        elif wi >= 0:
            if wi <= bucket_cap:
                buckets[wi].append(i)

    # For each weight bucket, keep a pointer to next not-removed element.
    # We'll use a remaining flag plus per-bucket cursor for amortized O(1) pops.
    remaining = [True] * n
    remaining_count = n

    # Weight counts among remaining (for complement-aware heuristic)
    # Only for <=C weights.
    count_by_weight = [0] * (bucket_cap + 1)
    for ww in range(bucket_cap + 1):
        count_by_weight[ww] = len(buckets[ww])

    bucket_pos = [0] * (bucket_cap + 1)

    def remaining_remove(idx: int) -> None:
        nonlocal remaining_count
        if remaining[idx]:
            remaining[idx] = False
            remaining_count -= 1
            wi = wts[idx]
            if 0 <= wi <= bucket_cap:
                count_by_weight[wi] -= 1

    def bucket_take_weight_leq(r: int) -> Optional[int]:
        # Take best-fit deterministically: find heaviest weight <= r with any remaining item.
        if r > bucket_cap:
            r = bucket_cap
        for ww in range(r, -1, -1):
            if count_by_weight[ww] <= 0:
                continue
            pos = bucket_pos[ww]
            arr = buckets[ww]
            # advance to next remaining
            while pos < len(arr) and not remaining[arr[pos]]:
                pos += 1
            bucket_pos[ww] = pos
            if pos < len(arr):
                idx = arr[pos]
                # consume by advancing pos
                bucket_pos[ww] = pos + 1
                if remaining[idx]:
                    remaining_remove(idx)
                    return idx
        return None

    def build_candidates(r: int) -> List[int]:
        # Candidate list from near-tight weights: r, r-1, ..., r-window
        if r <= 0:
            return []
        rr = r
        if rr > bucket_cap:
            rr = bucket_cap
        lo = max(1, rr - window)
        cand: List[int] = []
        # Prefer exact/near-exact by scanning descending weights
        for ww in range(rr, lo - 1, -1):
            if count_by_weight[ww] <= 0:
                continue
            # collect a few available items from this bucket
            pos = bucket_pos[ww]
            arr = buckets[ww]
            # We do not advance the global cursor here permanently; just search locally.
            # But we can use it as a starting point.
            # Add up to 2 items per weight to diversify.
            added = 0
            p = pos
            while p < len(arr) and added < 2:
                idx = arr[p]
                if remaining[idx]:
                    cand.append(idx)
                    added += 1
                p += 1
            if len(cand) >= K_items_cap:
                break
        return cand

    def eta_from_residual(r: int, wi: int) -> float:
        # Tight fill convex heuristic + near-perfect bonus.
        left = r - wi
        if left < 0:
            return 0.0
        # base: 1/(1+left)^p, p~3
        base = 1.0 / ((1.0 + left) ** 3.0)
        if left <= 2:
            base *= (6.0 if left == 0 else (3.0 if left == 1 else 2.0))
        # complement-aware: if after placing wi, there exists an exact complement for new residual
        new_r = left
        if 0 <= new_r <= bucket_cap and count_by_weight[new_r] > 0:
            base *= 1.35
        return base

    def choose_starter(large_candidates: List[int], q0: float) -> int:
        # Exploration among top L largest remaining using tau_start and eta_start.
        # eta_start: weight/C (prefer large)
        # If no candidates (shouldn't happen), fallback to any remaining.
        if not large_candidates:
            for i in order_desc:
                if remaining[i] and wts[i] <= C:
                    return i
            # If only overweight remains, return one overweight (will be handled elsewhere)
            for i in range(n):
                if remaining[i]:
                    return i
            return 0

        if random.random() < q0:
            best_idx = large_candidates[0]
            best_sc = -1.0
            for j in large_candidates:
                sc = (tau_start_get(j) ** alpha) * ((wts[j] / C) ** 2.0)
                if sc > best_sc:
                    best_sc = sc
                    best_idx = j
            return best_idx

        scores: List[float] = []
        total = 0.0
        for j in large_candidates:
            sc = (tau_start_get(j) ** alpha) * ((wts[j] / C) ** 2.0)
            scores.append(sc)
            total += sc
        if total <= 0.0:
            return random.choice(large_candidates)
        r = random.random() * total
        acc = 0.0
        for j, sc in zip(large_candidates, scores):
            acc += sc
            if r <= acc:
                return j
        return large_candidates[-1]

    def local_update_start(j: int, phi: float) -> None:
        # tau = (1-phi)*tau + phi*tau0
        v = (1.0 - phi) * tau_start_get(j) + phi * tau0
        tau_start_set(j, v)

    def local_update_adj(i: int, j: int, phi: float) -> None:
        v = (1.0 - phi) * tau_get(i, j) + phi * tau0
        tau_set(i, j, v)

    def construct_solution(q0: float, phi: float) -> Tuple[List[List[int]], List[int], List[Tuple[int, int]]]:
        # Returns packing, bin_weights, and the used edges as (u,v) where u=-1 means start->v.
        # Work on shared remaining[]; the caller will provide a fresh remaining state per ant.
        packing: List[List[int]] = []
        bin_w: List[int] = []
        used_edges: List[Tuple[int, int]] = []

        # Put overweight items immediately (forced bins), but still mark as used.
        for idx in overweight_items:
            if remaining[idx]:
                remaining_remove(idx)
                packing.append([idx])
                bin_w.append(wts[idx])

        # Build bins until all non-overweight items placed.
        # Starter candidate list: top L largest remaining.
        L = 10 if n <= 2000 else 6

        step = 0
        while remaining_count > 0:
            # Time checks (hard requirement)
            if (step & 63) == 0 and now() >= deadline:
                break

            # Find up to L largest remaining items (<=C)
            large: List[int] = []
            for idx in order_desc:
                if remaining[idx] and wts[idx] <= C:
                    large.append(idx)
                    if len(large) >= L:
                        break
            if not large:
                # only overweight may remain (or nothing); handle deterministically
                for idx in range(n):
                    if remaining[idx]:
                        remaining_remove(idx)
                        packing.append([idx])
                        bin_w.append(wts[idx])
                break

            first = choose_starter(large, q0)
            if not remaining[first]:
                # rare race due to fallback; find any remaining feasible
                for idx in large:
                    if remaining[idx]:
                        first = idx
                        break

            wi = wts[first]
            remaining_remove(first)
            packing.append([first])
            cur_w = wi
            rcap = C - wi

            used_edges.append((-1, first))
            local_update_start(first, phi)

            last = first

            # Fill this bin
            while rcap > 0:
                if (step & 63) == 0 and now() >= deadline:
                    break

                cand = build_candidates(rcap)
                if not cand:
                    break

                # Evaluate candidates with adjacency pheromone from last
                best_j = -1
                best_sc = -1.0
                scores = []
                total = 0.0

                for j in cand:
                    wj = wts[j]
                    if wj > rcap:
                        continue
                    eta = eta_from_residual(rcap, wj)
                    if eta <= 0.0:
                        continue
                    sc = (tau_get(last, j) ** alpha) * (eta ** beta)
                    if sc > best_sc:
                        best_sc = sc
                        best_j = j
                    scores.append((j, sc))
                    total += sc

                if best_j < 0:
                    break

                if random.random() < q0:
                    chosen = best_j
                else:
                    if total <= 0.0:
                        chosen = best_j
                    else:
                        rr = random.random() * total
                        acc = 0.0
                        chosen = scores[-1][0]
                        for j, sc in scores:
                            acc += sc
                            if rr <= acc:
                                chosen = j
                                break

                if not remaining[chosen]:
                    # candidate became invalid; try to continue
                    continue

                wch = wts[chosen]
                if wch > rcap:
                    continue

                remaining_remove(chosen)
                packing[-1].append(chosen)
                cur_w += wch
                rcap -= wch

                used_edges.append((last, chosen))
                local_update_adj(last, chosen, phi)

                last = chosen
                step += 1

            bin_w.append(cur_w)
            step += 1

        return packing, bin_w, used_edges

    def complete_remaining_with_bfd(packing: List[List[int]]) -> None:
        # Deterministic completion: BFD into existing bins then new bins.
        bin_w = compute_bin_weights(packing)
        rem = [max(0, C - bw) for bw in bin_w]
        # pack remaining items in descending weight
        remaining_items = [i for i in order_desc if remaining[i]]
        for idx in remaining_items:
            wi = wts[idx]
            remaining_remove(idx)
            if wi > C:
                packing.append([idx])
                bin_w.append(wi)
                rem.append(0)
                continue
            best_b = -1
            best_rem_after = 10**18
            for b in range(len(packing)):
                if rem[b] >= wi:
                    ra = rem[b] - wi
                    if ra < best_rem_after:
                        best_rem_after = ra
                        best_b = b
                        if ra == 0:
                            break
            if best_b >= 0:
                packing[best_b].append(idx)
                bin_w[best_b] += wi
                rem[best_b] -= wi
            else:
                packing.append([idx])
                bin_w.append(wi)
                rem.append(C - wi)

    def global_reinforce(best_edges: List[Tuple[int, int]], packing: List[List[int]], bin_w: List[int], factor: float) -> None:
        # Deposit pheromone on used edges only.
        m, waste, _, _ = solution_cost(packing, bin_w)
        lam = 0.25
        denom = m + lam * (waste / max(1, C))
        Q = 1.0
        delta = factor * (Q / max(1e-9, denom))

        # Reinforce start edges and adjacency edges
        for u, v in best_edges:
            if u < 0:
                tau_start_set(v, tau_start_get(v) + delta)
            else:
                tau_set(u, v, tau_get(u, v) + delta)

    # ---- Main ACS loop with stagnation control ----
    best_edges_global: List[Tuple[int, int]] = []

    no_improve = 0
    elite_every = 7

    it = 0
    while it < max_iters:
        if now() >= deadline:
            break

        # Adaptive parameters under stagnation
        if no_improve >= 25:
            q0 = 0.60
            phi = 0.14
        elif no_improve >= 10:
            q0 = 0.70
            phi = 0.12
        else:
            q0 = q0_base
            phi = phi_base

        iter_best_p: Optional[List[List[int]]] = None
        iter_best_bw: Optional[List[int]] = None
        iter_best_edges: Optional[List[Tuple[int, int]]] = None
        iter_best_cost: Optional[Tuple[int, int, int, int]] = None

        # Run ants
        for _ in range(n_ants):
            if now() >= deadline:
                break

            # Reset remaining state for this ant
            for i in range(n):
                remaining[i] = True
            remaining_count = n
            for ww in range(bucket_cap + 1):
                count_by_weight[ww] = len(buckets[ww])
                bucket_pos[ww] = 0

            p, bw, edges = construct_solution(q0, phi)

            # If timeout or some items left, complete deterministically
            if remaining_count > 0:
                complete_remaining_with_bfd(p)
                bw = compute_bin_weights(p)

            c = solution_cost(p, bw)
            if iter_best_cost is None or c < iter_best_cost:
                iter_best_cost = c
                iter_best_p = p
                iter_best_bw = bw
                iter_best_edges = edges

            if c < best_cost:
                best_cost = c
                best_packing = p
                best_binw = bw
                best_edges_global = edges
                no_improve = 0
            else:
                no_improve += 0  # updated below per-iteration

        # Global evaporation
        evaporate_global()

        # Global update with iteration-best
        if iter_best_p is not None and iter_best_edges is not None:
            global_reinforce(iter_best_edges, iter_best_p, iter_best_bw, factor=2.0)  # type: ignore

        # Elite reinforcement occasionally
        if (it % elite_every) == 0 and best_edges_global:
            global_reinforce(best_edges_global, best_packing, best_binw, factor=1.0)

        # Stagnation control: partial pheromone reset
        if no_improve >= 35:
            partial_reset(0.30)
            no_improve = 0

        # Update stagnation counter per iteration: if we didn't improve during iteration
        if iter_best_cost is None or iter_best_cost >= best_cost:
            no_improve += 1
        else:
            no_improve = 0

        it += 1

    # ---- Final validity repair (safety) ----
    packing_out = [b[:] for b in best_packing]

    seen = [False] * n
    for b in packing_out:
        for idx in b:
            if 0 <= idx < n:
                seen[idx] = True
    missing = [i for i, ok in enumerate(seen) if not ok]
    if missing:
        # Repair with BFD from scratch into existing
        # Reuse a simple deterministic BFD insertion
        bin_w = compute_bin_weights(packing_out)
        rem = [max(0, C - bw) for bw in bin_w]
        # insert missing in descending weight
        missing.sort(key=lambda i: wts[i], reverse=True)
        for idx in missing:
            wi = wts[idx]
            if wi > C:
                packing_out.append([idx])
                bin_w.append(wi)
                rem.append(0)
                continue
            best_b = -1
            best_rem_after = 10**18
            for b in range(len(packing_out)):
                if rem[b] >= wi:
                    ra = rem[b] - wi
                    if ra < best_rem_after:
                        best_rem_after = ra
                        best_b = b
                        if ra == 0:
                            break
            if best_b >= 0:
                packing_out[best_b].append(idx)
                bin_w[best_b] += wi
                rem[best_b] -= wi
            else:
                packing_out.append([idx])
                bin_w.append(wi)
                rem.append(C - wi)

    bin_weights_out = compute_bin_weights(packing_out)
    return {"packing": packing_out, "bin_weights": bin_weights_out}
