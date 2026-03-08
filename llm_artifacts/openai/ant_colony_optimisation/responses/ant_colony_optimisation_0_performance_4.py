import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    wts = weights

    start = time.time()
    # Allow up to 100 seconds hard stop as stated.
    hard_budget = min(100.0, max(0.0, float(time_limit)))
    deadline = start + hard_budget

    def now() -> float:
        return time.time()

    if C <= 0:
        packing = [[i] for i in range(n)]
        return {"packing": packing, "bin_weights": [wts[i] for i in range(n)]}

    # ---- ordering: classic for BPP ----
    order = list(range(n))
    order.sort(key=lambda i: wts[i], reverse=True)

    overweight = [i for i in range(n) if wts[i] > C]
    normal = [i for i in order if wts[i] <= C]

    total_normal_weight = sum(wts[i] for i in normal)
    lb = (total_normal_weight + C - 1) // C
    lb += len(overweight)

    # ---- helpers ----
    def compute_bin_weights(packing: List[List[int]]) -> List[int]:
        return [sum(wts[i] for i in b) for b in packing]

    def cost_tuple(packing: List[List[int]], bin_w: List[int]) -> Tuple[int, int, int]:
        # primary: bins; secondary: total waste; tertiary: max waste
        m = len(packing)
        total_waste = 0
        max_waste = 0
        for bw in bin_w:
            waste = C - bw
            if waste < 0:
                waste = 0
            total_waste += waste
            if waste > max_waste:
                max_waste = waste
        return (m, total_waste, max_waste)

    def bfd_solution(items: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        bw: List[int] = []
        rem: List[int] = []
        # forced overweight first
        for i in overweight:
            bins.append([i])
            bw.append(wts[i])
            rem.append(0)
        for idx in items:
            wi = wts[idx]
            if wi > C:
                continue
            best_b = -1
            best_after = 10**18
            for b in range(len(bins)):
                if rem[b] >= wi:
                    after = rem[b] - wi
                    if after < best_after:
                        best_after = after
                        best_b = b
                        if after == 0:
                            break
            if best_b >= 0:
                bins[best_b].append(idx)
                bw[best_b] += wi
                rem[best_b] -= wi
            else:
                bins.append([idx])
                bw.append(wi)
                rem.append(C - wi)
        return bins, bw

    # Baseline best-fit decreasing
    best_packing, best_binw = bfd_solution(normal)
    best_cost = cost_tuple(best_packing, best_binw)

    # ---- ACO (MMAS-like) pheromones over item-item (same-bin sequence) ----
    # We build each bin as a sequence; pheromone encourages certain consecutive pairs.
    dense_limit = 1400
    use_dense = (n <= dense_limit)

    # initial tau
    tau0 = 1.0 / max(1.0, float(n * max(1, best_cost[0])))

    # dynamic bounds will be updated from best solution
    tau_min = max(1e-9, 0.05 * tau0)
    tau_max = max(tau0 * 50.0, tau0 + 1e-9)

    if use_dense:
        tau_adj = [[tau0] * n for _ in range(n)]
        tau_start = [tau0] * n

        def t_get(i: int, j: int) -> float:
            return tau_adj[i][j]

        def t_set(i: int, j: int, v: float) -> None:
            if v < tau_min:
                v = tau_min
            elif v > tau_max:
                v = tau_max
            tau_adj[i][j] = v

        def ts_get(j: int) -> float:
            return tau_start[j]

        def ts_set(j: int, v: float) -> None:
            if v < tau_min:
                v = tau_min
            elif v > tau_max:
                v = tau_max
            tau_start[j] = v

        def evaporate(rho: float) -> None:
            mult = 1.0 - rho
            for i in range(n):
                row = tau_adj[i]
                for j in range(n):
                    nv = row[j] * mult
                    if nv < tau_min:
                        nv = tau_min
                    row[j] = nv
            for j in range(n):
                nv = tau_start[j] * mult
                if nv < tau_min:
                    nv = tau_min
                tau_start[j] = nv

        def reinit(blend: float) -> None:
            inv = 1.0 - blend
            for i in range(n):
                row = tau_adj[i]
                for j in range(n):
                    row[j] = row[j] * inv + tau0 * blend
            for j in range(n):
                tau_start[j] = tau_start[j] * inv + tau0 * blend

    else:
        class SparseAdj:
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
                if abs(v - tau0) <= 1e-15:
                    row.pop(j, None)
                    if not row:
                        self.d.pop(i, None)
                else:
                    row[j] = v

            def evaporate(self, rho: float) -> None:
                mult = 1.0 - rho
                for i, row in list(self.d.items()):
                    for j, val in list(row.items()):
                        nv = val * mult
                        if nv <= tau0:
                            row.pop(j, None)
                        else:
                            if nv < tau_min:
                                nv = tau_min
                            row[j] = nv
                    if not row:
                        self.d.pop(i, None)

            def reinit(self, blend: float) -> None:
                inv = 1.0 - blend
                for i, row in list(self.d.items()):
                    for j, val in list(row.items()):
                        nv = val * inv + tau0 * blend
                        if abs(nv - tau0) <= 1e-15:
                            row.pop(j, None)
                        else:
                            row[j] = nv
                    if not row:
                        self.d.pop(i, None)

        class SparseStart:
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

            def evaporate(self, rho: float) -> None:
                mult = 1.0 - rho
                for j, val in list(self.d.items()):
                    nv = val * mult
                    if nv <= tau0:
                        self.d.pop(j, None)
                    else:
                        if nv < tau_min:
                            nv = tau_min
                        self.d[j] = nv

            def reinit(self, blend: float) -> None:
                inv = 1.0 - blend
                for j, val in list(self.d.items()):
                    nv = val * inv + tau0 * blend
                    if abs(nv - tau0) <= 1e-15:
                        self.d.pop(j, None)
                    else:
                        self.d[j] = nv

        sadj = SparseAdj()
        sstart = SparseStart()

        def t_get(i: int, j: int) -> float:
            return sadj.get(i, j)

        def t_set(i: int, j: int, v: float) -> None:
            sadj.set(i, j, v)

        def ts_get(j: int) -> float:
            return sstart.get(j)

        def ts_set(j: int, v: float) -> None:
            sstart.set(j, v)

        def evaporate(rho: float) -> None:
            sadj.evaporate(rho)
            sstart.evaporate(rho)

        def reinit(blend: float) -> None:
            sadj.reinit(blend)
            sstart.reinit(blend)

    # ---- heuristic support: counts for complement bonuses ----
    max_w = max(wts)
    bucket_cap = min(C, max_w)
    count_by_w = [0] * (bucket_cap + 1)
    for i in normal:
        wi = wts[i]
        if 0 <= wi <= bucket_cap:
            count_by_w[wi] += 1

    def eta(residual: int, wi: int) -> float:
        # Strong best-fit pressure with exact-fit bonus and complement awareness
        left = residual - wi
        if left < 0:
            return 0.0
        # base best-fit: smaller left => larger
        # exponent tuned for BPP
        val = 1.0 / ((1.0 + left) ** 4.0)
        if left == 0:
            val *= 12.0
        elif left == 1:
            val *= 5.0
        elif left == 2:
            val *= 2.5
        # complement for the remaining space after placement
        if 0 <= left <= bucket_cap and count_by_w[left] > 0:
            val *= 1.4
        return val

    # ---- construct solution (ACS/MMAS-style decision) ----
    # Each bin is built sequentially; next item chosen from candidates that fit.
    # We keep a remaining set via boolean list.

    def construct(q0: float, alpha: float, beta: float, phi: float) -> Tuple[List[List[int]], List[int], List[Tuple[int, int]]]:
        remaining = [True] * n
        for i in overweight:
            remaining[i] = False

        # copy counts for complement heuristic
        cts = count_by_w[:]

        packing: List[List[int]] = []
        binw: List[int] = []
        edges: List[Tuple[int, int]] = []  # (-1,v) start edges and (u,v) adjacency

        # add overweight bins
        for i in overweight:
            packing.append([i])
            binw.append(wts[i])

        # list of remaining normal items count
        rem_normal = len(normal)

        # candidate parameters
        K = 32 if n <= 2000 else 20
        Lstart = 10 if n <= 1500 else 7

        # precomputed normal order list for scanning
        while rem_normal > 0:
            if now() >= deadline:
                break

            # choose bin starter among largest remaining
            starters: List[int] = []
            for idx in normal:
                if remaining[idx]:
                    starters.append(idx)
                    if len(starters) >= Lstart:
                        break

            if not starters:
                break

            if random.random() < q0:
                bestj = starters[0]
                bestsc = -1.0
                for j in starters:
                    sc = (ts_get(j) ** alpha) * ((wts[j] / C) ** 3.0)
                    if sc > bestsc:
                        bestsc = sc
                        bestj = j
                first = bestj
            else:
                scores = []
                tot = 0.0
                for j in starters:
                    sc = (ts_get(j) ** alpha) * ((wts[j] / C) ** 3.0)
                    scores.append(sc)
                    tot += sc
                if tot <= 0:
                    first = random.choice(starters)
                else:
                    r = random.random() * tot
                    acc = 0.0
                    first = starters[-1]
                    for j, sc in zip(starters, scores):
                        acc += sc
                        if r <= acc:
                            first = j
                            break

            # open bin with first
            remaining[first] = False
            rem_normal -= 1
            wi = wts[first]
            if 0 <= wi <= bucket_cap:
                cts[wi] -= 1

            packing.append([first])
            curw = wi
            residual = C - wi
            edges.append((-1, first))
            ts_set(first, (1.0 - phi) * ts_get(first) + phi * tau0)

            last = first

            # fill bin
            while residual > 0 and rem_normal > 0:
                # time check every few inner steps
                if (rem_normal & 127) == 0 and now() >= deadline:
                    break

                # Build candidate list: scan normal order and collect those that fit.
                cand: List[int] = []
                for idx in normal:
                    if not remaining[idx]:
                        continue
                    if wts[idx] <= residual:
                        cand.append(idx)
                        if len(cand) >= K:
                            break

                if not cand:
                    break

                # choose next item
                bestj = -1
                bestsc = -1.0
                scored: List[Tuple[int, float]] = []
                tot = 0.0
                for j in cand:
                    wj = wts[j]
                    et = eta(residual, wj)
                    if et <= 0.0:
                        continue
                    sc = (t_get(last, j) ** alpha) * (et ** beta)
                    scored.append((j, sc))
                    tot += sc
                    if sc > bestsc:
                        bestsc = sc
                        bestj = j

                if bestj < 0:
                    break

                if random.random() < q0:
                    chosen = bestj
                else:
                    if tot <= 0.0:
                        chosen = bestj
                    else:
                        r = random.random() * tot
                        acc = 0.0
                        chosen = scored[-1][0]
                        for j, sc in scored:
                            acc += sc
                            if r <= acc:
                                chosen = j
                                break

                if not remaining[chosen]:
                    continue

                wch = wts[chosen]
                if wch > residual:
                    continue

                remaining[chosen] = False
                rem_normal -= 1
                if 0 <= wch <= bucket_cap:
                    cts[wch] -= 1

                packing[-1].append(chosen)
                curw += wch
                residual -= wch

                edges.append((last, chosen))
                t_set(last, chosen, (1.0 - phi) * t_get(last, chosen) + phi * tau0)
                last = chosen

            binw.append(curw)

        # If stopped early, pack the rest quickly by BFD into existing bins.
        if rem_normal > 0:
            # build current rem list
            rem_caps = [max(0, C - bw) for bw in binw]
            for idx in normal:
                if remaining[idx]:
                    wi = wts[idx]
                    remaining[idx] = False
                    rem_normal -= 1
                    # best-fit into existing bins
                    bestb = -1
                    best_after = 10**18
                    for b in range(len(packing)):
                        if rem_caps[b] >= wi:
                            after = rem_caps[b] - wi
                            if after < best_after:
                                best_after = after
                                bestb = b
                                if after == 0:
                                    break
                    if bestb >= 0:
                        packing[bestb].append(idx)
                        binw[bestb] += wi
                        rem_caps[bestb] -= wi
                    else:
                        packing.append([idx])
                        binw.append(wi)
                        rem_caps.append(C - wi)

        return packing, binw, edges

    # ---- daemon action (standard in ACO BPP): limited repack of worst bins ----
    def daemon_repack(packing: List[List[int]], binw: List[int], max_bins_to_repack: int) -> Tuple[List[List[int]], List[int]]:
        # Try to eliminate some light bins by removing their items and reinserting with BFD.
        m = len(packing)
        if m <= 1:
            return packing, binw

        # choose candidate bins: lightest ones among normal bins (exclude overweight-only bins where weight>C)
        idxs = list(range(m))
        idxs.sort(key=lambda b: binw[b])

        to_repack = []
        for b in idxs:
            if now() >= deadline:
                break
            # skip bins containing overweight items (weight > C) - they are fixed
            if binw[b] > C:
                continue
            to_repack.append(b)
            if len(to_repack) >= max_bins_to_repack:
                break

        if not to_repack:
            return packing, binw

        remove_set = set(to_repack)
        items: List[int] = []
        new_p: List[List[int]] = []
        new_bw: List[int] = []

        for b in range(m):
            if b in remove_set:
                items.extend(packing[b])
            else:
                new_p.append(packing[b][:])
                new_bw.append(binw[b])

        # separate overweight items to keep singleton
        # (they should already be singleton, but ensure)
        fixed_over = [i for i in items if wts[i] > C]
        items2 = [i for i in items if wts[i] <= C]
        for i in fixed_over:
            new_p.append([i])
            new_bw.append(wts[i])

        # reinsert removed normal items by BFD into current bins
        # sort decreasing
        items2.sort(key=lambda i: wts[i], reverse=True)
        rem_caps = [max(0, C - bw) for bw in new_bw]
        for idx in items2:
            if now() >= deadline:
                break
            wi = wts[idx]
            bestb = -1
            best_after = 10**18
            for b in range(len(new_p)):
                if rem_caps[b] >= wi:
                    after = rem_caps[b] - wi
                    if after < best_after:
                        best_after = after
                        bestb = b
                        if after == 0:
                            break
            if bestb >= 0:
                new_p[bestb].append(idx)
                new_bw[bestb] += wi
                rem_caps[bestb] -= wi
            else:
                new_p.append([idx])
                new_bw.append(wi)
                rem_caps.append(C - wi)

        return new_p, new_bw

    # ---- global update ----
    def update_bounds_from_best(best_bins: int) -> None:
        nonlocal tau_min, tau_max
        # MMAS heuristic bounds
        # tau_max ~ 1/((1-rho)*cost)
        # we use cost = bins (primary objective)
        rho_eff = 0.12
        tau_max = 1.0 / max(1e-12, (rho_eff * float(best_bins)))
        tau_min = tau_max / 50.0
        if tau_min < 1e-9:
            tau_min = 1e-9

    def deposit(edges: List[Tuple[int, int]], packing: List[List[int]], binw: List[int], rho: float, strength: float) -> None:
        m, waste, _ = cost_tuple(packing, binw)
        # reward mainly by bin count, with mild waste component
        denom = float(m) + 0.15 * (float(waste) / max(1.0, float(C)))
        delta = strength / max(1e-12, denom)
        for u, v in edges:
            if u < 0:
                ts_set(v, ts_get(v) + delta)
            else:
                t_set(u, v, t_get(u, v) + delta)

    # ---- main loop ----
    # More aggressive search; still fixed iterations, time-checked.
    if n <= 400:
        n_ants = 44
        max_iters = 5000
    elif n <= 1200:
        n_ants = 30
        max_iters = 4500
    elif n <= 2500:
        n_ants = 22
        max_iters = 3800
    else:
        n_ants = 16
        max_iters = 3200

    # Parameters tuned for quality
    alpha = 1.0
    beta = 7.0
    rho = 0.12
    phi = 0.08
    q0 = 0.85

    best_edges: List[Tuple[int, int]] = []
    update_bounds_from_best(best_cost[0])

    no_improve = 0
    stagnation_limit = 80

    it = 0
    while it < max_iters:
        if now() >= deadline:
            break

        iter_best_p: Optional[List[List[int]]] = None
        iter_best_bw: Optional[List[int]] = None
        iter_best_edges: Optional[List[Tuple[int, int]]] = None
        iter_best_cost: Optional[Tuple[int, int, int]] = None

        # adapt exploration under stagnation
        if no_improve >= 40:
            q0_it = 0.70
            phi_it = 0.10
            rho_it = 0.14
        elif no_improve >= 15:
            q0_it = 0.78
            phi_it = 0.09
            rho_it = 0.13
        else:
            q0_it = q0
            phi_it = phi
            rho_it = rho

        for a in range(n_ants):
            if now() >= deadline:
                break

            p, bw, edges = construct(q0_it, alpha, beta, phi_it)

            # daemon action: bounded repack of a few worst bins
            # (kept small to preserve time for more ants/iters)
            if now() < deadline:
                max_repack = 2 if n > 1500 else 3
                p2, bw2 = daemon_repack(p, bw, max_bins_to_repack=max_repack)
                c2 = cost_tuple(p2, bw2)
                c1 = cost_tuple(p, bw)
                if c2 < c1:
                    p, bw = p2, bw2
                    # edges unchanged; acceptable for daemon action (common in ACO).

            c = cost_tuple(p, bw)
            if iter_best_cost is None or c < iter_best_cost:
                iter_best_cost = c
                iter_best_p = p
                iter_best_bw = bw
                iter_best_edges = edges

            if c < best_cost:
                best_cost = c
                best_packing = p
                best_binw = bw
                best_edges = edges
                no_improve = 0
                update_bounds_from_best(best_cost[0])
                # If we hit the lower bound, we can still continue but it’s likely optimal.
                if best_cost[0] <= lb:
                    # still allow slight time for confirmation but avoid wasting the whole budget
                    pass

        # evaporate
        evaporate(rho_it)

        # deposit: iteration-best always, best-so-far sometimes (MMAS style)
        if iter_best_p is not None and iter_best_edges is not None and iter_best_bw is not None:
            deposit(iter_best_edges, iter_best_p, iter_best_bw, rho_it, strength=2.2)

        if (it % 5) == 0 and best_edges:
            deposit(best_edges, best_packing, best_binw, rho_it, strength=1.3)

        # stagnation handling: partial reinit
        if iter_best_cost is None or iter_best_cost >= best_cost:
            no_improve += 1
        else:
            no_improve = 0

        if no_improve >= stagnation_limit:
            reinit(0.35)
            no_improve = 0

        it += 1

    # ---- final safety repair ----
    packing_out = [b[:] for b in best_packing]
    seen = [False] * n
    for b in packing_out:
        for idx in b:
            if 0 <= idx < n:
                seen[idx] = True
    missing = [i for i, ok in enumerate(seen) if not ok]
    if missing:
        # insert missing with BFD
        bw = compute_bin_weights(packing_out)
        rem = [max(0, C - x) for x in bw]
        missing.sort(key=lambda i: wts[i], reverse=True)
        for idx in missing:
            wi = wts[idx]
            if wi > C:
                packing_out.append([idx])
                bw.append(wi)
                rem.append(0)
                continue
            bestb = -1
            best_after = 10**18
            for b in range(len(packing_out)):
                if rem[b] >= wi:
                    after = rem[b] - wi
                    if after < best_after:
                        best_after = after
                        bestb = b
                        if after == 0:
                            break
            if bestb >= 0:
                packing_out[bestb].append(idx)
                bw[bestb] += wi
                rem[bestb] -= wi
            else:
                packing_out.append([idx])
                bw.append(wi)
                rem.append(C - wi)

    bin_weights_out = compute_bin_weights(packing_out)
    return {"packing": packing_out, "bin_weights": bin_weights_out}
