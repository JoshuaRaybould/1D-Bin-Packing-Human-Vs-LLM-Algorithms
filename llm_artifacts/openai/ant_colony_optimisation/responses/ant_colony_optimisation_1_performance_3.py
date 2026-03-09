import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    w = weights

    # Infeasible safeguard
    for wi in w:
        if wi > C:
            packing = [[i] for i in range(n)]
            bin_weights = [w[i] for i in range(n)]
            return {"packing": packing, "bin_weights": bin_weights}

    start = time.time()
    # Allowed to run longer; cap at 100s hard wall
    internal_limit = min(100.0, max(0.0, float(time_limit)))
    deadline = start + internal_limit

    order = sorted(range(n), key=lambda i: w[i], reverse=True)
    total_w = sum(w)

    # Lower bounds
    lb1 = (total_w + C - 1) // C
    big = [i for i in range(n) if w[i] > C // 2]
    sum_small = total_w - sum(w[i] for i in big)
    lb2 = len(big) + (sum_small + C - 1) // C
    lb = max(lb1, lb2)

    # ------------------------
    # Fast packers for init and completion
    # ------------------------
    def pack_first_fit(seq: List[int], tie_random: bool = False) -> Tuple[List[List[int]], List[int]]:
        bins_items: List[List[int]] = []
        bins_rem: List[int] = []
        for it in seq:
            wi = w[it]
            chosen = -1
            if tie_random and bins_rem:
                feas = [j for j, rem in enumerate(bins_rem) if rem >= wi]
                if feas:
                    # choose among earliest feasible bins
                    k = min(7, len(feas))
                    chosen = random.choice(feas[:k])
            if chosen == -1:
                for j, rem in enumerate(bins_rem):
                    if rem >= wi:
                        chosen = j
                        break
            if chosen == -1:
                bins_items.append([it])
                bins_rem.append(C - wi)
            else:
                bins_items[chosen].append(it)
                bins_rem[chosen] -= wi
        return bins_items, [C - r for r in bins_rem]

    def pack_best_fit(seq: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins_items: List[List[int]] = []
        bins_rem: List[int] = []
        for it in seq:
            wi = w[it]
            best_j = -1
            best_after = None
            for j, rem in enumerate(bins_rem):
                if rem >= wi:
                    after = rem - wi
                    if best_after is None or after < best_after:
                        best_after = after
                        best_j = j
            if best_j == -1:
                bins_items.append([it])
                bins_rem.append(C - wi)
            else:
                bins_items[best_j].append(it)
                bins_rem[best_j] -= wi
        return bins_items, [C - r for r in bins_rem]

    def pack_bfd_with_window(seq: List[int], window: int = 20) -> Tuple[List[List[int]], List[int]]:
        # Best-fit decreasing with limited scan (speed)
        bins_items: List[List[int]] = []
        bins_rem: List[int] = []
        for it in seq:
            wi = w[it]
            best_j = -1
            best_after = None
            # scan last `window` bins first (often best candidates)
            startj = max(0, len(bins_rem) - window)
            for j in range(startj, len(bins_rem)):
                rem = bins_rem[j]
                if rem >= wi:
                    after = rem - wi
                    if best_after is None or after < best_after:
                        best_after = after
                        best_j = j
            if best_j == -1:
                # full scan fallback
                for j, rem in enumerate(bins_rem):
                    if rem >= wi:
                        after = rem - wi
                        if best_after is None or after < best_after:
                            best_after = after
                            best_j = j
            if best_j == -1:
                bins_items.append([it])
                bins_rem.append(C - wi)
            else:
                bins_items[best_j].append(it)
                bins_rem[best_j] -= wi
        return bins_items, [C - r for r in bins_rem]

    def slack_score(bin_weights: List[int]) -> int:
        s = 0
        for bw in bin_weights:
            d = C - bw
            s += d * d
        return s

    def sol_key(packing: List[List[int]], bin_weights: List[int]) -> Tuple[int, int]:
        return (len(packing), slack_score(bin_weights))

    # ------------------------
    # Initialization ensemble
    # ------------------------
    best_packing, best_bin_weights = pack_best_fit(order)
    best_key = sol_key(best_packing, best_bin_weights)

    init_trials = 18 if n <= 600 else 12
    for t in range(init_trials):
        if time.time() >= deadline:
            break
        if t == 0:
            p, bw = pack_first_fit(order)
        elif t == 1:
            p, bw = pack_bfd_with_window(order, window=25)
        else:
            # randomized within equal-weight blocks + occasional small perturbations
            seq = order[:]
            i = 0
            while i < n:
                j = i + 1
                wi = w[seq[i]]
                while j < n and w[seq[j]] == wi:
                    j += 1
                if j - i > 1:
                    block = seq[i:j]
                    random.shuffle(block)
                    seq[i:j] = block
                i = j
            # mild shuffle of top segment
            if n > 50 and (t % 3 == 0):
                top = min(n, 60)
                block = seq[:top]
                random.shuffle(block)
                seq[:top] = block
            if t % 2 == 0:
                p, bw = pack_best_fit(seq)
            else:
                p, bw = pack_first_fit(seq, tie_random=True)
        key = sol_key(p, bw)
        if key < best_key:
            best_packing, best_bin_weights, best_key = p, bw, key

    # ------------------------
    # Build sparse symmetric neighbor graph for pair pheromone
    # ------------------------
    asc = sorted(range(n), key=lambda i: w[i])

    if n <= 250:
        L = 90
    elif n <= 800:
        L = 65
    else:
        L = 45

    # helper: binary search rightmost <= limit
    def upper_pos(limit: int) -> int:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if w[asc[mid]] <= limit:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    neighbors: List[List[int]] = [[] for _ in range(n)]
    neigh_set: List[set] = [set() for _ in range(n)]

    # For each i, collect complements close to (C-w[i]) from asc.
    for i in order:
        limit = C - w[i]
        if limit <= 0:
            continue
        pos = upper_pos(limit)
        cand = []
        # take a bounded walk left
        steps = 0
        want = L
        while pos >= 0 and len(cand) < want and steps < want * 10:
            j = asc[pos]
            if j != i and w[j] <= limit:
                cand.append(j)
            pos -= 1
            steps += 1
        # add a few random feasible for diversity
        if limit > 0:
            for _ in range(10):
                j = random.randrange(n)
                if j != i and w[j] <= limit:
                    cand.append(j)

        # unique + sort by complement closeness
        seen = set()
        uniq = []
        for j in cand:
            if j == i or j in seen:
                continue
            seen.add(j)
            uniq.append(j)
        uniq.sort(key=lambda j: (abs((w[i] + w[j]) - C), -w[j]))
        uniq = uniq[:L]

        for j in uniq:
            a, b = (i, j) if i < j else (j, i)
            if a == b:
                continue
            if b not in neigh_set[a]:
                neighbors[a].append(b)
                neigh_set[a].add(b)

    # ------------------------
    # Pheromones: pair + residual class + (item, bin-position class)
    # ------------------------
    tau0 = 1.0
    TAU_MIN_FLOOR = 1e-6

    tau_pair: List[Dict[int, float]] = [dict() for _ in range(n)]
    for i in range(n):
        for j in neighbors[i]:
            tau_pair[i][j] = tau0

    # residual classes
    B = 28 if C >= 50 else 20

    def rclass(rem: int) -> int:
        if rem <= 0:
            return 0
        # rem in [1..C], map to 0..B-1
        return min(B - 1, (rem * B) // (C + 1))

    tau_res: List[List[float]] = [[tau0 for _ in range(B)] for _ in range(n)]

    # bin-position classes (encourage good early grouping): position is bin index rank among bins opened
    PCLS = 12
    tau_pos: List[List[float]] = [[tau0 for _ in range(PCLS)] for _ in range(n)]

    def pcl(bin_idx: int) -> int:
        # compress potentially large bin counts
        if bin_idx <= 0:
            return 0
        return min(PCLS - 1, bin_idx if bin_idx < PCLS - 2 else PCLS - 1)

    def get_tau(i: int, j: int) -> float:
        if i == j:
            return tau0
        a, b = (i, j) if i < j else (j, i)
        v = tau_pair[a].get(b)
        return v if v is not None else TAU_MIN_FLOOR

    def support_tau(x: int, bin_items: List[int]) -> float:
        s = 0.0
        cnt = 0
        for y in bin_items:
            if x == y:
                continue
            s += get_tau(x, y)
            cnt += 1
        return (s / cnt) if cnt else tau0

    def eta_item(x: int, rem: int) -> float:
        after = rem - w[x]
        # favor tight fits heavily, but keep some size bias
        tight = 1.0 / (1.0 + after)
        sizeb = 0.75 + 0.25 * (w[x] / C)
        # extra pressure to close bins when remainder is small
        if rem <= int(0.20 * C):
            tight *= 1.0 + 1.6 * (1.0 - rem / C)
        return tight * sizeb

    # ------------------------
    # MMAS/ACS parameters
    # ------------------------
    # Iterations: fixed cap; time checks will stop earlier.
    if n <= 200:
        iter_cap = 9000
        ants0 = 34
    elif n <= 600:
        iter_cap = 7000
        ants0 = 24
    else:
        iter_cap = 5200
        ants0 = 16

    rho = 0.12
    gamma_res = 0.9
    gamma_pos = 0.55

    # ACS-style exploitation probability
    q0_start = 0.25
    q0_end = 0.70

    # Candidate sizes
    K1, K2, K3 = 14, 14, 6
    P_partner = 9

    def cost_from_key(key: Tuple[int, int]) -> float:
        k, ss = key
        return (k - lb + 1) * (1.0 + ss / (C * C + 1.0))

    def strength_from_key(key: Tuple[int, int]) -> float:
        k, ss = key
        q_bins = 1.0 / (k - lb + 1)
        q_slack = 1.0 / (1.0 + ss / (C * C + 1.0))
        return 3.6 * q_bins * q_slack

    def compute_tau_bounds(best_key_local: Tuple[int, int]) -> Tuple[float, float]:
        c = cost_from_key(best_key_local)
        tau_max = 1.0 / max(1e-9, rho * c)
        tau_min = max(TAU_MIN_FLOOR, tau_max / 120.0)
        return tau_min, tau_max

    tau_min, tau_max = compute_tau_bounds(best_key)

    def evaporate_all() -> None:
        keep = 1.0 - rho
        for i in range(n):
            row = tau_pair[i]
            for j in row.keys():
                v = row[j] * keep
                row[j] = tau_min if v < tau_min else v
        for i in range(n):
            rr = tau_res[i]
            for b in range(B):
                v = rr[b] * keep
                rr[b] = tau_min if v < tau_min else v
        for i in range(n):
            pr = tau_pos[i]
            for b in range(PCLS):
                v = pr[b] * keep
                pr[b] = tau_min if v < tau_min else v

    def deposit_pairs(packing: List[List[int]], strength: float) -> None:
        for bin_items in packing:
            if len(bin_items) <= 1:
                continue
            items = sorted(bin_items, key=lambda i: w[i], reverse=True)
            m = len(items)
            for a_pos in range(m):
                a = items[a_pos]
                lim = min(m, a_pos + 1 + P_partner)
                for b_pos in range(a_pos + 1, lim):
                    b = items[b_pos]
                    i, j = (a, b) if a < b else (b, a)
                    row = tau_pair[i]
                    if j in row:
                        nv = row[j] + strength
                        row[j] = tau_max if nv > tau_max else nv

    def deposit_residual(place_orders: List[List[int]], strength: float) -> None:
        add = 0.60 * strength
        for ord_items in place_orders:
            rem = C
            for it in ord_items:
                rc = rclass(rem)
                nv = tau_res[it][rc] + add
                tau_res[it][rc] = tau_max if nv > tau_max else nv
                rem -= w[it]
                if rem <= 0:
                    break

    def deposit_position(packing: List[List[int]], strength: float) -> None:
        add = 0.35 * strength
        for bidx, bin_items in enumerate(packing):
            pc = pcl(bidx)
            for it in bin_items:
                nv = tau_pos[it][pc] + add
                tau_pos[it][pc] = tau_max if nv > tau_max else nv

    def restart_pheromones(preserve_best: bool = True) -> None:
        # Standard MMAS restart; optionally preserve some trails from best solution
        for i in range(n):
            row = tau_pair[i]
            for j in row.keys():
                row[j] = tau0
        for i in range(n):
            rr = tau_res[i]
            for b in range(B):
                rr[b] = tau0
        for i in range(n):
            pr = tau_pos[i]
            for b in range(PCLS):
                pr[b] = tau0

        if preserve_best and best_packing:
            s = strength_from_key(best_key)
            deposit_pairs(best_packing, 0.7 * s)
            approx_orders = [sorted(b, key=lambda i: w[i], reverse=True) for b in best_packing]
            deposit_residual(approx_orders, 0.7 * s)
            deposit_position(best_packing, 0.5 * s)

    asc_idx = asc

    def best_fit_candidates(unused: List[bool], rem: int, want: int) -> List[int]:
        res = []
        # binary search rightmost <= rem
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if w[asc_idx[mid]] <= rem:
                lo = mid + 1
            else:
                hi = mid
        pos = lo - 1
        steps = 0
        while pos >= 0 and len(res) < want and steps < want * 10:
            it = asc_idx[pos]
            if unused[it] and w[it] <= rem:
                res.append(it)
            pos -= 1
            steps += 1
        return res

    def pheromone_candidates(unused: List[bool], bin_items: List[int], rem: int, want: int) -> List[int]:
        cand: Dict[int, float] = {}
        for y in bin_items:
            # scan only stored edges where y is min index
            row = tau_pair[y] if y < n else None
            if row:
                for j, tv in row.items():
                    if unused[j] and w[j] <= rem:
                        cand[j] = cand.get(j, 0.0) + tv
        if not cand:
            return []
        items = list(cand.items())
        items.sort(key=lambda kv: kv[1], reverse=True)
        return [i for i, _ in items[:want]]

    def random_candidates(unused: List[bool], rem: int, want: int) -> List[int]:
        # reservoir-ish sampling without building huge list when n is large
        res = []
        tries = 0
        while len(res) < want and tries < want * 30:
            j = random.randrange(n)
            if unused[j] and w[j] <= rem:
                res.append(j)
            tries += 1
        return res

    def choose_seed(unused: List[bool]) -> int:
        # choose among top heaviest remaining; bias by pheromone degree
        cand = []
        for i in order:
            if unused[i]:
                cand.append(i)
                if len(cand) >= 18:
                    break
        if not cand:
            for i in range(n):
                if unused[i]:
                    return i
            return 0
        vals = []
        tot = 0.0
        for s in cand:
            base = (w[s] / C) ** 2.3
            deg = 0.0
            row = tau_pair[s]
            if row:
                k = 0
                for tv in row.values():
                    deg += tv
                    k += 1
                    if k >= 12:
                        break
            val = base * (1.0 + 0.02 * deg)
            vals.append(val)
            tot += val
        r = random.random() * tot
        acc = 0.0
        for s, v in zip(cand, vals):
            acc += v
            if acc >= r:
                return s
        return cand[-1]

    def construct_solution(elapsed_ratio: float) -> Tuple[List[List[int]], List[int], List[List[int]]]:
        # dynamic balance
        alpha = 0.8 + 0.8 * elapsed_ratio
        beta = 3.2 - 0.9 * elapsed_ratio
        q0 = q0_start + (q0_end - q0_start) * elapsed_ratio

        unused = [True] * n
        remaining = n
        packing: List[List[int]] = []
        bin_weights: List[int] = []
        placement_orders: List[List[int]] = []

        bidx = 0
        bins_rem: List[int] = []

        # Construct bins sequentially
        while remaining > 0:
            if (bidx & 15) == 0 and time.time() >= deadline:
                break

            s = choose_seed(unused)
            unused[s] = False
            remaining -= 1

            rem = C - w[s]
            bin_items = [s]
            place_order = [s]

            while rem > 0:
                cands = []
                cands.extend(best_fit_candidates(unused, rem, K1))
                cands.extend(pheromone_candidates(unused, bin_items, rem, K2))
                cands.extend(random_candidates(unused, rem, K3))

                seen = set()
                uniq = []
                for x in cands:
                    if x in seen:
                        continue
                    if unused[x] and w[x] <= rem:
                        seen.add(x)
                        uniq.append(x)
                if not uniq:
                    break

                rc = rclass(rem)
                pc = pcl(bidx)

                # ACS decision: exploit best with prob q0, else roulette
                if random.random() < q0:
                    best_x = None
                    best_val = -1.0
                    for x in uniq:
                        ts = support_tau(x, bin_items)
                        et = eta_item(x, rem)
                        tr = tau_res[x][rc]
                        tp = tau_pos[x][pc]
                        val = (ts ** alpha) * (et ** beta) * (tr ** gamma_res) * (tp ** gamma_pos)
                        if val > best_val:
                            best_val = val
                            best_x = x
                    x = best_x if best_x is not None else uniq[-1]
                else:
                    total = 0.0
                    scores = []
                    for x in uniq:
                        ts = support_tau(x, bin_items)
                        et = eta_item(x, rem)
                        tr = tau_res[x][rc]
                        tp = tau_pos[x][pc]
                        val = (ts ** alpha) * (et ** beta) * (tr ** gamma_res) * (tp ** gamma_pos)
                        if val <= 0.0:
                            val = 0.0
                        scores.append(val)
                        total += val
                    if total <= 0.0:
                        x = max(uniq, key=lambda z: w[z])
                    else:
                        r = random.random() * total
                        acc = 0.0
                        x = uniq[-1]
                        for cand_x, val in zip(uniq, scores):
                            acc += val
                            if acc >= r:
                                x = cand_x
                                break

                unused[x] = False
                remaining -= 1
                bin_items.append(x)
                place_order.append(x)
                rem -= w[x]

            packing.append(bin_items)
            bw = 0
            for it in bin_items:
                bw += w[it]
            bin_weights.append(bw)
            placement_orders.append(place_order)
            bins_rem.append(C - bw)
            bidx += 1

            # Daemon action (standard in ACO bin packing):
            # occasionally try to insert one of the largest remaining items into existing bins before opening new ones
            # This is still constructive and cheap.
            if remaining > 0 and (bidx % 9 == 0) and time.time() < deadline:
                # take up to a few heaviest leftovers and best-fit them into current bins
                moved = 0
                for it in order:
                    if not unused[it]:
                        continue
                    wi = w[it]
                    best_j = -1
                    best_after = None
                    for j, rrem in enumerate(bins_rem):
                        if rrem >= wi:
                            after = rrem - wi
                            if best_after is None or after < best_after:
                                best_after = after
                                best_j = j
                    if best_j != -1:
                        unused[it] = False
                        remaining -= 1
                        packing[best_j].append(it)
                        placement_orders[best_j].append(it)
                        bin_weights[best_j] += wi
                        bins_rem[best_j] -= wi
                        moved += 1
                        if moved >= 3:
                            break

        # complete greedily if timed out mid-construction
        if remaining > 0:
            leftovers = [i for i in range(n) if unused[i]]
            bins_rem = [C - bw for bw in bin_weights]
            for it in sorted(leftovers, key=lambda i: w[i], reverse=True):
                wi = w[it]
                best_j = -1
                best_after = None
                # best-fit into existing
                for j, rrem in enumerate(bins_rem):
                    if rrem >= wi:
                        after = rrem - wi
                        if best_after is None or after < best_after:
                            best_after = after
                            best_j = j
                if best_j == -1:
                    packing.append([it])
                    bin_weights.append(wi)
                    placement_orders.append([it])
                    bins_rem.append(C - wi)
                else:
                    packing[best_j].append(it)
                    placement_orders[best_j].append(it)
                    bin_weights[best_j] += wi
                    bins_rem[best_j] -= wi

        return packing, bin_weights, placement_orders

    # ------------------------
    # Main loop
    # ------------------------
    it = 0
    since_improve = 0

    if n <= 200:
        stagn_S = 55
    elif n <= 600:
        stagn_S = 70
    else:
        stagn_S = 85

    while it < iter_cap:
        now = time.time()
        if now >= deadline:
            break

        tl = max(1e-9, (deadline - start))
        elapsed_ratio = min(1.0, max(0.0, (now - start) / tl))

        # ants schedule (more early)
        ants = max(8, int(ants0 * (1.15 - 0.45 * elapsed_ratio)))

        iter_best_pack = None
        iter_best_bw = None
        iter_best_orders = None
        iter_best_key = None

        for _ in range(ants):
            if time.time() >= deadline:
                break
            p, bw, ords = construct_solution(elapsed_ratio)
            key = sol_key(p, bw)
            if iter_best_key is None or key < iter_best_key:
                iter_best_pack, iter_best_bw, iter_best_orders = p, bw, ords
                iter_best_key = key

        if iter_best_key is None:
            break

        improved = False
        if iter_best_key < best_key:
            best_key = iter_best_key
            best_packing = iter_best_pack
            best_bin_weights = iter_best_bw
            since_improve = 0
            improved = True
            tau_min, tau_max = compute_tau_bounds(best_key)
        else:
            since_improve += 1

        evaporate_all()

        # Deposit policy: increasingly elitist over time
        p_elite = 0.35 + 0.55 * elapsed_ratio
        if improved or (random.random() < p_elite):
            dep_pack = best_packing
            dep_orders = None
            dep_key = best_key
        else:
            dep_pack = iter_best_pack
            dep_orders = iter_best_orders
            dep_key = iter_best_key

        strength = strength_from_key(dep_key)
        deposit_pairs(dep_pack, strength)
        if dep_orders is None:
            approx_orders = [sorted(b, key=lambda i: w[i], reverse=True) for b in dep_pack]
            deposit_residual(approx_orders, strength)
        else:
            deposit_residual(dep_orders, strength)
        deposit_position(dep_pack, strength)

        if since_improve >= stagn_S:
            restart_pheromones(preserve_best=True)
            tau_min, tau_max = compute_tau_bounds(best_key)
            since_improve = 0

        it += 1

    # ------------------------
    # Output normalization / validation
    # ------------------------
    packing = [b for b in best_packing if b]
    bin_weights = [sum(w[i] for i in b) for b in packing]

    if not packing:
        packing, bin_weights = pack_first_fit(order)
    else:
        seen = [0] * n
        ok = True
        for bw, b in zip(bin_weights, packing):
            if bw > C:
                ok = False
                break
            for i in b:
                if i < 0 or i >= n:
                    ok = False
                    break
                seen[i] += 1
            if not ok:
                break
        if ok:
            for c in seen:
                if c != 1:
                    ok = False
                    break
        if not ok:
            packing, bin_weights = pack_first_fit(order)

    return {"packing": packing, "bin_weights": bin_weights}
