import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    w = weights

    # Handle infeasible instances gracefully
    for wi in w:
        if wi > C:
            packing = [[i] for i in range(n)]
            bin_weights = [w[i] for i in range(n)]
            return {"packing": packing, "bin_weights": bin_weights}

    start = time.time()
    deadline = start + max(0.0, float(time_limit))

    # Offline ordering
    order = sorted(range(n), key=lambda i: w[i], reverse=True)

    total_w = sum(w)

    # F2) stronger (still simple) lower bound
    lb1 = (total_w + C - 1) // C
    big = [i for i in range(n) if w[i] > C // 2]
    sum_small = total_w - sum(w[i] for i in big)
    lb2 = len(big) + (sum_small + C - 1) // C
    lb = max(lb1, lb2)

    # ------------------------
    # F1) heuristic ensemble for strong initial best
    # ------------------------
    def pack_first_fit(seq: List[int], tie_random: bool = False) -> Tuple[List[List[int]], List[int]]:
        bins_items: List[List[int]] = []
        bins_rem: List[int] = []
        for it in seq:
            wi = w[it]
            chosen = -1
            # scan in order; random tie-breaking among feasible bins with same rem is optional
            if tie_random:
                feas = [j for j, rem in enumerate(bins_rem) if rem >= wi]
                if feas:
                    # choose among the earliest few feasible bins to keep it FFD-like
                    k = min(5, len(feas))
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

    def pack_ffd_lookahead(seq: List[int]) -> Tuple[List[List[int]], List[int]]:
        # Best-fit-like with a mild bias to keep remainders near existing item weights
        bins_items: List[List[int]] = []
        bins_rem: List[int] = []
        for it in seq:
            wi = w[it]
            best_j = -1
            best_score = None
            for j, rem in enumerate(bins_rem):
                if rem >= wi:
                    after = rem - wi
                    # primary: tighter; secondary: remainder close to some item weight (encourage later closure)
                    # approximate by closeness to wi itself (cheap proxy)
                    score = (after, abs(after - wi))
                    if best_score is None or score < best_score:
                        best_score = score
                        best_j = j
            if best_j == -1:
                bins_items.append([it])
                bins_rem.append(C - wi)
            else:
                bins_items[best_j].append(it)
                bins_rem[best_j] -= wi
        return bins_items, [C - r for r in bins_rem]

    def slack_score(bin_weights: List[int]) -> int:
        # C2) correlate with reducibility: squared slack
        s = 0
        for bw in bin_weights:
            d = C - bw
            s += d * d
        return s

    def sol_key(packing: List[List[int]], bin_weights: List[int]) -> Tuple[int, int]:
        return (len(packing), slack_score(bin_weights))

    # Build a few fast variants
    best_packing, best_bin_weights = pack_first_fit(order)
    best_key = sol_key(best_packing, best_bin_weights)

    # small multi-start (kept small for speed)
    init_trials = 10 if n <= 600 else 7
    for t in range(init_trials):
        if time.time() >= deadline:
            break
        if t == 0:
            p, bw = pack_best_fit(order)
        elif t == 1:
            p, bw = pack_ffd_lookahead(order)
        else:
            # slight randomization in tie-breaking and/or perturb order among equal weights
            # create a stable-ish randomized sequence: group by weight then shuffle within group
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
            p, bw = pack_first_fit(seq, tie_random=True)
        key = sol_key(p, bw)
        if key < best_key:
            best_packing, best_bin_weights, best_key = p, bw, key

    best_k = best_key[0]

    # ------------------------
    # A1) Build sparse neighbor lists N(i) for pair pheromone
    # ------------------------
    # Neighbor size L
    if n <= 250:
        L = 70
    elif n <= 800:
        L = 50
    else:
        L = 35

    # Feasible complements lists by weight threshold; do a cheap candidate scan by sampling
    # We bias toward complementary weights (close to filling a bin)
    # Store only j > i to keep a canonical direction.

    # Precompute indices sorted by weight ascending for sampling small fits
    asc = sorted(range(n), key=lambda i: w[i])

    # Helper: pick candidates with w[j] <= limit, close to limit
    def best_complements(i: int, limit: int, want: int) -> List[int]:
        # Scan from the right of asc to find weights <= limit, taking closest weights
        # We do a bounded scan to avoid O(n^2)
        res = []
        # binary search for rightmost <= limit
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if w[asc[mid]] <= limit:
                lo = mid + 1
            else:
                hi = mid
        pos = lo - 1
        # walk left collecting distinct indices
        steps = 0
        while pos >= 0 and len(res) < want and steps < want * 6:
            j = asc[pos]
            if j != i and w[j] <= limit:
                res.append(j)
            pos -= 1
            steps += 1
        return res

    # Candidate random feasible neighbors for exploration
    feasible_by_limit_cache = {}

    def random_feasible(i: int, limit: int, want: int) -> List[int]:
        # Build a pool for this limit lazily (limits take only a few values typically)
        pool = feasible_by_limit_cache.get(limit)
        if pool is None:
            # include items with w <= limit
            pool = [j for j in range(n) if w[j] <= limit]
            feasible_by_limit_cache[limit] = pool
        if not pool:
            return []
        res = []
        # bounded attempts
        attempts = 0
        while len(res) < want and attempts < want * 10:
            j = random.choice(pool)
            if j != i:
                res.append(j)
            attempts += 1
        return res

    neighbors: List[List[int]] = [[] for _ in range(n)]
    neighbor_set: List[set] = [set() for _ in range(n)]

    for i in order:
        limit = C - w[i]
        # take complementary best fits
        cand = best_complements(i, limit, max(0, L - 10))
        # add some random feasible
        cand += random_feasible(i, limit, 12)
        # keep feasible and unique, prefer those closest to limit (largest weight)
        cand2 = []
        seen = set()
        for j in cand:
            if j == i or j in seen:
                continue
            if w[j] <= limit:
                seen.add(j)
                cand2.append(j)
        # sort by closeness to fill: minimize |(w[i]+w[j]) - C| i.e. maximize w[j] under limit
        cand2.sort(key=lambda j: (abs((w[i] + w[j]) - C), -w[j]))
        # keep top L
        cand2 = cand2[:L]

        # store only j>i
        for j in cand2:
            a, b = (i, j) if i < j else (j, i)
            if a == b:
                continue
            if b not in neighbor_set[a]:
                neighbors[a].append(b)
                neighbor_set[a].add(b)

    # ------------------------
    # A2) residual-class pheromone
    # ------------------------
    B = 20

    def rclass(rem: int) -> int:
        # rem in [0..C]
        # class based on rem/C
        if rem <= 0:
            return 0
        # use min(B-1, ...)
        return min(B - 1, (rem * B) // (C + 1))

    # ------------------------
    # MMAS pheromones
    # ------------------------
    tau0 = 1.0

    # tau_pair: list of dicts with only stored edges
    tau_pair: List[Dict[int, float]] = [dict() for _ in range(n)]
    for i in range(n):
        for j in neighbors[i]:
            tau_pair[i][j] = tau0

    tau_res: List[List[float]] = [[tau0 for _ in range(B)] for _ in range(n)]

    # MMAS parameters (E1/E3)
    if n <= 200:
        ants = 30
        stagn_S = 45
        iter_cap = 8000
    elif n <= 600:
        ants = 20
        stagn_S = 55
        iter_cap = 6000
    else:
        ants = 14
        stagn_S = 70
        iter_cap = 4500

    rho = 0.20
    gamma = 1.0  # residual pheromone importance

    p_best = 0.05  # used only for bounding ratio heuristic; we use simpler tau_min rule

    # Candidate sizes (B1)
    K1, K2, K3 = 10, 10, 5

    # Deposit pair limit (D1)
    P_partner = 7

    # Helper: get tau on an edge (canonical i<j)
    TAU_MIN_FLOOR = 1e-6

    def get_tau(i: int, j: int) -> float:
        if i == j:
            return tau0
        a, b = (i, j) if i < j else (j, i)
        v = tau_pair[a].get(b)
        return v if v is not None else TAU_MIN_FLOOR

    def support_tau(x: int, bin_items: List[int]) -> float:
        # aggregate (average) pair pheromone between x and items in bin
        s = 0.0
        cnt = 0
        for y in bin_items:
            if x == y:
                continue
            s += get_tau(x, y)
            cnt += 1
        if cnt == 0:
            return tau0
        return s / cnt

    def eta_item(x: int, rem: int) -> float:
        # tight fill + mild preference for larger items
        after = rem - w[x]
        # base tightness
        e = 1.0 / (1.0 + after)
        # size bias
        e *= 0.85 + 0.15 * (w[x] / C)
        # B2) controlled bin closing bias
        if rem <= int(0.15 * C):
            # boost tighter fits more when close to closing
            close_factor = 1.0 + 1.2 * (1.0 - rem / C)
            e *= close_factor
        return e

    def choose_seed(unused: List[bool], remaining: int) -> int:
        # pick among first few heaviest remaining with roulette weight^p
        # also bias by how much pheromone connectivity the item has (sum of stored tau)
        p = 2.0
        candidates = []
        for i in order:
            if unused[i]:
                candidates.append(i)
                if len(candidates) >= 12:
                    break
        if not candidates:
            # fallback scan
            for i in range(n):
                if unused[i]:
                    return i
            return 0
        vals = []
        total = 0.0
        for s in candidates:
            base = (w[s] / C) ** p
            # connectivity proxy
            conn = 0.0
            row = tau_pair[s] if s < n else None
            if row:
                # cap work
                k = 0
                for v in row.values():
                    conn += v
                    k += 1
                    if k >= 10:
                        break
            val = base * (1.0 + 0.03 * conn)
            vals.append(val)
            total += val
        r = random.random() * total
        acc = 0.0
        for s, val in zip(candidates, vals):
            acc += val
            if acc >= r:
                return s
        return candidates[-1]

    # Build a static list sorted by weight ascending for best-fit scanning windows
    # We will scan from the right for <= rem.
    asc_idx = asc

    def best_fit_candidates(unused: List[bool], rem: int, want: int) -> List[int]:
        # find up to want items with weight <= rem, closest to rem
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
        # among neighbors of items in current bin, pick feasible unused with high support
        cand_map: Dict[int, float] = {}
        for y in bin_items:
            a = y
            # check both directions: edges stored only for min index
            # gather from tau_pair[min][max]
            if a < n:
                # outgoing stored where a is min
                for j, tv in tau_pair[a].items():
                    if unused[j] and w[j] <= rem:
                        cand_map[j] = cand_map.get(j, 0.0) + tv
            # also edges where a is max: scan neighbors of smaller indices is expensive; skip.
            # The best-fit + random sample covers the missed direction.
        if not cand_map:
            return []
        items = list(cand_map.items())
        items.sort(key=lambda kv: kv[1], reverse=True)
        return [i for i, _ in items[:want]]

    def random_candidates(unused: List[bool], rem: int, want: int) -> List[int]:
        feas = [i for i in range(n) if unused[i] and w[i] <= rem]
        if not feas:
            return []
        if len(feas) <= want:
            return feas
        return random.sample(feas, want)

    # B1) construct bin-by-bin and also store placement order for residual deposit
    def construct_solution(elapsed_ratio: float) -> Tuple[List[List[int]], List[int], List[List[int]]]:
        # dynamic alpha/beta schedule (E3)
        # early: alpha low, beta high; later: alpha higher, beta lower
        alpha = 0.6 + 0.6 * elapsed_ratio
        beta = 3.5 - 1.0 * elapsed_ratio

        unused = [True] * n
        remaining = n
        packing: List[List[int]] = []
        bin_weights: List[int] = []
        placement_orders: List[List[int]] = []

        bins_done = 0
        # time check granularity: every few bins
        while remaining > 0:
            bins_done += 1
            if (bins_done & 7) == 0 and time.time() >= deadline:
                break

            s = choose_seed(unused, remaining)
            unused[s] = False
            remaining -= 1
            rem = C - w[s]
            bin_items = [s]
            place_order = [s]

            while True:
                if rem <= 0:
                    break

                cands = []
                # union of candidate sources
                cands.extend(best_fit_candidates(unused, rem, K1))
                cands.extend(pheromone_candidates(unused, bin_items, rem, K2))
                cands.extend(random_candidates(unused, rem, K3))

                # unique and feasible
                uniq = []
                seen = set()
                for x in cands:
                    if x in seen:
                        continue
                    if unused[x] and w[x] <= rem:
                        seen.add(x)
                        uniq.append(x)
                if not uniq:
                    break

                rc = rclass(rem)

                # roulette
                total = 0.0
                scores = []
                for x in uniq:
                    ts = support_tau(x, bin_items)
                    et = eta_item(x, rem)
                    tr = tau_res[x][rc]
                    val = (ts ** alpha) * (et ** beta) * (tr ** gamma)
                    # avoid zeros
                    if val <= 0.0:
                        val = 0.0
                    scores.append(val)
                    total += val

                if total <= 0.0:
                    # fallback: deterministic best-fit among uniq
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
            bin_weights.append(sum(w[i] for i in bin_items))
            placement_orders.append(place_order)

        # If timed out mid-construction, pack any leftovers greedily (feasible completion)
        if remaining > 0:
            leftovers = [i for i in range(n) if unused[i]]
            # place leftovers by best-fit into existing bins, else new
            bins_rem = [C - bw for bw in bin_weights]
            for it in sorted(leftovers, key=lambda i: w[i], reverse=True):
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

    # C2) cost and deposit strength
    def cost_from_key(key: Tuple[int, int]) -> float:
        k, ss = key
        # avoid division by zero
        return (k - lb + 1) * (1.0 + ss / (C * C + 1.0))

    def strength_from_key(key: Tuple[int, int]) -> float:
        k, ss = key
        q_bins = 1.0 / (k - lb + 1)
        q_slack = 1.0 / (1.0 + ss / (C * C + 1.0))
        return 3.0 * q_bins * q_slack

    # C1) tau bounds (simple robust)
    def compute_tau_bounds(best_key_local: Tuple[int, int]) -> Tuple[float, float]:
        # tau_max proportional to 1/(rho*cost)
        c = cost_from_key(best_key_local)
        tau_max = 1.0 / max(1e-9, rho * c)
        # robust min ratio
        tau_min = tau_max / 80.0
        if tau_min < TAU_MIN_FLOOR:
            tau_min = TAU_MIN_FLOOR
        return tau_min, tau_max

    tau_min, tau_max = compute_tau_bounds(best_key)

    # Evaporate + clamp
    def evaporate_all() -> None:
        keep = 1.0 - rho
        for i in range(n):
            row = tau_pair[i]
            for j in list(row.keys()):
                v = row[j] * keep
                if v < tau_min:
                    v = tau_min
                row[j] = v
        for i in range(n):
            rrow = tau_res[i]
            for b in range(B):
                v = rrow[b] * keep
                if v < tau_min:
                    v = tau_min
                rrow[b] = v

    # D1) pairwise deposit (limited partners)
    def deposit_pairs(packing: List[List[int]], strength: float) -> None:
        for bin_items in packing:
            if len(bin_items) <= 1:
                continue
            # sort by weight desc to focus on impactful items
            items = sorted(bin_items, key=lambda i: w[i], reverse=True)
            m = len(items)
            for a_pos in range(m):
                a = items[a_pos]
                # deposit with up to P_partner following
                lim = min(m, a_pos + 1 + P_partner)
                for b_pos in range(a_pos + 1, lim):
                    b = items[b_pos]
                    i, j = (a, b) if a < b else (b, a)
                    # update only if stored
                    if j in tau_pair[i]:
                        nv = tau_pair[i][j] + strength
                        if nv > tau_max:
                            nv = tau_max
                        tau_pair[i][j] = nv

    # D2) residual deposit
    def deposit_residual(place_orders: List[List[int]], strength: float) -> None:
        for ord_items in place_orders:
            rem = C
            for it in ord_items:
                rc = rclass(rem)
                nv = tau_res[it][rc] + 0.6 * strength
                if nv > tau_max:
                    nv = tau_max
                tau_res[it][rc] = nv
                rem -= w[it]
                if rem <= 0:
                    break

    # C3) restart
    def restart_pheromones() -> None:
        for i in range(n):
            row = tau_pair[i]
            for j in row.keys():
                row[j] = tau0
        for i in range(n):
            rrow = tau_res[i]
            for b in range(B):
                rrow[b] = tau0

    # ------------------------
    # Main MMAS loop
    # ------------------------
    it = 0
    since_improve = 0

    while it < iter_cap:
        now = time.time()
        if now >= deadline:
            break

        # elapsed ratio for alpha/beta schedule
        tl = max(1e-9, (deadline - start))
        elapsed_ratio = min(1.0, max(0.0, (now - start) / tl))

        iter_best_pack = None
        iter_best_bw = None
        iter_best_orders = None
        iter_best_key = None

        # E2) time check per ant (not per item); construction checks per few bins
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
            best_k = best_key[0]
            since_improve = 0
            improved = True
            # update tau bounds based on new best
            tau_min, tau_max = compute_tau_bounds(best_key)
        else:
            since_improve += 1

        # MMAS update
        evaporate_all()

        # Choose deposit source (C1)
        # early: iteration-best; later: best-so-far more often
        p_elite = 0.45 + 0.45 * elapsed_ratio
        if improved:
            dep_pack, dep_orders, dep_key = iter_best_pack, iter_best_orders, iter_best_key
        else:
            if random.random() < p_elite:
                dep_pack, dep_orders, dep_key = best_packing, None, best_key
            else:
                dep_pack, dep_orders, dep_key = iter_best_pack, iter_best_orders, iter_best_key

        strength = strength_from_key(dep_key)
        deposit_pairs(dep_pack, strength)
        if dep_orders is not None:
            deposit_residual(dep_orders, strength)
        else:
            # approximate heavy-first order for best_packing
            approx_orders = [sorted(b, key=lambda i: w[i], reverse=True) for b in dep_pack]
            deposit_residual(approx_orders, strength)

        # Stagnation restart
        if since_improve >= stagn_S:
            restart_pheromones()
            # keep bounds consistent
            tau_min, tau_max = compute_tau_bounds(best_key)
            since_improve = 0

        # If we already hit LB, we can still use time but stop early is fine; keep iterating.
        it += 1

    # ------------------------
    # Output normalization and correctness
    # ------------------------
    packing = [b for b in best_packing if b]
    bin_weights = [sum(w[i] for i in b) for b in packing]

    # Final sanity: ensure feasibility and all items exactly once
    # (Keep it lightweight)
    # If something went wrong, fall back to FFD
    if len(packing) == 0:
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
