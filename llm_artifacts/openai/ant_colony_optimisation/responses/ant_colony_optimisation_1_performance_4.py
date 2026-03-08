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
    internal_limit = min(100.0, max(0.0, float(time_limit)))
    deadline = start + internal_limit

    order_desc = sorted(range(n), key=lambda i: w[i], reverse=True)
    order_asc = sorted(range(n), key=lambda i: w[i])
    total_w = sum(w)

    # Lower bounds
    lb1 = (total_w + C - 1) // C
    big = [i for i in range(n) if w[i] > C // 2]
    sum_small = total_w - sum(w[i] for i in big)
    lb2 = len(big) + (sum_small + C - 1) // C
    lb = max(lb1, lb2)

    # ------------------------
    # Packing given an item permutation: deterministic Best-Fit (strong for BPP)
    # ------------------------
    def pack_best_fit(seq: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins_items: List[List[int]] = []
        bins_rem: List[int] = []
        for it in seq:
            wi = w[it]
            best_j = -1
            best_after = None
            # scan all bins (quality > speed; n is moderate and we rely on time checks outside)
            for j, rem in enumerate(bins_rem):
                if rem >= wi:
                    after = rem - wi
                    if best_after is None or after < best_after:
                        best_after = after
                        best_j = j
                        if after == 0:
                            break
            if best_j == -1:
                bins_items.append([it])
                bins_rem.append(C - wi)
            else:
                bins_items[best_j].append(it)
                bins_rem[best_j] -= wi
        return bins_items, [C - r for r in bins_rem]

    def pack_first_fit(seq: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins_items: List[List[int]] = []
        bins_rem: List[int] = []
        for it in seq:
            wi = w[it]
            chosen = -1
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

    def slack_score(bin_weights: List[int]) -> int:
        s = 0
        for bw in bin_weights:
            d = C - bw
            s += d * d
        return s

    def sol_key(packing: List[List[int]], bin_weights: List[int]) -> Tuple[int, int]:
        return (len(packing), slack_score(bin_weights))

    # ------------------------
    # Strong initialization ensemble
    # ------------------------
    best_packing, best_bin_weights = pack_best_fit(order_desc)
    best_key = sol_key(best_packing, best_bin_weights)

    # More diverse BFD-style permutations: shuffle equal-weight blocks + random perturbations
    init_trials = 40 if n <= 400 else 28
    for t in range(init_trials):
        if (t & 3) == 0 and time.time() >= deadline:
            break

        if t == 0:
            seq = order_desc
            p, bw = pack_first_fit(seq)
        elif t == 1:
            seq = order_desc
            p, bw = pack_best_fit(seq)
        else:
            seq = order_desc[:]
            # shuffle equal-weight blocks
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

            # stronger randomization of top segment sometimes
            if n > 60 and (t % 3 == 0):
                top = min(n, 120)
                block = seq[:top]
                random.shuffle(block)
                seq[:top] = block

            # occasional swap noise
            if n > 40 and (t % 5 == 0):
                swaps = 8
                for _ in range(swaps):
                    a = random.randrange(min(n, 180))
                    b = random.randrange(n)
                    seq[a], seq[b] = seq[b], seq[a]

            p, bw = pack_best_fit(seq)

        key = sol_key(p, bw)
        if key < best_key:
            best_key = key
            best_packing, best_bin_weights = p, bw

    # ------------------------
    # Build sparse neighbor graph for pheromone on adjacency in the permutation
    # ------------------------
    # We want candidate next items that either:
    # - complement the last chosen item towards capacity
    # - are of similar size (helps multi-item fits)
    # - are globally large remaining

    if n <= 250:
        L = 120
    elif n <= 800:
        L = 90
    else:
        L = 65

    # helper: rightmost index in order_asc with weight <= limit
    def upper_pos(limit: int) -> int:
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if w[order_asc[mid]] <= limit:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    neighbors: List[List[int]] = [[] for _ in range(n)]
    neigh_set: List[set] = [set() for _ in range(n)]

    for i in order_desc:
        wi = w[i]
        # complements around C - wi
        limit = C - wi
        cand = []
        if limit > 0:
            pos = upper_pos(limit)
            # walk left taking near-best complements
            steps = 0
            while pos >= 0 and len(cand) < L and steps < L * 10:
                j = order_asc[pos]
                if j != i and w[j] <= limit:
                    cand.append(j)
                pos -= 1
                steps += 1

        # similar-size neighborhood (helps 3+ item combinations)
        # pick around the position of wi in ascending list
        # find approximate position by binary search for wi
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if w[order_asc[mid]] < wi:
                lo = mid + 1
            else:
                hi = mid
        posw = lo
        span = min(n, 2 * L)
        a = max(0, posw - span // 2)
        b = min(n, posw + span // 2)
        for k in range(a, b):
            j = order_asc[k]
            if j != i:
                cand.append(j)

        # a few random feasible complements
        if limit > 0:
            for _ in range(18):
                j = random.randrange(n)
                if j != i and w[j] <= limit:
                    cand.append(j)

        # unique and rank
        seen = set()
        uniq = []
        for j in cand:
            if j == i or j in seen:
                continue
            seen.add(j)
            uniq.append(j)

        # scoring: close to perfect complement is best; tie by larger weight
        uniq.sort(key=lambda j: (abs((wi + w[j]) - C), -w[j]))
        uniq = uniq[:L]

        # store directed neighbor list (for fast candidate union); pheromone stored undirected
        for j in uniq:
            if j not in neigh_set[i]:
                neighbors[i].append(j)
                neigh_set[i].add(j)

    # ------------------------
    # MMAS pheromones on adjacency edges (undirected)
    # ------------------------
    tau0 = 1.0
    TAU_MIN_FLOOR = 1e-8

    tau_edge: List[Dict[int, float]] = [dict() for _ in range(n)]

    def edge_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    # initialize edges only for neighbor pairs
    for i in range(n):
        for j in neighbors[i]:
            a, b = edge_key(i, j)
            if b not in tau_edge[a]:
                tau_edge[a][b] = tau0

    def get_tau(a: int, b: int) -> float:
        if a == b:
            return tau0
        x, y = edge_key(a, b)
        v = tau_edge[x].get(y)
        return v if v is not None else TAU_MIN_FLOOR

    # Heuristic for choosing next item given last item (and global size)
    def eta(last: int, x: int) -> float:
        # prefer larger items, and good complement with last
        wx = w[x]
        wl = w[last]
        # complement closeness
        comp = 1.0 / (1.0 + abs((wl + wx) - C))
        # size pressure
        sizeb = 0.20 + 0.80 * (wx / C) ** 1.3
        return comp * sizeb

    def cost_from_key(key: Tuple[int, int]) -> float:
        k, ss = key
        return (k - lb + 1) * (1.0 + ss / (C * C + 1.0))

    def strength_from_key(key: Tuple[int, int]) -> float:
        k, ss = key
        q_bins = 1.0 / (k - lb + 1)
        q_slack = 1.0 / (1.0 + ss / (C * C + 1.0))
        return 5.0 * q_bins * q_slack

    rho = 0.10

    def compute_tau_bounds(best_key_local: Tuple[int, int]) -> Tuple[float, float]:
        c = cost_from_key(best_key_local)
        tau_max = 1.0 / max(1e-12, rho * c)
        tau_min = max(TAU_MIN_FLOOR, tau_max / 200.0)
        return tau_min, tau_max

    tau_min, tau_max = compute_tau_bounds(best_key)

    def evaporate() -> None:
        keep = 1.0 - rho
        for a in range(n):
            row = tau_edge[a]
            for b in list(row.keys()):
                v = row[b] * keep
                if v < tau_min:
                    v = tau_min
                row[b] = v

    def deposit_from_sequence(seq: List[int], strength: float) -> None:
        # deposit on consecutive edges in the permutation
        add = strength
        for i in range(len(seq) - 1):
            a = seq[i]
            b = seq[i + 1]
            x, y = edge_key(a, b)
            row = tau_edge[x]
            if y in row:
                nv = row[y] + add
                row[y] = tau_max if nv > tau_max else nv

    def best_solution_sequence(packing: List[List[int]]) -> List[int]:
        # Convert packing into an induced order that reflects good bin groupings:
        # bins ordered by descending fill; within each bin, descending weights.
        bins = sorted(packing, key=lambda bin_items: -sum(w[i] for i in bin_items))
        seq: List[int] = []
        for b in bins:
            bb = sorted(b, key=lambda i: w[i], reverse=True)
            seq.extend(bb)
        return seq

    def restart_pheromones(preserve_best: bool = True) -> None:
        for a in range(n):
            row = tau_edge[a]
            for b in row.keys():
                row[b] = tau0
        if preserve_best and best_packing:
            sseq = best_solution_sequence(best_packing)
            deposit_from_sequence(sseq, 0.8 * strength_from_key(best_key))

    # ------------------------
    # ACO construction: build a permutation using candidate lists + MMAS/ACS choice
    # then pack with Best-Fit.
    # ------------------------
    # Parameters (fixed iteration count, time checked periodically)
    if n <= 200:
        iter_cap = 5200
        ants0 = 40
        cand_base = 26
    elif n <= 700:
        iter_cap = 4200
        ants0 = 28
        cand_base = 22
    else:
        iter_cap = 3200
        ants0 = 20
        cand_base = 18

    q0_start = 0.20
    q0_end = 0.75
    alpha_start, alpha_end = 1.0, 1.8
    beta_start, beta_end = 2.6, 1.8

    def choose_start(unused: List[bool]) -> int:
        # pick among a few heaviest unused
        cand = []
        for i in order_desc:
            if unused[i]:
                cand.append(i)
                if len(cand) >= 24:
                    break
        if not cand:
            for i in range(n):
                if unused[i]:
                    return i
            return 0
        # biased roulette by size
        tot = 0.0
        vals = []
        for x in cand:
            v = (w[x] / C) ** 2.2
            vals.append(v)
            tot += v
        r = random.random() * tot
        acc = 0.0
        for x, v in zip(cand, vals):
            acc += v
            if acc >= r:
                return x
        return cand[-1]

    def construct_permutation(elapsed_ratio: float) -> List[int]:
        alpha = alpha_start + (alpha_end - alpha_start) * elapsed_ratio
        beta = beta_start + (beta_end - beta_start) * elapsed_ratio
        q0 = q0_start + (q0_end - q0_start) * elapsed_ratio

        unused = [True] * n
        seq: List[int] = []

        cur = choose_start(unused)
        unused[cur] = False
        seq.append(cur)

        # also keep a small set of globally heavy candidates as fallback
        heavy_pool = order_desc[: min(n, 120)]

        while len(seq) < n:
            if (len(seq) & 63) == 0 and time.time() >= deadline:
                break

            # candidate union: neighbor list of cur + some heavy remaining + a few random
            cand = []
            cand.extend(neighbors[cur])

            # add heavy pool (cheap)
            for x in heavy_pool:
                if unused[x]:
                    cand.append(x)

            # random additions
            want_rand = 6
            tries = 0
            while want_rand > 0 and tries < 200:
                x = random.randrange(n)
                if unused[x]:
                    cand.append(x)
                    want_rand -= 1
                tries += 1

            # unique + unused, limit
            seen = set()
            uniq: List[int] = []
            for x in cand:
                if x in seen or not unused[x]:
                    continue
                seen.add(x)
                uniq.append(x)
                if len(uniq) >= cand_base:
                    break

            if not uniq:
                # pick any remaining (fallback)
                for x in order_desc:
                    if unused[x]:
                        nxt = x
                        break
                else:
                    break
                unused[nxt] = False
                seq.append(nxt)
                cur = nxt
                continue

            # ACS decision
            if random.random() < q0:
                best_x = None
                best_val = -1.0
                for x in uniq:
                    val = (get_tau(cur, x) ** alpha) * (eta(cur, x) ** beta)
                    if val > best_val:
                        best_val = val
                        best_x = x
                nxt = best_x if best_x is not None else uniq[-1]
            else:
                total = 0.0
                scores = []
                for x in uniq:
                    v = (get_tau(cur, x) ** alpha) * (eta(cur, x) ** beta)
                    if v < 0.0:
                        v = 0.0
                    scores.append(v)
                    total += v
                if total <= 0.0:
                    nxt = max(uniq, key=lambda z: w[z])
                else:
                    r = random.random() * total
                    acc = 0.0
                    nxt = uniq[-1]
                    for x, v in zip(uniq, scores):
                        acc += v
                        if acc >= r:
                            nxt = x
                            break

            unused[nxt] = False
            seq.append(nxt)
            cur = nxt

        # if timed out early, append remaining by decreasing weight
        if len(seq) < n:
            for x in order_desc:
                if x not in set(seq):
                    seq.append(x)
        return seq

    # ------------------------
    # Main loop (fixed iterations, time checked)
    # ------------------------
    it = 0
    since_improve = 0
    stagn_S = 70 if n <= 700 else 85

    while it < iter_cap:
        now = time.time()
        if now >= deadline:
            break

        elapsed_ratio = (now - start) / max(1e-9, (deadline - start))
        if elapsed_ratio < 0.0:
            elapsed_ratio = 0.0
        if elapsed_ratio > 1.0:
            elapsed_ratio = 1.0

        ants = max(10, int(ants0 * (1.20 - 0.55 * elapsed_ratio)))

        iter_best_key = None
        iter_best_pack = None
        iter_best_bw = None

        for a in range(ants):
            if (a & 3) == 0 and time.time() >= deadline:
                break
            seq = construct_permutation(elapsed_ratio)
            p, bw = pack_best_fit(seq)
            key = sol_key(p, bw)
            if iter_best_key is None or key < iter_best_key:
                iter_best_key = key
                iter_best_pack, iter_best_bw = p, bw

        if iter_best_key is None:
            break

        improved = False
        if iter_best_key < best_key:
            best_key = iter_best_key
            best_packing, best_bin_weights = iter_best_pack, iter_best_bw
            tau_min, tau_max = compute_tau_bounds(best_key)
            since_improve = 0
            improved = True
        else:
            since_improve += 1

        evaporate()

        # deposit policy: increasingly elitist
        p_elite = 0.40 + 0.55 * elapsed_ratio
        if improved or (random.random() < p_elite):
            dep_pack = best_packing
            dep_key = best_key
        else:
            dep_pack = iter_best_pack
            dep_key = iter_best_key

        strength = strength_from_key(dep_key)
        dep_seq = best_solution_sequence(dep_pack)
        deposit_from_sequence(dep_seq, strength)

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
        packing, bin_weights = pack_best_fit(order_desc)
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
            packing, bin_weights = pack_best_fit(order_desc)

    return {"packing": packing, "bin_weights": bin_weights}
