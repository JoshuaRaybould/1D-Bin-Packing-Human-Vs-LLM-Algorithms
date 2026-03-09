import time
import math
import random
from typing import List, Tuple, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = int(bin_capacity)
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    start = time.perf_counter()
    deadline = start + max(0.0, float(time_limit))

    def now() -> float:
        return time.perf_counter()

    # ---------------- Objective (lexicographic) ----------------
    # Primary: B = # non-empty bins
    # Secondary: S = sum(slack^2) over non-empty bins
    # SA energy: E = B*K + S
    K = max(1, (C * C) * 16)

    def phi_slack(sl: int) -> int:
        return sl * sl

    # ---------------- Utilities ----------------
    order_desc = list(range(n))
    order_desc.sort(key=lambda i: weights[i], reverse=True)

    total_w = sum(weights)
    LB = (total_w + C - 1) // C

    # Build bins/loads/assign/pos from an insertion order using a heuristic variant.
    def build_initial(variant: int) -> Tuple[List[List[int]], List[int], List[int], List[int]]:
        bins: List[List[int]] = []
        loads: List[int] = []
        assign = [-1] * n
        pos = [-1] * n

        # Several orderings (standard trick):
        # 0: pure descending
        # 1: shuffle within equal blocks
        # 2: random-key order biased by weight
        # 3: descending but with occasional swaps
        if variant == 0:
            items = order_desc
        elif variant == 1:
            items = order_desc[:]
            i = 0
            while i < n:
                j = i + 1
                wi = weights[items[i]]
                while j < n and weights[items[j]] == wi:
                    j += 1
                if j - i > 1:
                    random.shuffle(items[i:j])
                i = j
        elif variant == 2:
            # random keys: heavier items tend to earlier positions
            # key = rank + noise scaled by (1 - w/C)
            items = order_desc[:]
            keys = []
            for r, it in enumerate(items):
                w = weights[it]
                noise = random.random() * (0.35 + 0.65 * (1.0 - min(1.0, w / max(1, C))))
                keys.append((r + noise, it))
            keys.sort(key=lambda t: t[0])
            items = [it for _, it in keys]
        else:
            items = order_desc[:]
            # a few random adjacent swaps
            for _ in range(min(3 * n, 2000)):
                a = random.randrange(n)
                b = a + 1
                if b < n and random.random() < 0.25:
                    items[a], items[b] = items[b], items[a]

        # Placement rule variants:
        # 0/1: Best-Fit
        # 2: First-Fit
        # 3: Best-Fit with mild randomness in key
        place = variant % 4

        for it in items:
            w = weights[it]
            best_b = -1
            best_key = None

            m = len(bins)
            if m:
                if place == 2:
                    # First-Fit with random start
                    s = random.randrange(m)
                    for t in range(m):
                        b = (s + t) % m
                        if loads[b] + w <= C:
                            best_b = b
                            break
                else:
                    # Best-Fit (sample + full fallback)
                    tries = min(18, m)
                    seen = set()
                    for _ in range(tries):
                        b = random.randrange(m)
                        if b in seen:
                            continue
                        seen.add(b)
                        if loads[b] + w <= C:
                            res = C - (loads[b] + w)
                            key = res
                            if place == 3:
                                key = res + random.random() * 0.35
                            if best_key is None or key < best_key:
                                best_key = key
                                best_b = b
                    if best_b == -1:
                        for b in range(m):
                            if loads[b] + w <= C:
                                res = C - (loads[b] + w)
                                key = res
                                if place == 3:
                                    key = res
                                if best_key is None or key < best_key:
                                    best_key = key
                                    best_b = b

            if best_b == -1:
                best_b = len(bins)
                bins.append([])
                loads.append(0)

            assign[it] = best_b
            pos[it] = len(bins[best_b])
            bins[best_b].append(it)
            loads[best_b] += w

        return bins, loads, assign, pos

    def contrib(L: int) -> int:
        if L <= 0:
            return 0
        return phi_slack(C - L)

    def compute_BS(loads: List[int]) -> Tuple[int, int]:
        B = 0
        S = 0
        for L in loads:
            if L > 0:
                B += 1
                S += phi_slack(C - L)
        return B, S

    # ---------------- Initial solutions ----------------
    init_trials = 24
    best_bins = None
    best_loads = None
    best_assign0 = None
    best_pos0 = None
    best_B = None
    best_S = None

    for v in range(init_trials):
        bins, loads, assign, pos = build_initial(v)
        B, S = compute_BS(loads)
        if best_B is None or (B, S) < (best_B, best_S):
            best_bins, best_loads, best_assign0, best_pos0 = bins, loads, assign, pos
            best_B, best_S = B, S
        if best_B == LB or now() >= deadline:
            break

    bins = best_bins
    loads = best_loads
    assign = best_assign0
    pos = best_pos0
    cur_B, cur_S = best_B, best_S

    best_assign = assign[:]  # store best assignment
    best_B, best_S = cur_B, cur_S

    nbins = len(bins)

    # ---------------- O(1) bin ops ----------------
    def remove_item(item: int, b: int) -> None:
        idx = pos[item]
        last = bins[b][-1]
        bins[b][idx] = last
        pos[last] = idx
        bins[b].pop()
        pos[item] = -1

    def add_item(item: int, b: int) -> None:
        pos[item] = len(bins[b])
        bins[b].append(item)

    # ---------------- Target bin selection (stronger) ----------------
    def pick_target_bin(b_from: int, w: int, tries: int = 28) -> int:
        # choose feasible bin with tightest residual after insertion (best-fit)
        best_b = -1
        best_res = None
        m = nbins
        for _ in range(tries):
            b = random.randrange(m)
            if b == b_from:
                continue
            L = loads[b]
            if L + w <= C:
                res = C - (L + w)
                # prefer filling non-empty bins, but allow empty bins too (escape mechanism)
                bias = 0 if L > 0 else 2
                key = res * 3 + bias
                if best_res is None or key < best_res:
                    best_res = key
                    best_b = b
        if best_b != -1:
            return best_b

        # small deterministic scan of a few candidates
        # (helps when random misses feasibility)
        step = 1 + random.randrange(7)
        s = random.randrange(m)
        for t in range(min(m, 48)):
            b = (s + t * step) % m
            if b == b_from:
                continue
            if loads[b] + w <= C:
                return b
        return -1

    # ---------------- Incremental objective updates ----------------
    def apply_move(item: int, b_from: int, b_to: int) -> Tuple[int, int]:
        nonlocal cur_B, cur_S
        w = weights[item]
        Lf0 = loads[b_from]
        Lt0 = loads[b_to]
        S0 = contrib(Lf0) + contrib(Lt0)

        remove_item(item, b_from)
        loads[b_from] = Lf0 - w
        add_item(item, b_to)
        loads[b_to] = Lt0 + w
        assign[item] = b_to

        dB = 0
        if Lf0 > 0 and loads[b_from] == 0:
            dB -= 1
        if Lt0 == 0 and loads[b_to] > 0:
            dB += 1

        S1 = contrib(loads[b_from]) + contrib(loads[b_to])
        dS = S1 - S0
        cur_B += dB
        cur_S += dS
        return dB, dS

    def undo_move(item: int, b_from: int, b_to: int) -> None:
        nonlocal cur_B, cur_S
        w = weights[item]
        Lf0 = loads[b_from]
        Lt0 = loads[b_to]
        S0 = contrib(Lf0) + contrib(Lt0)

        remove_item(item, b_to)
        loads[b_to] = Lt0 - w
        add_item(item, b_from)
        loads[b_from] = Lf0 + w
        assign[item] = b_from

        if Lf0 == 0 and loads[b_from] > 0:
            cur_B += 1
        if Lt0 > 0 and loads[b_to] == 0:
            cur_B -= 1

        S1 = contrib(loads[b_from]) + contrib(loads[b_to])
        cur_S += (S1 - S0)

    def apply_swap(i1: int, b1: int, i2: int, b2: int) -> int:
        nonlocal cur_S
        w1 = weights[i1]
        w2 = weights[i2]
        L10 = loads[b1]
        L20 = loads[b2]
        S0 = contrib(L10) + contrib(L20)

        remove_item(i1, b1)
        loads[b1] = L10 - w1
        remove_item(i2, b2)
        loads[b2] = L20 - w2

        add_item(i2, b1)
        loads[b1] += w2
        add_item(i1, b2)
        loads[b2] += w1

        assign[i1], assign[i2] = b2, b1

        S1 = contrib(loads[b1]) + contrib(loads[b2])
        dS = S1 - S0
        cur_S += dS
        return dS

    def undo_swap(i1: int, b1: int, i2: int, b2: int) -> None:
        nonlocal cur_S
        w1 = weights[i1]
        w2 = weights[i2]
        L10 = loads[b1]
        L20 = loads[b2]
        S0 = contrib(L10) + contrib(L20)

        # currently i2 in b1, i1 in b2
        remove_item(i2, b1)
        loads[b1] = L10 - w2
        remove_item(i1, b2)
        loads[b2] = L20 - w1

        add_item(i1, b1)
        loads[b1] += w1
        add_item(i2, b2)
        loads[b2] += w2

        assign[i1], assign[i2] = b1, b2

        S1 = contrib(loads[b1]) + contrib(loads[b2])
        cur_S += (S1 - S0)

    # ---------------- Stronger bin-emptying (short ejection chains) ----------------
    def try_place_with_ejection(item: int, b_src: int, max_depth: int = 3) -> List[Tuple[int, int, int]]:
        # Returns a sequence of moves [(it, from, to), ...] that results in moving `item`
        # out of b_src feasibly, using up to max_depth-1 ejections.
        w = weights[item]

        # Direct placement
        b_to = pick_target_bin(b_src, w, tries=40)
        if b_to != -1:
            return [(item, b_src, b_to)]

        # Depth-2/3: choose a destination bin b where we can eject a victim v to make room.
        # We prefer bins where needed slack is small.
        candidates = []
        for _ in range(40):
            b = random.randrange(nbins)
            if b == b_src or loads[b] <= 0:
                continue
            need = loads[b] + w - C
            if need <= 0:
                continue
            candidates.append((need, b))
        if not candidates:
            return []
        candidates.sort(key=lambda t: t[0])
        candidates = candidates[:12]

        def find_victim(bin_idx: int, need: int) -> int:
            # pick smallest item with weight >= need (best to minimize disruption)
            best = -1
            best_w = None
            for _ in range(min(10, len(bins[bin_idx]))):
                v = random.choice(bins[bin_idx])
                vw = weights[v]
                if vw >= need:
                    if best_w is None or vw < best_w:
                        best_w = vw
                        best = v
            # fallback scan if random missed
            if best == -1:
                for v in bins[bin_idx]:
                    vw = weights[v]
                    if vw >= need and (best_w is None or vw < best_w):
                        best_w = vw
                        best = v
            return best

        for need, b in candidates:
            victim = find_victim(b, need)
            if victim == -1:
                continue
            vw = weights[victim]

            # place victim elsewhere (not b_src, not b)
            vt = pick_target_bin(b, vw, tries=50)
            if vt != -1 and vt != b_src and vt != b:
                # then item can go to b
                if loads[b] - vw + w <= C:
                    return [(victim, b, vt), (item, b_src, b)]

            if max_depth >= 3:
                # try a second ejection for victim
                # pick b2 and victim2 to move out
                for _ in range(10):
                    b2 = random.randrange(nbins)
                    if b2 in (b_src, b):
                        continue
                    if loads[b2] <= 0:
                        continue
                    need2 = loads[b2] + vw - C
                    if need2 <= 0:
                        # victim fits directly
                        if loads[b2] + vw <= C and loads[b] - vw + w <= C:
                            return [(victim, b, b2), (item, b_src, b)]
                        continue
                    victim2 = find_victim(b2, need2)
                    if victim2 == -1:
                        continue
                    v2w = weights[victim2]
                    b3 = pick_target_bin(b2, v2w, tries=60)
                    if b3 != -1 and b3 not in (b_src, b, b2):
                        if loads[b2] - v2w + vw <= C and loads[b] - vw + w <= C:
                            return [(victim2, b2, b3), (victim, b, b2), (item, b_src, b)]

        return []

    def empty_bin_move() -> Tuple[bool, List[Tuple[int, int, int]]]:
        # Choose a source bin to empty: biased to low load (easy to empty) but not too tight.
        # We use a tournament on (load, size) and also consider slack.
        sample = min(nbins, 36)
        cand = []
        for _ in range(sample):
            b = random.randrange(nbins)
            if loads[b] <= 0 or not bins[b]:
                continue
            cand.append(b)
        if not cand:
            return False, []
        # favor small loads first, then fewer items
        cand.sort(key=lambda b: (loads[b], len(bins[b])))
        b_src = cand[0]

        items_src = bins[b_src][:]
        items_src.sort(key=lambda i: weights[i], reverse=True)

        moves: List[Tuple[int, int, int]] = []
        # Plan sequentially; after each planned move we will apply immediately in SA loop.
        # Here we only build a feasible plan w.r.t current state by assuming sequential application.
        # We do it on a simulated load/assignment snapshot for robustness.
        sim_loads = loads[:]
        sim_bins_items = [None] * nbins  # lazily copy lists if needed
        sim_assign = assign[:]  # only for moved items count (n can be large but acceptable once per proposal)

        def sim_bin_list(b: int) -> List[int]:
            lst = sim_bins_items[b]
            if lst is None:
                sim_bins_items[b] = bins[b][:]
            return sim_bins_items[b]

        for it_item in items_src:
            if sim_assign[it_item] != b_src:
                continue
            w = weights[it_item]
            # direct
            bt = -1
            best_res = None
            for _ in range(48):
                b = random.randrange(nbins)
                if b == b_src:
                    continue
                if sim_loads[b] + w <= C:
                    res = C - (sim_loads[b] + w)
                    key = res + (0 if sim_loads[b] > 0 else 2)
                    if best_res is None or key < best_res:
                        best_res = key
                        bt = b
            if bt != -1:
                moves.append((it_item, b_src, bt))
                sim_loads[b_src] -= w
                sim_loads[bt] += w
                sim_assign[it_item] = bt
                continue

            # ejection chain attempt depth 3 (using current real state is hard; use approximate via real then trust feasibility check on apply)
            chain = try_place_with_ejection(it_item, b_src, max_depth=3)
            if not chain:
                return False, []
            # Update sim for chain
            for x, bf, bt2 in chain:
                wx = weights[x]
                sim_loads[bf] -= wx
                sim_loads[bt2] += wx
                sim_assign[x] = bt2
            moves.extend(chain)

        return True, moves

    def merge_bins_move() -> Tuple[bool, List[Tuple[int, int, int]]]:
        # Attempt to empty a small bin into a larger bin (or several bins) using best-fit placements.
        cand = []
        for _ in range(min(40, nbins)):
            b = random.randrange(nbins)
            if loads[b] > 0:
                cand.append(b)
        if len(cand) < 2:
            return False, []
        cand = list(dict.fromkeys(cand))
        cand.sort(key=lambda b: loads[b])
        src = cand[0]
        dst = cand[-1]
        if src == dst or loads[src] <= 0 or loads[dst] <= 0:
            return False, []

        items = bins[src][:]
        items.sort(key=lambda i: weights[i], reverse=True)

        # direct full merge
        if loads[src] + loads[dst] <= C:
            return True, [(it, src, dst) for it in items]

        # otherwise, try to move as many as possible into best-fit targets (including dst)
        moves: List[Tuple[int, int, int]] = []
        for it_item in items:
            w = weights[it_item]
            bt = pick_target_bin(src, w, tries=36)
            if bt == -1:
                return False, []
            moves.append((it_item, src, bt))
        # only worth if likely empties src (it will if all moved)
        return True, moves

    # ---------------- SA temperature control ----------------
    def estimate_T0(samples: int = 220) -> float:
        pos_dE = []
        if nbins <= 1:
            return 1.0
        for _ in range(samples):
            b_from = random.randrange(nbins)
            if loads[b_from] <= 0 or not bins[b_from] or len(bins[b_from]) <= 1:
                continue
            item = random.choice(bins[b_from])
            w = weights[item]
            b_to = pick_target_bin(b_from, w, tries=24)
            if b_to == -1:
                continue
            # approximate dE via local contributions
            Lf0, Lt0 = loads[b_from], loads[b_to]
            Lf1, Lt1 = Lf0 - w, Lt0 + w
            dB = 0
            if Lf0 > 0 and Lf1 == 0:
                dB -= 1
            if Lt0 == 0 and Lt1 > 0:
                dB += 1
            dS = (contrib(Lf1) + contrib(Lt1)) - (contrib(Lf0) + contrib(Lt0))
            dE = dB * K + dS
            if dE > 0:
                pos_dE.append(dE)
        if not pos_dE:
            return 1.0
        avg = sum(pos_dE) / len(pos_dE)
        return max(1e-9, avg / max(1e-12, -math.log(0.75)))

    T0 = estimate_T0()
    T = T0
    alpha = 0.99985

    # ---------------- Move-type probabilities ----------------
    move_types = ["single", "swap", "empty", "merge"]
    probs = {"single": 0.52, "swap": 0.16, "empty": 0.26, "merge": 0.06}

    def renormalize() -> None:
        s = sum(probs.values())
        if s <= 0:
            for m in move_types:
                probs[m] = 1.0 / len(move_types)
        else:
            inv = 1.0 / s
            for m in move_types:
                probs[m] *= inv

    def pick_move_type() -> str:
        r = random.random()
        acc = 0.0
        for m in move_types:
            acc += probs[m]
            if r <= acc:
                return m
        return move_types[-1]

    # Adapt temperature to keep acceptance from collapsing
    window = 25000
    proposed = 0
    accepted_count = 0
    iter_since_bestB = 0

    max_iters = 120_000_000
    check_every = 6000

    for it in range(max_iters):
        if (it % check_every) == 0 and now() >= deadline:
            break

        mtype = pick_move_type()
        proposed += 1
        accepted = False

        if mtype == "single":
            # bias: pick from non-empty bins; sometimes pick from a small-load bin (hard bins)
            if random.random() < 0.35:
                # tournament for small load non-empty
                b_from = -1
                bestL = None
                for _ in range(12):
                    b = random.randrange(nbins)
                    if loads[b] <= 0 or not bins[b]:
                        continue
                    if bestL is None or loads[b] < bestL:
                        bestL = loads[b]
                        b_from = b
                if b_from == -1:
                    b_from = random.randrange(nbins)
            else:
                b_from = random.randrange(nbins)

            if loads[b_from] <= 0 or not bins[b_from]:
                T *= alpha
                iter_since_bestB += 1
                continue

            # pick an item: heavier more often
            if len(bins[b_from]) == 1 or random.random() < 0.7:
                item = max(bins[b_from], key=lambda x: weights[x]) if random.random() < 0.55 else random.choice(bins[b_from])
            else:
                item = random.choice(bins[b_from])
            w = weights[item]

            b_to = pick_target_bin(b_from, w, tries=32)
            if b_to == -1:
                T *= alpha
                iter_since_bestB += 1
                continue

            B0, S0 = cur_B, cur_S
            apply_move(item, b_from, b_to)
            dE = (cur_B - B0) * K + (cur_S - S0)
            if dE <= 0 or random.random() < math.exp(-dE / max(1e-12, T)):
                accepted = True
            else:
                undo_move(item, b_from, b_to)

        elif mtype == "swap":
            b1 = random.randrange(nbins)
            b2 = random.randrange(nbins)
            if b1 == b2 or loads[b1] <= 0 or loads[b2] <= 0 or not bins[b1] or not bins[b2]:
                T *= alpha
                iter_since_bestB += 1
                continue

            # choose heavier items with some probability
            i1 = max(bins[b1], key=lambda x: weights[x]) if random.random() < 0.35 else random.choice(bins[b1])
            i2 = max(bins[b2], key=lambda x: weights[x]) if random.random() < 0.35 else random.choice(bins[b2])
            w1, w2 = weights[i1], weights[i2]
            if loads[b1] - w1 + w2 > C or loads[b2] - w2 + w1 > C:
                T *= alpha
                iter_since_bestB += 1
                continue

            B0, S0 = cur_B, cur_S
            apply_swap(i1, b1, i2, b2)
            dE = (cur_B - B0) * K + (cur_S - S0)
            if dE <= 0 or random.random() < math.exp(-dE / max(1e-12, T)):
                accepted = True
            else:
                undo_swap(i1, b1, i2, b2)

        elif mtype == "empty":
            ok, moves = empty_bin_move()
            if not ok or not moves:
                T *= alpha
                iter_since_bestB += 1
                continue

            B0, S0 = cur_B, cur_S
            applied = 0
            feasible = True
            for item, bf, bt in moves:
                if assign[item] != bf or loads[bt] + weights[item] > C:
                    feasible = False
                    break
                apply_move(item, bf, bt)
                applied += 1
            if not feasible:
                for k in range(applied - 1, -1, -1):
                    item, bf, bt = moves[k]
                    undo_move(item, bf, bt)
                T *= alpha
                iter_since_bestB += 1
                continue

            dE = (cur_B - B0) * K + (cur_S - S0)
            if dE <= 0 or random.random() < math.exp(-dE / max(1e-12, T)):
                accepted = True
            else:
                for k in range(len(moves) - 1, -1, -1):
                    item, bf, bt = moves[k]
                    undo_move(item, bf, bt)

        else:  # merge
            ok, moves = merge_bins_move()
            if not ok or not moves:
                T *= alpha
                iter_since_bestB += 1
                continue
            B0, S0 = cur_B, cur_S
            applied = 0
            feasible = True
            for item, bf, bt in moves:
                if assign[item] != bf or loads[bt] + weights[item] > C:
                    feasible = False
                    break
                apply_move(item, bf, bt)
                applied += 1
            if not feasible:
                for k in range(applied - 1, -1, -1):
                    item, bf, bt = moves[k]
                    undo_move(item, bf, bt)
                T *= alpha
                iter_since_bestB += 1
                continue

            dE = (cur_B - B0) * K + (cur_S - S0)
            if dE <= 0 or random.random() < math.exp(-dE / max(1e-12, T)):
                accepted = True
            else:
                for k in range(len(moves) - 1, -1, -1):
                    item, bf, bt = moves[k]
                    undo_move(item, bf, bt)

        if accepted:
            accepted_count += 1
            # Update best
            if (cur_B, cur_S) < (best_B, best_S):
                best_B, best_S = cur_B, cur_S
                best_assign = assign[:]
                iter_since_bestB = 0
            else:
                iter_since_bestB += 1
        else:
            iter_since_bestB += 1

        # Cooling
        T *= alpha

        # Reheating / probability tweaks if stalled
        if iter_since_bestB > 400000:
            T = max(T, 0.6 * T0)
            probs["empty"] = min(0.45, probs["empty"] + 0.10)
            probs["merge"] = min(0.15, probs["merge"] + 0.03)
            probs["single"] = max(0.30, probs["single"] - 0.10)
            renormalize()
            iter_since_bestB = 0

        # Temperature feedback control every window to maintain exploration
        if (it + 1) % window == 0:
            acc_rate = accepted_count / max(1, proposed)
            # target acceptance: ~0.18..0.30 late, higher earlier
            target = 0.28 if best_B > LB else 0.20
            if acc_rate < target * 0.75:
                T *= 1.35
            elif acc_rate > target * 1.35:
                T *= 0.80
            # also push empty moves more if B not at LB
            if best_B > LB:
                probs["empty"] = min(0.42, probs["empty"] + 0.03)
                probs["single"] = max(0.32, probs["single"] - 0.03)
                renormalize()
            proposed = 0
            accepted_count = 0

        if best_B == LB and (it % (3 * window) == 0):
            # When at LB, focus on tightening (S) with single/swap
            probs["single"] = 0.70
            probs["swap"] = 0.26
            probs["empty"] = 0.03
            probs["merge"] = 0.01
            renormalize()

    # ---------------- Rebuild packing from best_assign ----------------
    bin_map: Dict[int, int] = {}
    packing: List[List[int]] = []
    bin_weights: List[int] = []

    for i, b in enumerate(best_assign):
        if b < 0:
            continue
        if b not in bin_map:
            bin_map[b] = len(packing)
            packing.append([])
            bin_weights.append(0)
        nb = bin_map[b]
        packing[nb].append(i)
        bin_weights[nb] += weights[i]

    for k in range(len(packing)):
        packing[k].sort()

    return {"packing": packing, "bin_weights": bin_weights}
