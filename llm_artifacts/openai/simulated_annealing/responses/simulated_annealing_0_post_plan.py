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
    # Secondary: S = sum(slack^2) over non-empty bins, slack = C-load
    # SA energy: E = B*K + S, with K dominating typical S deltas.
    K = max(1, (C * C) * 8)

    def phi_slack(sl: int) -> int:
        return sl * sl

    # ---------------- Utilities ----------------
    order_desc = list(range(n))
    order_desc.sort(key=lambda i: weights[i], reverse=True)

    total_w = sum(weights)
    LB = (total_w + C - 1) // C

    # Build bins/loads/assign/pos from an insertion order using a heuristic variant.
    # nbins grows as needed here; later SA will not create new bins beyond initial nbins.
    def build_initial(variant: int) -> Tuple[List[List[int]], List[int], List[int], List[int]]:
        bins: List[List[int]] = []
        loads: List[int] = []
        assign = [-1] * n
        pos = [-1] * n

        # Make a possibly perturbed item order (diversity among equal weights)
        if variant == 0:
            items = order_desc
        else:
            items = order_desc[:]
            # shuffle inside equal-weight blocks
            i = 0
            while i < n:
                j = i + 1
                wi = weights[items[i]]
                while j < n and weights[items[j]] == wi:
                    j += 1
                if j - i > 1:
                    random.shuffle(items[i:j])
                i = j

        # Choose placement rule
        # 0/1/2: Best-Fit Decreasing with random tie breaking / residual noise
        # 3/4: First-Fit Decreasing (random bin scan)
        for it in items:
            w = weights[it]
            best_b = -1
            best_key = None

            m = len(bins)
            if m:
                # random scan order without allocations
                # try a limited number of candidates, then fall back to full scan
                tries = 12 if m > 12 else m
                seen = set()
                for _ in range(tries):
                    b = random.randrange(m)
                    if b in seen:
                        continue
                    seen.add(b)
                    if loads[b] + w <= C:
                        res = C - (loads[b] + w)
                        if variant in (2,):
                            # small noise to diversify while still best-fit-ish
                            key = res + random.random() * 0.25
                        elif variant in (3, 4):
                            # first-fit: accept first feasible under a randomized scan
                            best_b = b
                            best_key = 0
                            break
                        else:
                            # best-fit
                            key = res
                        if best_key is None or key < best_key:
                            best_key = key
                            best_b = b

                if best_b == -1 and variant in (0, 1, 2):
                    # full best-fit scan
                    # variant 1 random tie-break via shuffled bin indices in chunks
                    if variant == 1:
                        idxs = list(range(m))
                        random.shuffle(idxs)
                        for b in idxs:
                            if loads[b] + w <= C:
                                res = C - (loads[b] + w)
                                if best_key is None or res < best_key:
                                    best_key = res
                                    best_b = b
                    else:
                        for b in range(m):
                            if loads[b] + w <= C:
                                res = C - (loads[b] + w)
                                if best_key is None or res < best_key:
                                    best_key = res
                                    best_b = b

                if best_b == -1 and variant in (3, 4):
                    # full first-fit scan (variant 4 scans from random start)
                    if variant == 4 and m > 0:
                        s = random.randrange(m)
                        for t in range(m):
                            b = (s + t) % m
                            if loads[b] + w <= C:
                                best_b = b
                                break
                    else:
                        for b in range(m):
                            if loads[b] + w <= C:
                                best_b = b
                                break

            if best_b == -1:
                best_b = len(bins)
                bins.append([])
                loads.append(0)

            assign[it] = best_b
            pos[it] = len(bins[best_b])
            bins[best_b].append(it)
            loads[best_b] += w

        return bins, loads, assign, pos

    def compute_BS(loads: List[int]) -> Tuple[int, int]:
        B = 0
        S = 0
        for L in loads:
            if L > 0:
                B += 1
                sl = C - L
                S += phi_slack(sl)
        return B, S

    # Multiple initial solutions
    init_trials = 12
    best_bins = None
    best_loads = None
    best_assign = None
    best_pos = None
    best_B = None
    best_S = None

    for v in range(init_trials):
        bins, loads, assign, pos = build_initial(v % 5)
        B, S = compute_BS(loads)
        if best_B is None or (B, S) < (best_B, best_S):
            best_bins, best_loads, best_assign, best_pos = bins, loads, assign, pos
            best_B, best_S = B, S

        # quick exit if already at lower bound
        if best_B == LB:
            break

        if now() >= deadline:
            break

    bins = best_bins
    loads = best_loads
    assign = best_assign
    pos = best_pos
    cur_B, cur_S = best_B, best_S

    best_assign = assign[:]  # store only assignment for best
    best_B, best_S = cur_B, cur_S

    nbins = len(bins)
    if nbins == 0:
        return {"packing": [], "bin_weights": []}

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

    # ---------------- Slack-target selection ----------------
    # We avoid heavy data structures; pick several candidates and prefer slack close to w.
    def pick_target_bin(b_from: int, w: int, tries: int = 18) -> int:
        best_b = -1
        best_key = None
        for _ in range(tries):
            b = random.randrange(nbins)
            if b == b_from:
                continue
            if loads[b] + w <= C:
                sl = C - loads[b]
                # want remaining slack after insertion close to 0 (tight),
                # and also target bins whose current slack is close to w
                res = sl - w
                key = (res * res) + (abs(sl - w))
                # small randomness
                key = key + (random.random() * 0.5)
                if best_key is None or key < best_key:
                    best_key = key
                    best_b = b
        if best_b != -1:
            return best_b

        # fallback: single random feasible scan attempts
        for _ in range(12):
            b = random.randrange(nbins)
            if b != b_from and loads[b] + w <= C:
                return b
        return -1

    # ---------------- Incremental objective updates ----------------
    def contrib(L: int) -> int:
        if L <= 0:
            return 0
        sl = C - L
        return phi_slack(sl)

    # Apply move item: b_from -> b_to, assumes feasible. Returns (dB, dS).
    def apply_move(item: int, b_from: int, b_to: int) -> Tuple[int, int]:
        nonlocal cur_B, cur_S
        w = weights[item]
        Lf0 = loads[b_from]
        Lt0 = loads[b_to]
        dB = 0

        # remove old contributions (only if bin non-empty)
        S0 = contrib(Lf0) + contrib(Lt0)

        remove_item(item, b_from)
        loads[b_from] = Lf0 - w
        add_item(item, b_to)
        loads[b_to] = Lt0 + w
        assign[item] = b_to

        Lf1 = loads[b_from]
        Lt1 = loads[b_to]

        if Lf0 > 0 and Lf1 == 0:
            dB -= 1
        if Lt0 == 0 and Lt1 > 0:
            dB += 1

        S1 = contrib(Lf1) + contrib(Lt1)
        dS = S1 - S0

        cur_B += dB
        cur_S += dS
        return dB, dS

    def undo_move(item: int, b_from: int, b_to: int) -> None:
        # item currently in b_to; move it back to b_from
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

        Lf1 = loads[b_from]
        Lt1 = loads[b_to]

        # update B
        if Lf0 == 0 and Lf1 > 0:
            cur_B += 1
        if Lt0 > 0 and Lt1 == 0:
            cur_B -= 1

        S1 = contrib(Lf1) + contrib(Lt1)
        cur_S += (S1 - S0)

    # Swap items across bins, assumes feasible. Returns (dS) (B unchanged unless a bin was empty which we avoid).
    def apply_swap(i1: int, b1: int, i2: int, b2: int) -> int:
        nonlocal cur_S
        w1 = weights[i1]
        w2 = weights[i2]
        L10 = loads[b1]
        L20 = loads[b2]
        S0 = contrib(L10) + contrib(L20)

        # remove i1 from b1, i2 from b2
        remove_item(i1, b1)
        loads[b1] = L10 - w1
        remove_item(i2, b2)
        loads[b2] = L20 - w2

        # add swapped
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
        # swap back
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

    # ---------------- Compound SA neighborhoods ----------------
    # Empty a bin via destroy/repair with at most one ejection.
    def empty_bin_move() -> Tuple[bool, List[Tuple[int, int, int]]]:
        # returns (proposed, move_list) where move_list is [(item, from, to), ...] applied in order
        # If proposed False: no feasible neighbor generated.
        # Choose src among worst by slack (largest slack) in a random sample.
        # Avoid empty bins.
        sample = 24 if nbins > 24 else nbins
        best_b = -1
        best_sl = -1
        for _ in range(sample):
            b = random.randrange(nbins)
            if loads[b] <= 0 or not bins[b]:
                continue
            sl = C - loads[b]
            if sl > best_sl:
                best_sl = sl
                best_b = b
        if best_b == -1:
            return False, []
        b_src = best_b

        items_src = bins[b_src][:]
        # heavy-first to improve feasibility
        items_src.sort(key=lambda i: weights[i], reverse=True)

        moves: List[Tuple[int, int, int]] = []

        for item in items_src:
            w = weights[item]
            b_to = pick_target_bin(b_src, w, tries=26)
            if b_to != -1:
                moves.append((item, b_src, b_to))
                continue

            # One ejection attempt: pick a destination bin, eject one item from it.
            # try a handful of candidate bins
            done = False
            for _ in range(10):
                b = random.randrange(nbins)
                if b == b_src or loads[b] <= 0:
                    continue
                if loads[b] + w <= C:
                    continue  # would have been feasible

                need = loads[b] + w - C
                # find an item in b with weight >= need (random trials)
                if not bins[b]:
                    continue
                for _t in range(6):
                    victim = random.choice(bins[b])
                    if weights[victim] >= need:
                        # find place for victim (not b_src, not b)
                        vt = pick_target_bin(b, weights[victim], tries=26)
                        if vt == -1 or vt == b_src:
                            continue
                        # perform: victim b->vt, then item b_src->b
                        moves.append((victim, b, vt))
                        moves.append((item, b_src, b))
                        done = True
                        break
                if done:
                    break
            if not done:
                return False, []

        # success if this would empty src
        return True, moves

    def merge_bins_move() -> Tuple[bool, List[Tuple[int, int, int]]]:
        # Attempt to empty one of two low-load bins.
        # Pick two non-empty bins biased to small loads.
        candidates = []
        for _ in range(18):
            b = random.randrange(nbins)
            if loads[b] > 0:
                candidates.append(b)
        if len(candidates) < 2:
            return False, []
        candidates.sort(key=lambda b: loads[b])
        a = candidates[0]
        b = candidates[1]
        if a == b or loads[a] <= 0 or loads[b] <= 0:
            return False, []

        # direct merge?
        if loads[a] + loads[b] <= C:
            # move all items of a into b
            items_a = bins[a][:]
            items_a.sort(key=lambda i: weights[i], reverse=True)
            return True, [(it, a, b) for it in items_a]

        # partial: try to empty smaller bin into the other using repair
        src, dst = (a, b) if loads[a] <= loads[b] else (b, a)
        items_src = bins[src][:]
        items_src.sort(key=lambda i: weights[i], reverse=True)

        moves: List[Tuple[int, int, int]] = []
        # Greedy subset to fit into dst
        cap = C - loads[dst]
        for it in items_src:
            w = weights[it]
            if w <= cap:
                moves.append((it, src, dst))
                cap -= w
        if not moves:
            return False, []

        # if subset empties src: great, else still a valid neighbor but not bin-reducing.
        # We will propose only if empties or with some probability.
        if len(moves) == len(items_src) or random.random() < 0.25:
            return True, moves
        return False, []

    def compact_destroy_repair() -> Tuple[bool, List[Tuple[int, int, int]]]:
        # Pick a few slacky bins, remove all their items, then reinsert by best-fit.
        k = 2 + (1 if nbins >= 40 and random.random() < 0.6 else 0)
        k = min(k, 4)

        # choose bins with large slack in a sample
        sample = 28 if nbins > 28 else nbins
        pool = []
        for _ in range(sample):
            b = random.randrange(nbins)
            if loads[b] > 0:
                pool.append(b)
        if not pool:
            return False, []
        # keep unique
        pool = list(dict.fromkeys(pool))
        pool.sort(key=lambda b: (C - loads[b]), reverse=True)
        chosen = pool[: min(k, len(pool))]
        if not chosen:
            return False, []

        removed_items: List[Tuple[int, int]] = []
        for b in chosen:
            for it in bins[b]:
                removed_items.append((it, b))
        if not removed_items:
            return False, []

        removed_items.sort(key=lambda t: weights[t[0]], reverse=True)

        # Plan reinsert moves in a feasibility-first way, using current loads as if items removed.
        # We'll construct moves and then apply as one compound proposal.
        # First, simulate removal loads.
        sim_loads = loads[:]
        for it, b in removed_items:
            sim_loads[b] -= weights[it]

        moves: List[Tuple[int, int, int]] = []
        for it, b_from in removed_items:
            w = weights[it]
            # best-fit among random bins excluding its original removed-from bin (allowed though)
            best_b = -1
            best_res = None
            for _ in range(24):
                b = random.randrange(nbins)
                if b == b_from:
                    continue
                if sim_loads[b] + w <= C:
                    res = C - (sim_loads[b] + w)
                    if best_res is None or res < best_res:
                        best_res = res
                        best_b = b
            if best_b == -1:
                # allow putting back
                if sim_loads[b_from] + w <= C:
                    best_b = b_from
                else:
                    return False, []
            sim_loads[best_b] += w
            moves.append((it, b_from, best_b))

        return True, moves

    # ---------------- SA temperature init (S-only deltas with B unchanged) ----------------
    def estimate_T0(samples: int = 120) -> float:
        if nbins <= 1:
            return 1.0
        pos = []
        for _ in range(samples):
            # propose a single_move that doesn't change B: move item from non-empty to non-empty,
            # and don't empty the source.
            b_from = random.randrange(nbins)
            if loads[b_from] <= 0 or len(bins[b_from]) <= 1:
                continue
            item = random.choice(bins[b_from])
            w = weights[item]
            b_to = pick_target_bin(b_from, w, tries=20)
            if b_to == -1 or loads[b_to] <= 0:
                continue
            # compute dS quickly without applying
            Lf0 = loads[b_from]
            Lt0 = loads[b_to]
            Lf1 = Lf0 - w
            Lt1 = Lt0 + w
            if Lf1 <= 0 or Lt1 <= 0:
                continue
            dS = (contrib(Lf1) + contrib(Lt1)) - (contrib(Lf0) + contrib(Lt0))
            if dS > 0:
                pos.append(dS)
        if not pos:
            return 1.0
        avg = sum(pos) / len(pos)
        # target acceptance ~0.7
        return max(1e-9, avg / max(1e-12, -math.log(0.7)))

    T0 = estimate_T0(160)
    T = T0

    alpha = 0.9997

    # ---------------- Move-type probabilities with adaptation ----------------
    move_types = ["single", "swap", "empty", "merge", "compact"]
    probs = {
        "single": 0.50,
        "swap": 0.20,
        "empty": 0.20,
        "merge": 0.07,
        "compact": 0.03,
    }

    stats = {m: [0, 0, 0] for m in move_types}  # proposed, accepted, B_improved
    window = 20000
    iter_since_B = 0

    def pick_move_type() -> str:
        r = random.random()
        acc = 0.0
        for m in move_types:
            acc += probs[m]
            if r <= acc:
                return m
        return move_types[-1]

    def renormalize() -> None:
        s = sum(probs.values())
        if s <= 0:
            for m in move_types:
                probs[m] = 1.0 / len(move_types)
        else:
            inv = 1.0 / s
            for m in move_types:
                probs[m] *= inv

    # ---------------- SA loop ----------------
    max_iters = 60_000_000
    check_every = 4000

    for it in range(max_iters):
        if (it % check_every) == 0 and now() >= deadline:
            break

        if best_B == LB:
            # already optimal in bin count; keep running but bias to maintain and improve S.
            probs["single"] = 0.70
            probs["swap"] = 0.25
            probs["empty"] = 0.03
            probs["merge"] = 0.02
            probs["compact"] = 0.00
            renormalize()

        # Multi-phase: if stalled in B, reheat and push empty/merge a bit.
        if iter_since_B > 250000:
            T = max(T, 0.5 * T0)
            probs["empty"] = min(0.40, probs["empty"] + 0.08)
            probs["merge"] = min(0.18, probs["merge"] + 0.03)
            probs["single"] = max(0.25, probs["single"] - 0.08)
            renormalize()
            iter_since_B = 0

        mtype = pick_move_type()
        stats[mtype][0] += 1

        accepted = False
        B_improved = False

        if mtype == "single":
            b_from = random.randrange(nbins)
            if loads[b_from] <= 0 or not bins[b_from]:
                T *= alpha
                iter_since_B += 1
                continue
            item = random.choice(bins[b_from])
            w = weights[item]
            b_to = pick_target_bin(b_from, w, tries=18)
            if b_to == -1:
                T *= alpha
                iter_since_B += 1
                continue

            # compute energy delta exactly by applying and potentially undoing
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
                iter_since_B += 1
                continue
            i1 = random.choice(bins[b1])
            i2 = random.choice(bins[b2])
            w1 = weights[i1]
            w2 = weights[i2]
            if loads[b1] - w1 + w2 > C or loads[b2] - w2 + w1 > C:
                T *= alpha
                iter_since_B += 1
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
            if not ok:
                T *= alpha
                iter_since_B += 1
                continue

            B0, S0 = cur_B, cur_S
            # apply all moves
            applied = 0
            feasible = True
            for item, bf, bt in moves:
                if loads[bt] + weights[item] > C:
                    feasible = False
                    break
                if assign[item] != bf:
                    feasible = False
                    break
                apply_move(item, bf, bt)
                applied += 1

            if not feasible:
                # undo applied
                for k in range(applied - 1, -1, -1):
                    item, bf, bt = moves[k]
                    undo_move(item, bf, bt)
                T *= alpha
                iter_since_B += 1
                continue

            dE = (cur_B - B0) * K + (cur_S - S0)
            if dE <= 0 or random.random() < math.exp(-dE / max(1e-12, T)):
                accepted = True
            else:
                for k in range(len(moves) - 1, -1, -1):
                    item, bf, bt = moves[k]
                    undo_move(item, bf, bt)

        elif mtype == "merge":
            ok, moves = merge_bins_move()
            if not ok:
                T *= alpha
                iter_since_B += 1
                continue
            B0, S0 = cur_B, cur_S
            applied = 0
            feasible = True
            for item, bf, bt in moves:
                if loads[bt] + weights[item] > C or assign[item] != bf:
                    feasible = False
                    break
                apply_move(item, bf, bt)
                applied += 1
            if not feasible:
                for k in range(applied - 1, -1, -1):
                    item, bf, bt = moves[k]
                    undo_move(item, bf, bt)
                T *= alpha
                iter_since_B += 1
                continue
            dE = (cur_B - B0) * K + (cur_S - S0)
            if dE <= 0 or random.random() < math.exp(-dE / max(1e-12, T)):
                accepted = True
            else:
                for k in range(len(moves) - 1, -1, -1):
                    item, bf, bt = moves[k]
                    undo_move(item, bf, bt)

        else:  # compact
            ok, moves = compact_destroy_repair()
            if not ok:
                T *= alpha
                iter_since_B += 1
                continue

            # Apply as a compound move:
            # first remove all selected items to make state match the planner assumption.
            # We can do this by moving each item to a temporary empty bin? But SA forbids new bins.
            # Instead: perform explicit removals, then insert. We'll implement with direct removal/insertion
            # while tracking objective deltas exactly.
            #
            # To do that, we remove each unique item from its current bin (should match bf), making bins possibly empty,
            # then reinsert to bt.
            B0, S0 = cur_B, cur_S

            # Gather unique items in this compound plan
            items_plan = [it for it, _, _ in moves]
            if len(set(items_plan)) != len(items_plan):
                T *= alpha
                iter_since_B += 1
                continue

            # Phase 1: remove all
            removed: List[Tuple[int, int]] = []
            feasible = True
            for it_item, bf, bt in moves:
                if assign[it_item] != bf:
                    feasible = False
                    break
                # remove it_item from bf without placing
                w = weights[it_item]
                L0 = loads[bf]
                S_before = contrib(L0)
                remove_item(it_item, bf)
                loads[bf] = L0 - w
                assign[it_item] = -1
                # update B and S
                if L0 > 0 and loads[bf] == 0:
                    cur_B -= 1
                cur_S += (contrib(loads[bf]) - S_before)
                removed.append((it_item, bf))

            if not feasible:
                # rollback removals
                for it_item, bf in reversed(removed):
                    w = weights[it_item]
                    L0 = loads[bf]
                    S_before = contrib(L0)
                    add_item(it_item, bf)
                    loads[bf] = L0 + w
                    assign[it_item] = bf
                    if L0 == 0 and loads[bf] > 0:
                        cur_B += 1
                    cur_S += (contrib(loads[bf]) - S_before)
                T *= alpha
                iter_since_B += 1
                continue

            # Phase 2: insert all
            inserted: List[Tuple[int, int]] = []
            for it_item, bf, bt in moves:
                w = weights[it_item]
                if loads[bt] + w > C:
                    feasible = False
                    break
                L0 = loads[bt]
                S_before = contrib(L0)
                add_item(it_item, bt)
                loads[bt] = L0 + w
                assign[it_item] = bt
                if L0 == 0 and loads[bt] > 0:
                    cur_B += 1
                cur_S += (contrib(loads[bt]) - S_before)
                inserted.append((it_item, bt))

            if not feasible:
                # rollback insertions
                for it_item, bt in reversed(inserted):
                    w = weights[it_item]
                    L0 = loads[bt]
                    S_before = contrib(L0)
                    remove_item(it_item, bt)
                    loads[bt] = L0 - w
                    assign[it_item] = -1
                    if L0 > 0 and loads[bt] == 0:
                        cur_B -= 1
                    cur_S += (contrib(loads[bt]) - S_before)
                # rollback removals (put back)
                for it_item, bf in reversed(removed):
                    w = weights[it_item]
                    L0 = loads[bf]
                    S_before = contrib(L0)
                    add_item(it_item, bf)
                    loads[bf] = L0 + w
                    assign[it_item] = bf
                    if L0 == 0 and loads[bf] > 0:
                        cur_B += 1
                    cur_S += (contrib(loads[bf]) - S_before)

                T *= alpha
                iter_since_B += 1
                continue

            dE = (cur_B - B0) * K + (cur_S - S0)
            if dE <= 0 or random.random() < math.exp(-dE / max(1e-12, T)):
                accepted = True
            else:
                # undo by reversing the compound operation: remove inserted, add removed back
                for it_item, bt in reversed(inserted):
                    w = weights[it_item]
                    L0 = loads[bt]
                    S_before = contrib(L0)
                    remove_item(it_item, bt)
                    loads[bt] = L0 - w
                    assign[it_item] = -1
                    if L0 > 0 and loads[bt] == 0:
                        cur_B -= 1
                    cur_S += (contrib(loads[bt]) - S_before)
                for it_item, bf in reversed(removed):
                    w = weights[it_item]
                    L0 = loads[bf]
                    S_before = contrib(L0)
                    add_item(it_item, bf)
                    loads[bf] = L0 + w
                    assign[it_item] = bf
                    if L0 == 0 and loads[bf] > 0:
                        cur_B += 1
                    cur_S += (contrib(loads[bf]) - S_before)

        if accepted:
            stats[mtype][1] += 1
            # Update best lexicographically
            if (cur_B, cur_S) < (best_B, best_S):
                best_B, best_S = cur_B, cur_S
                best_assign = assign[:]
                iter_since_B = 0
                if cur_B < best_B:
                    B_improved = True
            else:
                iter_since_B += 1

            if cur_B < best_B:
                stats[mtype][2] += 1
                B_improved = True
        else:
            iter_since_B += 1

        # Cooling
        T *= alpha

        # Adapt move probabilities periodically
        if (it + 1) % window == 0:
            # compute simple utility: accepted rate + extra for B improvements
            util = {}
            for m in move_types:
                prop, acc, bimp = stats[m]
                if prop == 0:
                    u = 0.0
                else:
                    u = (acc / prop) + 2.5 * (bimp / prop)
                util[m] = u
            # convert to probabilities with a floor, keeping compact rare
            base = {"single": 0.18, "swap": 0.08, "empty": 0.10, "merge": 0.05, "compact": 0.01}
            for m in move_types:
                probs[m] = base[m] + util[m]
            # if B-stalled, boost empty/merge a bit
            if iter_since_B > 0.6 * window:
                probs["empty"] += 0.7
                probs["merge"] += 0.25
            renormalize()
            # reset stats
            for m in move_types:
                stats[m] = [0, 0, 0]

    # ---------------- Rebuild packing from best_assign ----------------
    # Compact non-empty bins in first-seen order.
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

    # sort indices within each bin for stable output
    for k in range(len(packing)):
        packing[k].sort()

    return {"packing": packing, "bin_weights": bin_weights}
