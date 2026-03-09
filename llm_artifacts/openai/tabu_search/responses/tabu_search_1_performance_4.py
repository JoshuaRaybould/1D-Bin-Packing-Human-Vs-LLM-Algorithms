import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    C = bin_capacity
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ---------------- Time helper ----------------
    def time_exceeded() -> bool:
        return (time.time() - start) >= time_limit

    # ---------------- Initial solutions ----------------
    def ffd_initial() -> Tuple[List[List[int]], List[int], List[int]]:
        order = sorted(range(n), key=lambda i: (-weights[i], i))
        bins: List[List[int]] = []
        loads: List[int] = []
        assign = [-1] * n
        for i in order:
            w = weights[i]
            best_b = -1
            best_rem = None
            for b, ld in enumerate(loads):
                if ld + w <= C:
                    rem = C - (ld + w)
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best_b = b
            if best_b == -1:
                best_b = len(bins)
                bins.append([])
                loads.append(0)
            bins[best_b].append(i)
            loads[best_b] += w
            assign[i] = best_b
        return bins, loads, assign

    def randomized_ffd(repeats: int) -> Tuple[List[List[int]], List[int], List[int]]:
        # Build several randomized greedy solutions quickly and keep the best.
        best = None
        best_obj = None

        idx = list(range(n))
        for _ in range(repeats):
            # Sort primarily decreasing but perturb by small noise to diversify
            idx.sort(key=lambda i: (-weights[i], random.random()))
            bins: List[List[int]] = []
            loads: List[int] = []
            assign = [-1] * n
            for i in idx:
                w = weights[i]
                best_b = -1
                best_rem = None
                # best-fit among existing bins
                for b, ld in enumerate(loads):
                    if ld + w <= C:
                        rem = C - (ld + w)
                        if best_rem is None or rem < best_rem:
                            best_rem = rem
                            best_b = b
                if best_b == -1:
                    best_b = len(bins)
                    bins.append([])
                    loads.append(0)
                bins[best_b].append(i)
                loads[best_b] += w
                assign[i] = best_b

            # quick objective
            m = len(bins)
            slack = sum(C - ld for ld in loads)
            sq = sum((C - ld) * (C - ld) for ld in loads)
            obj = (m, slack, sq)
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best = (bins, loads, assign)

            if time_exceeded():
                break
        return best  # type: ignore

    # ---------------- Packing utilities ----------------
    def cleanup_empty(bins: List[List[int]], loads: List[int], assign: List[int]) -> None:
        mapping = {}
        new_bins = []
        new_loads = []
        for b, items in enumerate(bins):
            if items:
                mapping[b] = len(new_bins)
                new_bins.append(items)
                new_loads.append(loads[b])
        for i in range(n):
            if assign[i] != -1:
                assign[i] = mapping[assign[i]]
        bins[:] = new_bins
        loads[:] = new_loads

    def apply_reloc(i: int, b_from: int, b_to: int,
                    bins: List[List[int]], loads: List[int], assign: List[int]) -> None:
        bins[b_from].remove(i)
        loads[b_from] -= weights[i]
        bins[b_to].append(i)
        loads[b_to] += weights[i]
        assign[i] = b_to

    def apply_swap(i: int, j: int, bi: int, bj: int,
                   bins: List[List[int]], loads: List[int], assign: List[int]) -> None:
        bins[bi].remove(i)
        bins[bj].remove(j)
        bins[bi].append(j)
        bins[bj].append(i)
        loads[bi] += weights[j] - weights[i]
        loads[bj] += weights[i] - weights[j]
        assign[i], assign[j] = bj, bi

    def apply_2_1(a1: int, a2: int, b1: int, A: int, B: int,
                 bins: List[List[int]], loads: List[int], assign: List[int]) -> None:
        # Move two items from A to B and move one item from B to A.
        # Preconditions: all are distinct, feasibility checked.
        bins[A].remove(a1)
        bins[A].remove(a2)
        bins[B].remove(b1)

        bins[A].append(b1)
        bins[B].append(a1)
        bins[B].append(a2)

        loads[A] += weights[b1] - weights[a1] - weights[a2]
        loads[B] += weights[a1] + weights[a2] - weights[b1]

        assign[a1] = B
        assign[a2] = B
        assign[b1] = A

    # ---------------- Objective (lexicographic) ----------------
    # Main goal: minimize number of bins.
    # Tie-breakers: minimize total slack (encourage tight packing), then squared slack.
    def obj_tuple(bins: List[List[int]], loads: List[int]) -> Tuple[int, int, int]:
        m = len(bins)
        slack = 0
        sq = 0
        for ld in loads:
            s = C - ld
            slack += s
            sq += s * s
        return (m, slack, sq)

    # ---------------- Initialize ----------------
    # Use a small multi-start to start from a stronger point; still cheap.
    bins, loads, assign = ffd_initial()
    cleanup_empty(bins, loads, assign)

    # If time allows, try a few randomized constructions and keep the best.
    if time_limit > 0.05:
        reps = 8
        if time_limit > 1.0:
            reps = 25
        if time_limit > 5.0:
            reps = 60
        cand = randomized_ffd(reps)
        if cand is not None:
            b2, l2, a2 = cand
            cleanup_empty(b2, l2, a2)
            if obj_tuple(b2, l2) < obj_tuple(bins, loads):
                bins, loads, assign = b2, l2, a2

    best_bins = [lst[:] for lst in bins]
    best_loads = loads[:]
    best_assign = assign[:]
    best_obj = obj_tuple(best_bins, best_loads)

    # ---------------- Tabu structures ----------------
    # Reloc tabu: (item, dest_bin) -> expiry
    # Swap tabu: (min(i,j), max(i,j)) -> expiry
    # 2-1 tabu: reuse swap tabu on (moved_item, other_item) pairs lightly + reloc on destinations.
    tabu_reloc: Dict[Tuple[int, int], int] = {}
    tabu_swap: Dict[Tuple[int, int], int] = {}

    # ---------------- Parameters ----------------
    # Fixed iteration budget; will also terminate on time.
    # Increased a lot vs the original to use the budget.
    max_iter = max(5000, 400 * n)

    # Candidate controls
    max_bins_sample = 40
    max_items_from_target = 60
    max_global_items = 80

    # Tabu tenure base
    base_tenure = max(7, int(0.9 * (n ** 0.5)))

    # Diversification schedule
    no_improve = 0
    last_best_it = 0

    # Periodic time check frequency
    TIME_CHECK_EVERY = 25

    it = 0
    while it < max_iter:
        it += 1
        if it % TIME_CHECK_EVERY == 0 and time_exceeded():
            break

        m = len(bins)
        if m <= 1:
            break

        # purge expired tabu
        if it % 200 == 0:
            tabu_reloc = {k: v for k, v in tabu_reloc.items() if v > it}
            tabu_swap = {k: v for k, v in tabu_swap.items() if v > it}

        curr_obj = obj_tuple(bins, loads)

        # -------- Choose a target bin to try to eliminate (intensification) --------
        # Mostly smallest load bin; occasionally a random light-ish bin for diversification.
        if random.random() < 0.85:
            target = min(range(m), key=lambda b: loads[b])
        else:
            by_load = sorted(range(m), key=lambda b: loads[b])
            target = by_load[random.randrange(min(10, m))]

        # Bin sampling: include target and other light bins + some random.
        if m <= max_bins_sample:
            bin_sample = list(range(m))
        else:
            by_load = sorted(range(m), key=lambda b: loads[b])
            chosen = set(by_load[: max_bins_sample // 2])
            chosen.add(target)
            while len(chosen) < max_bins_sample:
                chosen.add(random.randrange(m))
            bin_sample = list(chosen)

        # Items to consider: focus on target items + small global sample
        target_items = bins[target][:]
        if len(target_items) > max_items_from_target:
            target_items = random.sample(target_items, max_items_from_target)

        global_items = []
        for b in bin_sample:
            global_items.extend(bins[b])
        if len(global_items) > max_global_items:
            global_items = random.sample(global_items, max_global_items)

        # Ensure target items are present
        items_considered = list(dict.fromkeys(target_items + global_items))

        # Helper: best-fit destination list for an item weight
        def best_fit_dests(w: int, exclude_bin: int) -> List[int]:
            cands = []
            for b in bin_sample:
                if b == exclude_bin:
                    continue
                if loads[b] + w <= C:
                    rem = C - (loads[b] + w)
                    cands.append((rem, b))
            cands.sort()
            return [b for _, b in cands[:12]]

        best_move = None
        best_move_obj = None

        # Aspiration rule: allow tabu if improves best bins OR best objective
        def aspiration(cand_obj: Tuple[int, int, int]) -> bool:
            return cand_obj < best_obj

        # -------- Neighborhood 1: Relocate (target-driven) --------
        # Prioritize moving items out of target to empty it.
        for i_item in items_considered:
            b_from = assign[i_item]
            w = weights[i_item]

            # Candidate destinations: best-fit among sample + a couple random bins
            dests = best_fit_dests(w, b_from)
            if random.random() < 0.25:
                for _ in range(3):
                    b = random.randrange(m)
                    if b != b_from and loads[b] + w <= C:
                        dests.append(b)
            if not dests:
                continue

            # If item from target, explore more; else fewer
            if b_from != target and len(dests) > 6:
                dests = dests[:6]

            for b_to in dests:
                if b_to == b_from:
                    continue

                # Tabu check
                if tabu_reloc.get((i_item, b_to), 0) > it:
                    # aspiration later
                    pass

                new_load_from = loads[b_from] - w
                new_load_to = loads[b_to] + w

                # bins change if emptied
                new_m = m - 1 if new_load_from == 0 else m

                # compute new objective tuple fast-ish via deltas
                # total slack changes only for involved bins (but if bin removed, slack term for that bin disappears).
                # We'll compute cand slack/sq using current then delta; handle emptying as removing a bin with slack C.
                # Current slack and sq are in curr_obj.
                curr_m, curr_slack, curr_sq = curr_obj

                s_from_old = C - loads[b_from]
                s_to_old = C - loads[b_to]
                s_from_new = C - new_load_from
                s_to_new = C - new_load_to

                if new_load_from == 0:
                    # bin disappears: remove its slack term entirely (which would be C)
                    # but note s_from_new == C. The old term was s_from_old; removing means subtract old contribution.
                    new_slack = curr_slack - s_from_old + s_to_new - s_to_old
                    new_sq = curr_sq - (s_from_old * s_from_old) + (s_to_new * s_to_new) - (s_to_old * s_to_old)
                else:
                    new_slack = curr_slack + (s_from_new - s_from_old) + (s_to_new - s_to_old)
                    new_sq = curr_sq + (s_from_new * s_from_new - s_from_old * s_from_old) + (s_to_new * s_to_new - s_to_old * s_to_old)

                cand_obj = (new_m, new_slack, new_sq)

                is_tabu = tabu_reloc.get((i_item, b_to), 0) > it
                if is_tabu and not aspiration(cand_obj):
                    continue

                # strong preference: moves that empty target
                if best_move_obj is None or cand_obj < best_move_obj:
                    best_move_obj = cand_obj
                    best_move = ("reloc", i_item, b_from, b_to)

        # -------- Neighborhood 2: Swap --------
        # Useful when reloc is stuck.
        if best_move is None or (no_improve > 100 and random.random() < 0.7):
            trials = 250
            if n > 800:
                trials = 180
            for _ in range(trials):
                # bias: pick i from target often
                if bins[target] and random.random() < 0.7:
                    i = random.choice(bins[target])
                else:
                    i = random.choice(items_considered)
                bi = assign[i]

                bj = random.choice(bin_sample)
                if bj == bi or not bins[bj]:
                    continue
                j = random.choice(bins[bj])

                wi, wj = weights[i], weights[j]
                if loads[bi] - wi + wj > C:
                    continue
                if loads[bj] - wj + wi > C:
                    continue

                new_load_bi = loads[bi] - wi + wj
                new_load_bj = loads[bj] - wj + wi

                curr_m, curr_slack, curr_sq = curr_obj
                s_bi_old = C - loads[bi]
                s_bj_old = C - loads[bj]
                s_bi_new = C - new_load_bi
                s_bj_new = C - new_load_bj
                cand_obj = (curr_m,
                            curr_slack + (s_bi_new - s_bi_old) + (s_bj_new - s_bj_old),
                            curr_sq + (s_bi_new * s_bi_new - s_bi_old * s_bi_old) + (s_bj_new * s_bj_new - s_bj_old * s_bj_old))

                key = (i, j) if i < j else (j, i)
                is_tabu = tabu_swap.get(key, 0) > it
                if is_tabu and not aspiration(cand_obj):
                    continue

                if best_move_obj is None or cand_obj < best_move_obj:
                    best_move_obj = cand_obj
                    best_move = ("swap", i, j, bi, bj)

        # -------- Neighborhood 3: 2-1 exchange (classic stronger neighborhood) --------
        # Try to move two items out of target by swapping with one item in another bin.
        if (best_move is None or (no_improve > 150 and random.random() < 0.8)) and len(bins[target]) >= 2:
            trials = 180
            tgt_list = bins[target]
            for _ in range(trials):
                if len(tgt_list) < 2:
                    break
                a1, a2 = random.sample(tgt_list, 2)
                wa = weights[a1] + weights[a2]

                B = random.choice(bin_sample)
                if B == target or not bins[B]:
                    continue
                b1 = random.choice(bins[B])
                wb = weights[b1]

                # After move: target loses a1,a2 gains b1
                # other bin loses b1 gains a1,a2
                new_load_A = loads[target] - wa + wb
                new_load_B = loads[B] - wb + wa
                if new_load_A > C or new_load_B > C:
                    continue
                if new_load_A < 0:
                    continue

                new_m = m - 1 if new_load_A == 0 else m

                curr_m, curr_slack, curr_sq = curr_obj
                sA_old = C - loads[target]
                sB_old = C - loads[B]
                sA_new = C - new_load_A
                sB_new = C - new_load_B

                if new_load_A == 0:
                    new_slack = curr_slack - sA_old + (sB_new - sB_old)
                    new_sq = curr_sq - (sA_old * sA_old) + (sB_new * sB_new - sB_old * sB_old)
                else:
                    new_slack = curr_slack + (sA_new - sA_old) + (sB_new - sB_old)
                    new_sq = curr_sq + (sA_new * sA_new - sA_old * sA_old) + (sB_new * sB_new - sB_old * sB_old)

                cand_obj = (new_m, new_slack, new_sq)

                # Tabu: forbid putting a1/a2 back to target immediately; reuse reloc tabu as proxy.
                is_tabu = (tabu_reloc.get((a1, B), 0) > it) or (tabu_reloc.get((a2, B), 0) > it)
                if is_tabu and not aspiration(cand_obj):
                    continue

                if best_move_obj is None or cand_obj < best_move_obj:
                    best_move_obj = cand_obj
                    best_move = ("2-1", a1, a2, b1, target, B)

        # If still nothing, do a diversification kick (still within TS family: occasional random admissible move)
        if best_move is None:
            # random relocate attempt
            for _ in range(120):
                b_from = random.randrange(m)
                if not bins[b_from]:
                    continue
                i_item = random.choice(bins[b_from])
                w = weights[i_item]
                b_to = random.randrange(m)
                if b_to == b_from:
                    continue
                if loads[b_to] + w <= C:
                    best_move = ("reloc", i_item, b_from, b_to)
                    break
            if best_move is None:
                continue

        # -------- Apply move --------
        improved = False

        # tenure adapts with stagnation
        tenure = base_tenure + random.randint(0, base_tenure)
        if no_improve > 200:
            tenure = int(1.3 * tenure)
        if no_improve > 600:
            tenure = int(1.7 * tenure)

        if best_move[0] == "reloc":
            _, i_item, b_from, b_to = best_move
            apply_reloc(i_item, b_from, b_to, bins, loads, assign)

            # cleanup if emptied
            if loads[b_from] == 0:
                cleanup_empty(bins, loads, assign)

            # tabu: forbid moving the item back to source bin
            if b_from < len(bins):
                tabu_reloc[(i_item, b_from)] = it + tenure

        elif best_move[0] == "swap":
            _, i, j, bi, bj = best_move
            apply_swap(i, j, bi, bj, bins, loads, assign)
            key = (i, j) if i < j else (j, i)
            tabu_swap[key] = it + tenure

        else:  # "2-1"
            _, a1, a2, b1, A, B = best_move
            apply_2_1(a1, a2, b1, A, B, bins, loads, assign)
            if loads[A] == 0:
                cleanup_empty(bins, loads, assign)
            # tabu: prevent sending a1/a2 back to A soon (and b1 back to B soon)
            if A < len(bins):
                tabu_reloc[(a1, A)] = it + tenure
                tabu_reloc[(a2, A)] = it + tenure
            if B < len(bins):
                tabu_reloc[(b1, B)] = it + tenure

        # -------- Update best --------
        curr2 = obj_tuple(bins, loads)
        if curr2 < best_obj:
            best_obj = curr2
            best_bins = [lst[:] for lst in bins]
            best_loads = loads[:]
            best_assign = assign[:]
            improved = True
            last_best_it = it
            no_improve = 0
        else:
            no_improve += 1

        # -------- Strategic diversification (still TS-standard) --------
        # If stuck, apply a few random admissible swaps/relocs to move to a different region.
        if no_improve in (400, 900, 1600):
            kicks = 25 if no_improve < 900 else 45
            for _ in range(kicks):
                if it % TIME_CHECK_EVERY == 0 and time_exceeded():
                    break
                m2 = len(bins)
                if m2 <= 1:
                    break
                if random.random() < 0.6:
                    # random swap
                    b1 = random.randrange(m2)
                    b2 = random.randrange(m2)
                    if b1 == b2 or not bins[b1] or not bins[b2]:
                        continue
                    i = random.choice(bins[b1])
                    j = random.choice(bins[b2])
                    wi, wj = weights[i], weights[j]
                    if loads[b1] - wi + wj <= C and loads[b2] - wj + wi <= C:
                        apply_swap(i, j, b1, b2, bins, loads, assign)
                        key = (i, j) if i < j else (j, i)
                        tabu_swap[key] = it + tenure
                else:
                    # random relocate
                    b_from = random.randrange(m2)
                    if not bins[b_from]:
                        continue
                    i_item = random.choice(bins[b_from])
                    w = weights[i_item]
                    b_to = random.randrange(m2)
                    if b_to == b_from:
                        continue
                    if loads[b_to] + w <= C:
                        apply_reloc(i_item, b_from, b_to, bins, loads, assign)
                        if loads[b_from] == 0:
                            cleanup_empty(bins, loads, assign)
                        if b_from < len(bins):
                            tabu_reloc[(i_item, b_from)] = it + tenure

        # Optional: if we've improved bins count, intensify by resetting stagnation counters a bit
        if improved and best_obj[0] < curr_obj[0]:
            no_improve = 0

    # ---------------- Return best found ----------------
    packing = [lst[:] for lst in best_bins]
    bin_weights = [0] * len(packing)
    for b, items in enumerate(packing):
        s = 0
        for i in items:
            s += weights[i]
        bin_weights[b] = s

    # remove empties (should not exist)
    filtered_packing = []
    filtered_weights = []
    for b, items in enumerate(packing):
        if items:
            filtered_packing.append(items)
            filtered_weights.append(bin_weights[b])

    return {"packing": filtered_packing, "bin_weights": filtered_weights}
