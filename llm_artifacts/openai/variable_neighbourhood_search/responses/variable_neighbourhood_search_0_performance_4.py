import time
import random
from typing import List, Tuple, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = bin_capacity

    # ---------------- Objective ----------------
    def obj(num_bins: int, loads_nonempty: List[int]) -> Tuple[int, int, int]:
        waste = num_bins * C - sum(loads_nonempty)
        ss = sum(x * x for x in loads_nonempty)
        return (num_bins, waste, -ss)

    # ---------------- Solution representation helpers ----------------
    # bins: List[List[int]] of item indices
    # loads: List[int]
    # assign[item] = bin id
    # pos[item] = position of item in bins[assign[item]]

    def remove_from_bin(bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int], it: int):
        b = assign[it]
        lst = bins[b]
        i = pos[it]
        last = lst[-1]
        lst[i] = last
        pos[last] = i
        lst.pop()
        loads[b] -= weights[it]

    def add_to_bin(bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int], it: int, b: int):
        assign[it] = b
        pos[it] = len(bins[b])
        bins[b].append(it)
        loads[b] += weights[it]

    def do_relocate(bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int], it: int, b_to: int):
        remove_from_bin(bins, loads, assign, pos, it)
        add_to_bin(bins, loads, assign, pos, it, b_to)

    def feasible_relocate(loads: List[int], it: int, b_to: int) -> bool:
        return loads[b_to] + weights[it] <= C

    def do_swap(bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int], i1: int, i2: int):
        b1 = assign[i1]
        b2 = assign[i2]
        if b1 == b2:
            return

        # Remove both (order matters for pos updates, but our remove is safe)
        remove_from_bin(bins, loads, assign, pos, i1)
        remove_from_bin(bins, loads, assign, pos, i2)
        add_to_bin(bins, loads, assign, pos, i1, b2)
        add_to_bin(bins, loads, assign, pos, i2, b1)

    def feasible_swap(loads: List[int], assign: List[int], i1: int, i2: int) -> bool:
        b1 = assign[i1]
        b2 = assign[i2]
        if b1 == b2:
            return False
        w1 = weights[i1]
        w2 = weights[i2]
        return (loads[b1] - w1 + w2 <= C) and (loads[b2] - w2 + w1 <= C)

    def nonempty_bins(bins: List[List[int]]) -> List[int]:
        return [b for b in range(len(bins)) if bins[b]]

    def current_obj(bins: List[List[int]], loads: List[int]) -> Tuple[int, int, int]:
        lds = [loads[b] for b in range(len(bins)) if bins[b]]
        return obj(len(lds), lds)

    def pack_compact(bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int]):
        # Remove empties, remap bin ids densely, update assign.
        mapping = [-1] * len(bins)
        new_bins: List[List[int]] = []
        new_loads: List[int] = []
        for b in range(len(bins)):
            if bins[b]:
                mapping[b] = len(new_bins)
                new_bins.append(bins[b])
                new_loads.append(loads[b])
        for it in range(n):
            assign[it] = mapping[assign[it]]
        # rebuild pos for safety
        new_pos = pos
        for b, lst in enumerate(new_bins):
            for i, it in enumerate(lst):
                new_pos[it] = i
        return new_bins, new_loads

    # ---------------- Construction: randomized best-fit decreasing ----------------
    items_desc = list(range(n))
    items_desc.sort(key=lambda i: weights[i], reverse=True)

    def construct_rbfd(alpha: float = 0.35) -> Tuple[List[List[int]], List[int], List[int], List[int]]:
        # alpha controls randomness among a restricted candidate list of bins.
        bins: List[List[int]] = []
        loads: List[int] = []
        assign = [-1] * n
        pos = [-1] * n

        # slight shuffle inside equal-weight blocks
        items = items_desc[:]
        i = 0
        while i < n:
            j = i + 1
            wi = weights[items[i]]
            while j < n and weights[items[j]] == wi:
                j += 1
            if j - i > 1:
                block = items[i:j]
                random.shuffle(block)
                items[i:j] = block
            i = j

        for it in items:
            w = weights[it]
            best_b = -1
            best_rem = None

            # Consider candidate bins sorted by remaining capacity (best-fit).
            # To keep it fast, sample a subset when many bins exist.
            m = len(bins)
            if m == 0:
                bins.append([])
                loads.append(0)
                add_to_bin(bins, loads, assign, pos, it, 0)
                continue

            # Sample some bins + also check a few of the tightest bins by rem.
            sample_sz = 18 if m > 18 else m
            cand_bins = random.sample(range(m), sample_sz) if m > sample_sz else list(range(m))

            # Also include a few bins likely good for best-fit (small remaining)
            # by picking bins with high load.
            if m > 8:
                # pick 8 heaviest-load bins cheaply by random tournament
                extra = set()
                for _ in range(18):
                    b = random.randrange(m)
                    extra.add(b)
                cand_bins.extend(extra)

            # Evaluate candidates
            for b in cand_bins:
                rem = C - loads[b]
                if rem >= w:
                    r = rem - w
                    if best_rem is None or r < best_rem:
                        best_rem = r
                        best_b = b

            if best_b == -1:
                # open new bin
                b = len(bins)
                bins.append([])
                loads.append(0)
                add_to_bin(bins, loads, assign, pos, it, b)
            else:
                # With probability alpha, allow a slightly worse choice among top few
                if best_rem is not None and random.random() < alpha and len(cand_bins) >= 6:
                    feasible = []
                    for b in set(cand_bins):
                        rem = C - loads[b]
                        if rem >= w:
                            feasible.append((rem - w, b))
                    feasible.sort()
                    top = feasible[: min(5, len(feasible))]
                    _, chosen = random.choice(top)
                    add_to_bin(bins, loads, assign, pos, it, chosen)
                else:
                    add_to_bin(bins, loads, assign, pos, it, best_b)

        return bins, loads, assign, pos

    # Multi-start construction within early time slice
    best_bins: List[List[int]] = []
    best_loads: List[int] = []
    best_assign: List[int] = []
    best_pos: List[int] = []

    # Spend up to ~12% of time budget on initial construction attempts (but cap attempts)
    init_deadline = start + min(time_limit * 0.12, 3.0)
    attempts = 0
    while attempts < 40 and time.time() < init_deadline:
        bins, loads, assign, pos = construct_rbfd(alpha=0.45)
        o = current_obj(bins, loads)
        if not best_bins or o < current_obj(best_bins, best_loads):
            best_bins = [lst[:] for lst in bins]
            best_loads = loads[:]
            best_assign = assign[:]
            best_pos = pos[:]
        attempts += 1

    if not best_bins:
        best_bins, best_loads, best_assign, best_pos = construct_rbfd(alpha=0.35)

    best_bins, best_loads = pack_compact(best_bins, best_loads, best_assign, best_pos)
    best_o = current_obj(best_bins, best_loads)

    # ---------------- VND (Variable Neighbourhood Descent) ----------------
    def vnd(bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int], max_passes: int = 80):
        nonlocal best_bins, best_loads, best_assign, best_pos, best_o

        passes = 0
        while passes < max_passes:
            if time.time() - start > time_limit:
                return

            improved = False

            # Compact occasionally to remove empties
            if passes % 7 == 0:
                bins[:], loads[:] = pack_compact(bins, loads, assign, pos)

            b_ids = nonempty_bins(bins)
            # --- N1: relocate with strong bias to empty small bins ---
            b_ids.sort(key=lambda b: loads[b])
            for b_from in b_ids[: min(len(b_ids), 18)]:
                if time.time() - start > time_limit:
                    return
                if not bins[b_from]:
                    continue

                # Try moving items out; heavier-first to free capacity faster
                items = bins[b_from][:]
                items.sort(key=lambda it: weights[it], reverse=True)

                for it in items:
                    w = weights[it]
                    # best-fit destination search over a limited set of bins: prefer tight bins
                    # Evaluate bins ordered by residual after placement
                    candidates = []
                    # Use a mix: a few tightest by current rem and a few random
                    other_bins = [b for b in b_ids if b != b_from]
                    if not other_bins:
                        continue

                    # random sample
                    sample = random.sample(other_bins, min(18, len(other_bins)))
                    for b_to in sample:
                        rem = C - loads[b_to]
                        if rem >= w:
                            candidates.append((rem - w, b_to))

                    if not candidates:
                        continue
                    candidates.sort()

                    # Try up to 6 best
                    for _, b_to in candidates[:6]:
                        do_relocate(bins, loads, assign, pos, it, b_to)
                        o = current_obj(bins, loads)
                        if o < best_o:
                            best_o = o
                            best_bins = [lst[:] for lst in bins]
                            best_loads = loads[:]
                            best_assign = assign[:]
                            best_pos = pos[:]
                            improved = True
                            break
                        # revert
                        do_relocate(bins, loads, assign, pos, it, b_from)

                        if time.time() - start > time_limit:
                            return
                    if improved:
                        break
                if improved:
                    break

            if improved:
                passes += 1
                continue

            # --- N2: swap (1-1) focused on light/heavy bins ---
            b_ids = nonempty_bins(bins)
            if len(b_ids) >= 2:
                b_ids_sorted = sorted(b_ids, key=lambda b: loads[b])
                light = b_ids_sorted[: max(1, len(b_ids_sorted) // 3)]
                heavy = b_ids_sorted[-max(1, len(b_ids_sorted) // 3):]
                pool1 = []
                for b in light:
                    pool1.extend(bins[b])
                pool2 = []
                for b in heavy:
                    pool2.extend(bins[b])
                if not pool1:
                    pool1 = list(range(n))
                if not pool2:
                    pool2 = list(range(n))

                s1 = random.sample(pool1, min(50, len(pool1)))
                s2 = random.sample(pool2, min(70, len(pool2)))

                for i1 in s1:
                    if time.time() - start > time_limit:
                        return
                    for i2 in s2:
                        if i1 == i2:
                            continue
                        if feasible_swap(loads, assign, i1, i2):
                            do_swap(bins, loads, assign, pos, i1, i2)
                            o = current_obj(bins, loads)
                            if o < best_o:
                                best_o = o
                                best_bins = [lst[:] for lst in bins]
                                best_loads = loads[:]
                                best_assign = assign[:]
                                best_pos = pos[:]
                                improved = True
                                break
                            # revert by swapping back
                            do_swap(bins, loads, assign, pos, i1, i2)
                    if improved:
                        break

            if improved:
                passes += 1
                continue

            # --- N3: 2-1 move (move two items from a bin into another, and possibly move one out) ---
            # This helps restructure tightness and can enable later bin emptying.
            b_ids = nonempty_bins(bins)
            b_ids.sort(key=lambda b: loads[b])
            tried_bins = b_ids[: min(10, len(b_ids))]
            for b_from in tried_bins:
                if time.time() - start > time_limit:
                    return
                lst = bins[b_from]
                if len(lst) < 2:
                    continue
                # pick a few pairs
                cand = lst[:]
                random.shuffle(cand)
                cand = cand[: min(9, len(cand))]
                pairs = []
                for i in range(len(cand)):
                    for j in range(i + 1, len(cand)):
                        pairs.append((cand[i], cand[j]))
                random.shuffle(pairs)
                pairs = pairs[:12]

                other_bins = [b for b in b_ids if b != b_from]
                if not other_bins:
                    continue

                for it1, it2 in pairs:
                    wsum = weights[it1] + weights[it2]
                    # choose candidate target bins with enough space
                    targets = []
                    sample = random.sample(other_bins, min(18, len(other_bins)))
                    for b_to in sample:
                        rem = C - loads[b_to]
                        if rem >= wsum:
                            targets.append((rem - wsum, b_to))
                    if not targets:
                        continue
                    targets.sort()
                    for _, b_to in targets[:4]:
                        # try relocate both
                        do_relocate(bins, loads, assign, pos, it1, b_to)
                        do_relocate(bins, loads, assign, pos, it2, b_to)
                        o = current_obj(bins, loads)
                        if o < best_o:
                            best_o = o
                            best_bins = [lst2[:] for lst2 in bins]
                            best_loads = loads[:]
                            best_assign = assign[:]
                            best_pos = pos[:]
                            improved = True
                            break
                        # revert
                        do_relocate(bins, loads, assign, pos, it2, b_from)
                        do_relocate(bins, loads, assign, pos, it1, b_from)
                        if time.time() - start > time_limit:
                            return
                    if improved:
                        break
                if improved:
                    break

            if improved:
                passes += 1
                continue

            # --- N4: explicit bin-emptying via short ejection chain (classic intensification for bin packing VNS) ---
            # Pick a small bin and attempt to reinsert all its items into other bins,
            # allowing a small number of ejections.
            b_ids = nonempty_bins(bins)
            b_ids.sort(key=lambda b: loads[b])
            for b_from in b_ids[: min(6, len(b_ids))]:
                if time.time() - start > time_limit:
                    return
                if not bins[b_from]:
                    continue

                items_to_place = bins[b_from][:]
                items_to_place.sort(key=lambda it: weights[it], reverse=True)

                # Temporarily remove all items from b_from
                removed = items_to_place[:]
                for it in removed:
                    remove_from_bin(bins, loads, assign, pos, it)

                # Greedy reinsert with up to 2 ejections total
                ejections = 0
                success = True
                for it in items_to_place:
                    w = weights[it]
                    # best-fit among other bins
                    best = None
                    for b_to in b_ids:
                        if b_to == b_from or not bins[b_to]:
                            continue
                        rem = C - loads[b_to]
                        if rem >= w:
                            score = rem - w
                            if best is None or score < best[0]:
                                best = (score, b_to)
                    if best is not None:
                        add_to_bin(bins, loads, assign, pos, it, best[1])
                        continue

                    # try one ejection: pick a bin and swap out one item to make space
                    if ejections >= 2:
                        success = False
                        break
                    # choose a candidate bin where removing a small item makes room
                    candidate_bins = [b for b in b_ids if b != b_from and bins[b]]
                    random.shuffle(candidate_bins)
                    found = False
                    for b_to in candidate_bins[:12]:
                        need = w - (C - loads[b_to])
                        if need <= 0:
                            continue
                        # find an item in b_to with weight >= need (eject the smallest that works)
                        best_eject = None
                        for jt in bins[b_to]:
                            if weights[jt] >= need:
                                if best_eject is None or weights[jt] < weights[best_eject]:
                                    best_eject = jt
                        if best_eject is None:
                            continue
                        # eject best_eject to b_from (temporary holding)
                        do_relocate(bins, loads, assign, pos, best_eject, b_from)
                        ejections += 1
                        # now place it into b_to
                        add_to_bin(bins, loads, assign, pos, it, b_to)
                        found = True
                        break
                    if not found:
                        success = False
                        break

                if success:
                    # If b_from ended up empty, we reduced bin count.
                    o = current_obj(bins, loads)
                    if o < best_o:
                        best_o = o
                        best_bins = [lst2[:] for lst2 in bins]
                        best_loads = loads[:]
                        best_assign = assign[:]
                        best_pos = pos[:]
                        improved = True
                    if not improved:
                        # Even if not globally best, keep the change only if it improved current objective
                        pass
                if not improved:
                    # Revert fully: move everything currently in b_from back to original bin b_from,
                    # and restore removed items to b_from.
                    # Simplest safe revert: rebuild from best snapshot is too expensive; instead,
                    # undo by moving items back based on membership.
                    # We'll just restore by reconstructing from saved state if no improvement.
                    # Take a cheap route: restore from best state (still within VND, acceptable rarely).
                    bins[:] = [lst[:] for lst in best_bins]
                    loads[:] = best_loads[:]
                    assign[:] = best_assign[:]
                    pos[:] = best_pos[:]
                else:
                    # Compact after a successful empty attempt
                    bins[:], loads[:] = pack_compact(bins, loads, assign, pos)
                    break

            if improved:
                passes += 1
                continue

            break

    # ---------------- Shaking neighborhoods (for VNS) ----------------
    def best_fit_bin_for_item(bins: List[List[int]], loads: List[int], forbid_bin: int, it: int) -> int:
        w = weights[it]
        best_b = -1
        best_r = None
        for b in range(len(bins)):
            if b == forbid_bin or not bins[b]:
                continue
            rem = C - loads[b]
            if rem >= w:
                r = rem - w
                if best_r is None or r < best_r:
                    best_r = r
                    best_b = b
        return best_b

    def shaking(bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int], k: int):
        # k controls intensity. Use destroy/repair centered on a random small/medium bin.
        b_ids = nonempty_bins(bins)
        if len(b_ids) <= 1:
            return

        # choose target bin biased to small bins (more likely to enable bin reduction)
        b_ids.sort(key=lambda b: loads[b])
        pick_from = b_ids[: max(2, len(b_ids) // 2)]
        b0 = random.choice(pick_from)
        if not bins[b0]:
            return

        # number of removed items
        t = min(len(bins[b0]), 1 + (k // 2))
        t = max(1, t)

        removed = bins[b0][:]
        random.shuffle(removed)
        removed = removed[:t]

        # remove them
        for it in removed:
            remove_from_bin(bins, loads, assign, pos, it)

        # try to reinsert with randomized best-fit; allow opening a new bin rarely
        for it in removed:
            if time.time() - start > time_limit:
                break
            b = best_fit_bin_for_item(bins, loads, b0, it)
            if b != -1 and random.random() < 0.9:
                add_to_bin(bins, loads, assign, pos, it, b)
            else:
                # fallback: try a few random bins
                b_ids2 = nonempty_bins(bins)
                b_ids2 = [bb for bb in b_ids2 if bb != b0]
                placed = False
                if b_ids2:
                    for _ in range(8):
                        bb = random.choice(b_ids2)
                        if loads[bb] + weights[it] <= C:
                            add_to_bin(bins, loads, assign, pos, it, bb)
                            placed = True
                            break
                if not placed:
                    # open a new bin (diversification)
                    new_b = len(bins)
                    bins.append([])
                    loads.append(0)
                    add_to_bin(bins, loads, assign, pos, it, new_b)

        # additional random swaps/relocations proportional to k
        moves = k
        for _ in range(moves):
            if time.time() - start > time_limit:
                break
            if random.random() < 0.6:
                it = random.randrange(n)
                b_from = assign[it]
                b_ids3 = nonempty_bins(bins)
                if len(b_ids3) <= 1:
                    continue
                b_to = random.choice(b_ids3)
                if b_to != b_from and loads[b_to] + weights[it] <= C:
                    do_relocate(bins, loads, assign, pos, it, b_to)
            else:
                i1 = random.randrange(n)
                i2 = random.randrange(n)
                if i1 != i2 and feasible_swap(loads, assign, i1, i2):
                    do_swap(bins, loads, assign, pos, i1, i2)

        # occasional compact
        if random.random() < 0.25:
            bins[:], loads[:] = pack_compact(bins, loads, assign, pos)

    # ---------------- Main VNS loop ----------------
    # Fixed number of iterations; within each, VNS explores k=1..k_max.
    k_max = 22
    max_iters = 420

    # Intensify from best initial
    cur_bins = [lst[:] for lst in best_bins]
    cur_loads = best_loads[:]
    cur_assign = best_assign[:]
    cur_pos = best_pos[:]
    vnd(cur_bins, cur_loads, cur_assign, cur_pos, max_passes=90)

    # synchronize best in case vnd improved
    if current_obj(cur_bins, cur_loads) < best_o:
        best_bins = [lst[:] for lst in cur_bins]
        best_loads = cur_loads[:]
        best_assign = cur_assign[:]
        best_pos = cur_pos[:]
        best_o = current_obj(best_bins, best_loads)

    iters = 0
    while iters < max_iters:
        if time.time() - start > time_limit:
            break

        k = 1
        while k <= k_max:
            if time.time() - start > time_limit:
                break

            # start from best
            bins = [lst[:] for lst in best_bins]
            loads = best_loads[:]
            assign = best_assign[:]
            pos = best_pos[:]

            shaking(bins, loads, assign, pos, k)
            vnd(bins, loads, assign, pos, max_passes=60)

            o = current_obj(bins, loads)
            if o < best_o:
                best_o = o
                best_bins = [lst[:] for lst in bins]
                best_loads = loads[:]
                best_assign = assign[:]
                best_pos = pos[:]
                k = 1
            else:
                k += 1

        iters += 1

    # Final cleanup
    bins = [lst[:] for lst in best_bins]
    loads = best_loads[:]
    assign = best_assign[:]
    pos = best_pos[:]
    bins, loads = pack_compact(bins, loads, assign, pos)

    return {"packing": bins, "bin_weights": loads}
