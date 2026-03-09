import time
import random
from typing import List, Tuple, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = bin_capacity
    w = weights
    n = len(w)

    start = time.time()
    deadline = start + max(0.0, time_limit)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ---------- Basic helpers ----------
    items_desc = sorted(range(n), key=lambda i: (-w[i], i))

    def cleanup_empty(bins: List[List[int]], loads: List[int]) -> None:
        write = 0
        for i in range(len(bins)):
            if bins[i]:
                if write != i:
                    bins[write] = bins[i]
                    loads[write] = loads[i]
                write += 1
        del bins[write:]
        del loads[write:]

    def compute_positions(bins: List[List[int]]) -> Tuple[List[int], List[int]]:
        pos_bin = [-1] * n
        pos_idx = [-1] * n
        for b, lst in enumerate(bins):
            for j, it in enumerate(lst):
                pos_bin[it] = b
                pos_idx[it] = j
        return pos_bin, pos_idx

    def clone_solution(bins: List[List[int]], loads: List[int]) -> Tuple[List[List[int]], List[int]]:
        return [lst[:] for lst in bins], loads[:]

    def score_solution(loads: List[int]) -> int:
        # Tie-breaker: prefer fuller bins (min squared slack)
        s2 = 0
        for L in loads:
            d = C - L
            s2 += d * d
        return s2

    # ---------- Move primitives (O(1) updates using positions) ----------
    def relocate_item(bins: List[List[int]], loads: List[int], pos_bin: List[int], pos_idx: List[int], it: int, to_b: int) -> bool:
        bf = pos_bin[it]
        if bf == -1 or bf == to_b:
            return False
        wi = w[it]
        if to_b == len(bins):
            bins.append([])
            loads.append(0)
        if loads[to_b] + wi > C:
            if to_b == len(bins) - 1 and not bins[to_b]:
                bins.pop(); loads.pop()
            return False

        idx = pos_idx[it]
        last = bins[bf][-1]
        bins[bf][idx] = last
        pos_idx[last] = idx
        bins[bf].pop()
        loads[bf] -= wi

        pos_bin[it] = to_b
        pos_idx[it] = len(bins[to_b])
        bins[to_b].append(it)
        loads[to_b] += wi
        return True

    def swap_items(bins: List[List[int]], loads: List[int], pos_bin: List[int], pos_idx: List[int], a: int, b: int) -> bool:
        ba = pos_bin[a]
        bb = pos_bin[b]
        if ba == -1 or bb == -1 or ba == bb:
            return False
        wa = w[a]
        wb = w[b]
        if loads[ba] - wa + wb > C:
            return False
        if loads[bb] - wb + wa > C:
            return False
        ia = pos_idx[a]
        ib = pos_idx[b]
        bins[ba][ia], bins[bb][ib] = bins[bb][ib], bins[ba][ia]
        pos_bin[a], pos_bin[b] = bb, ba
        pos_idx[a], pos_idx[b] = ib, ia
        loads[ba] = loads[ba] - wa + wb
        loads[bb] = loads[bb] - wb + wa
        return True

    # ---------- Construction (multi-start friendly) ----------
    def construct_ffd(order: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        loads: List[int] = []
        for it in order:
            wi = w[it]
            placed = False
            for b in range(len(bins)):
                if loads[b] + wi <= C:
                    bins[b].append(it)
                    loads[b] += wi
                    placed = True
                    break
            if not placed:
                bins.append([it])
                loads.append(wi)
        return bins, loads

    def construct_bfd(order: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        loads: List[int] = []
        for it in order:
            wi = w[it]
            best_j = -1
            best_rem = C + 1
            for j, L in enumerate(loads):
                rem = C - (L + wi)
                if rem >= 0 and rem < best_rem:
                    best_rem = rem
                    best_j = j
            if best_j == -1:
                bins.append([it])
                loads.append(wi)
            else:
                bins[best_j].append(it)
                loads[best_j] += wi
        return bins, loads

    def construct_rcl_bfd(alpha: float) -> Tuple[List[List[int]], List[int]]:
        # Greedy randomized best-fit decreasing with restricted candidate list.
        # alpha in [0,1]: 0 nearly greedy, 1 more random.
        bins: List[List[int]] = []
        loads: List[int] = []
        for it in items_desc:
            wi = w[it]
            feasible: List[Tuple[int, int]] = []  # (rem, bin)
            for j, L in enumerate(loads):
                rem = C - (L + wi)
                if rem >= 0:
                    feasible.append((rem, j))
            if not feasible:
                bins.append([it])
                loads.append(wi)
                continue
            feasible.sort(key=lambda x: x[0])
            # Build RCL among best few
            m = len(feasible)
            rcl_size = max(1, min(m, int(1 + alpha * (min(8, m) - 1))))
            _, chosen_bin = random.choice(feasible[:rcl_size])
            bins[chosen_bin].append(it)
            loads[chosen_bin] += wi
        return bins, loads

    # ---------- Repair insertion (regret-based best-fit) ----------
    def regret_reinsert(bins: List[List[int]], loads: List[int], items: List[int]) -> None:
        # Insert items one by one choosing the item with maximum regret.
        # For each item compute best and second-best remaining capacity after insertion.
        # If no feasible bin => create new bin.
        while items:
            if time.time() >= deadline:
                # finish fast with simple best-fit
                for it in items:
                    wi = w[it]
                    best_j = -1
                    best_rem = C + 1
                    for j, L in enumerate(loads):
                        rem = C - (L + wi)
                        if rem >= 0 and rem < best_rem:
                            best_rem = rem
                            best_j = j
                    if best_j == -1:
                        bins.append([it]); loads.append(wi)
                    else:
                        bins[best_j].append(it); loads[best_j] += wi
                return

            best_pick = -1
            best_target = -1
            best_is_new = False
            best_regret = -1
            best_primary = None

            # sample if very large
            cand = items
            if len(items) > 80:
                cand = random.sample(items, 80)

            for it in cand:
                wi = w[it]
                best1 = (C + 1, -1)  # rem, bin
                best2 = (C + 1, -1)
                for j, L in enumerate(loads):
                    rem = C - (L + wi)
                    if rem >= 0:
                        if rem < best1[0]:
                            best2 = best1
                            best1 = (rem, j)
                        elif rem < best2[0]:
                            best2 = (rem, j)

                if best1[1] == -1:
                    # forced new bin; prioritize large items
                    regret = 10**9 + wi
                    primary = 10**9
                    target = -1
                    is_new = True
                else:
                    # regret = how much worse second option is compared to best
                    regret = (best2[0] - best1[0]) if best2[1] != -1 else (C - best1[0] + 1)
                    primary = best1[0]  # smaller rem is better
                    target = best1[1]
                    is_new = False

                if (regret > best_regret) or (regret == best_regret and primary is not None and best_primary is not None and primary < best_primary):
                    best_regret = regret
                    best_primary = primary
                    best_pick = it
                    best_target = target
                    best_is_new = is_new

            # apply
            items.remove(best_pick)
            wi = w[best_pick]
            if best_is_new or best_target == -1:
                bins.append([best_pick])
                loads.append(wi)
            else:
                bins[best_target].append(best_pick)
                loads[best_target] += wi

    # ---------- Stronger VND (bin elimination + limited exchanges) ----------
    def try_empty_bin_with_chain(bins: List[List[int]], loads: List[int], b: int, max_chain: int = 3) -> bool:
        # Attempt to empty bin b by relocating items, allowing a short ejection chain:
        # move x to bin j; if x doesn't fit, try swap with y in j (1-1) then place y elsewhere, etc.
        if not bins[b]:
            return True
        pos_bin, pos_idx = compute_positions(bins)
        items = bins[b][:]
        items.sort(key=lambda i: -w[i])

        saved_bins, saved_loads = clone_solution(bins, loads)

        # helper recursion
        def place_item(it: int, forbid_bin: int, depth: int) -> bool:
            wi = w[it]
            # best-fit order of target bins by resulting slack
            candidates = []
            for j in range(len(bins)):
                if j == forbid_bin or j == pos_bin[it]:
                    continue
                rem = C - (loads[j] + wi)
                if rem >= 0:
                    candidates.append((rem, j))
            candidates.sort(key=lambda x: x[0])
            # try a few best bins
            for _, j in candidates[:8]:
                if relocate_item(bins, loads, pos_bin, pos_idx, it, j):
                    return True

            if depth >= max_chain:
                return False

            # try 1-1 swap into some promising bins
            # pick a few bins with enough total slack when swapping
            bin_order = sorted(range(len(bins)), key=lambda j: C - loads[j])
            for j in bin_order[:8]:
                if j == forbid_bin or j == pos_bin[it]:
                    continue
                # try swapping with a small subset of items from bin j
                if not bins[j]:
                    continue
                # sample items; prefer larger to make room
                cand_y = bins[j]
                if len(cand_y) > 10:
                    cand_y = random.sample(cand_y, 10)
                cand_y = sorted(cand_y, key=lambda x: -w[x])
                for y in cand_y:
                    wy = w[y]
                    # after swapping it into j and y into current bin of it (which is b during emptying)
                    # But we actually want to move it into j; y becomes displaced and must be placed elsewhere.
                    if loads[j] - wy + wi > C:
                        continue
                    # perform swap it <-> y
                    bi = pos_bin[it]
                    by = pos_bin[y]
                    if bi == -1 or by == -1 or bi == by:
                        continue
                    if not swap_items(bins, loads, pos_bin, pos_idx, it, y):
                        continue
                    # now y is in bi; try to place y elsewhere (not in forbid_bin)
                    if place_item(y, forbid_bin, depth + 1):
                        return True
                    # rollback swap
                    swap_items(bins, loads, pos_bin, pos_idx, it, y)

            return False

        for it in items:
            if time.time() >= deadline:
                bins[:] = saved_bins
                loads[:] = saved_loads
                return False
            if pos_bin[it] == -1:
                continue
            if not place_item(it, b, 0):
                bins[:] = saved_bins
                loads[:] = saved_loads
                return False

        cleanup_empty(bins, loads)
        return True

    def vnd_improve(bins: List[List[int]], loads: List[int]) -> None:
        # Deterministic VND loop:
        # (A) eliminate bins (lightest first) with chain moves
        # (B) if no bin eliminated, improve fullness with best improving relocations/swaps (bounded)
        while True:
            if time.time() >= deadline:
                return
            cleanup_empty(bins, loads)
            B = len(bins)
            if B <= 1:
                return

            eliminated = False
            # try to eliminate several bins per call
            order_bins = sorted(range(B), key=lambda b: loads[b])
            for b in order_bins:
                if time.time() >= deadline:
                    return
                if b >= len(bins):
                    continue
                if not bins[b]:
                    continue
                # quick bound: if total free space excluding b is < load[b], impossible
                free = 0
                for j in range(len(bins)):
                    if j != b:
                        free += C - loads[j]
                if free < loads[b]:
                    continue
                if try_empty_bin_with_chain(bins, loads, b, max_chain=3):
                    eliminated = True
                    break

            if eliminated:
                continue

            # fullness improvement (tie-breaking) - bounded steepest improvements
            base_sc = score_solution(loads)
            pos_bin, pos_idx = compute_positions(bins)

            # try a limited number of improving relocations
            improved = False
            # prioritize items in low-filled bins
            bin_order = sorted(range(len(bins)), key=lambda b: loads[b])
            item_pool: List[int] = []
            for b in bin_order[: min(len(bin_order), 8)]:
                item_pool.extend(bins[b])
            if len(item_pool) < min(n, 30):
                # add some random items
                extra = list(range(n))
                random.shuffle(extra)
                item_pool.extend(extra[:30])

            # unique
            seen = set()
            uniq_pool = []
            for it in item_pool:
                if it not in seen:
                    seen.add(it)
                    uniq_pool.append(it)

            for it in uniq_pool[: min(len(uniq_pool), 80)]:
                if time.time() >= deadline:
                    return
                bf = pos_bin[it]
                if bf == -1:
                    continue
                wi = w[it]
                # examine a few best target bins by slack after insertion
                cand_bins = []
                for j in range(len(bins)):
                    if j == bf:
                        continue
                    rem = C - (loads[j] + wi)
                    if rem >= 0:
                        cand_bins.append((rem, j))
                cand_bins.sort(key=lambda x: x[0])

                for _, tb in cand_bins[:10]:
                    old = (C - loads[bf]) ** 2 + (C - loads[tb]) ** 2
                    newL_bf = loads[bf] - wi
                    newL_tb = loads[tb] + wi
                    new = (C - newL_bf) ** 2 + (C - newL_tb) ** 2
                    if base_sc - old + new < base_sc:
                        if relocate_item(bins, loads, pos_bin, pos_idx, it, tb):
                            cleanup_empty(bins, loads)
                            improved = True
                            break
                if improved:
                    break

            if improved:
                continue

            # try some improving swaps
            pos_bin, pos_idx = compute_positions(bins)
            trials = 400
            for _ in range(trials):
                if time.time() >= deadline:
                    return
                a = random.randrange(n)
                b = random.randrange(n)
                if a == b:
                    continue
                ba = pos_bin[a]
                bb = pos_bin[b]
                if ba == -1 or bb == -1 or ba == bb:
                    continue
                wa, wb = w[a], w[b]
                if loads[ba] - wa + wb > C or loads[bb] - wb + wa > C:
                    continue
                old = (C - loads[ba]) ** 2 + (C - loads[bb]) ** 2
                newL_a = loads[ba] - wa + wb
                newL_b = loads[bb] - wb + wa
                new = (C - newL_a) ** 2 + (C - newL_b) ** 2
                if base_sc - old + new < base_sc:
                    if swap_items(bins, loads, pos_bin, pos_idx, a, b):
                        improved = True
                        break

            if not improved:
                return

    # ---------- Shaking neighborhoods (stronger destroy/repair) ----------
    def shake(bins: List[List[int]], loads: List[int], k: int) -> Tuple[List[List[int]], List[int]]:
        sbins, sloads = clone_solution(bins, loads)
        cleanup_empty(sbins, sloads)
        if not sbins:
            return sbins, sloads

        pos_bin, pos_idx = compute_positions(sbins)

        # k=1..3: small perturbations
        if k == 1:
            # relocate one random item to a random feasible bin
            it = random.randrange(n)
            bf = pos_bin[it]
            if bf != -1:
                targets = list(range(len(sbins)))
                random.shuffle(targets)
                for tb in targets[: min(10, len(targets))]:
                    if tb != bf and sloads[tb] + w[it] <= C:
                        relocate_item(sbins, sloads, pos_bin, pos_idx, it, tb)
                        break
            cleanup_empty(sbins, sloads)
            return sbins, sloads

        if k == 2:
            # swap two random items from different bins
            for _ in range(40):
                a = random.randrange(n)
                b = random.randrange(n)
                if a != b and swap_items(sbins, sloads, pos_bin, pos_idx, a, b):
                    break
            cleanup_empty(sbins, sloads)
            return sbins, sloads

        if k == 3:
            # empty-part of a light bin (push items out)
            order_bins = sorted(range(len(sbins)), key=lambda b: sloads[b])
            b = order_bins[0]
            if sbins[b]:
                chosen = sbins[b][:]
                random.shuffle(chosen)
                chosen = chosen[: min(3, len(chosen))]
                chosen.sort(key=lambda i: -w[i])
                for it in chosen:
                    bf = pos_bin[it]
                    targets = list(range(len(sbins)))
                    random.shuffle(targets)
                    for tb in targets:
                        if tb != bf and sloads[tb] + w[it] <= C:
                            relocate_item(sbins, sloads, pos_bin, pos_idx, it, tb)
                            break
            cleanup_empty(sbins, sloads)
            return sbins, sloads

        # k>=4 destroy/repair
        # remove r items biased toward problematic bins (low fill and very high fill)
        B = len(sbins)
        r = min(n, 8 + 3 * k)

        # pick bins to draw from
        order_low = sorted(range(B), key=lambda b: sloads[b])
        order_high = sorted(range(B), key=lambda b: -sloads[b])
        focus_bins = order_low[: min(5, B)] + order_high[: min(3, B)]
        # ensure unique
        fb = []
        seenb = set()
        for b in focus_bins:
            if b not in seenb:
                seenb.add(b)
                fb.append(b)

        removed: List[int] = []
        # remove from focus bins first
        for b in fb:
            if len(removed) >= r:
                break
            if not sbins[b]:
                continue
            # remove a couple items from this bin
            take = min(len(sbins[b]), max(1, (r - len(removed)) // max(1, len(fb))))
            # prefer items that create difficult gaps (medium items)
            cand = sbins[b][:]
            if len(cand) > 12:
                cand = random.sample(cand, 12)
            cand.sort(key=lambda it: abs((C // 2) - w[it]))
            for it in cand[:take]:
                if len(removed) >= r:
                    break
                bf = pos_bin[it]
                if bf == -1:
                    continue
                idx = pos_idx[it]
                last = sbins[bf][-1]
                sbins[bf][idx] = last
                pos_idx[last] = idx
                sbins[bf].pop()
                sloads[bf] -= w[it]
                pos_bin[it] = -1
                pos_idx[it] = -1
                removed.append(it)

        # if still need removals, remove random items
        if len(removed) < r:
            cand_items = list(range(n))
            random.shuffle(cand_items)
            for it in cand_items:
                if len(removed) >= r:
                    break
                bf = pos_bin[it]
                if bf == -1:
                    continue
                idx = pos_idx[it]
                last = sbins[bf][-1]
                sbins[bf][idx] = last
                pos_idx[last] = idx
                sbins[bf].pop()
                sloads[bf] -= w[it]
                pos_bin[it] = -1
                pos_idx[it] = -1
                removed.append(it)

        cleanup_empty(sbins, sloads)
        removed.sort(key=lambda i: -w[i])
        regret_reinsert(sbins, sloads, removed)
        cleanup_empty(sbins, sloads)
        return sbins, sloads

    # ---------- Initial best (multi-start) ----------
    def make_initial() -> Tuple[List[List[int]], List[int]]:
        # Mix constructions; keep best.
        best = None
        best_key = None

        # Deterministic baselines
        for cons in (construct_bfd, construct_ffd):
            bins, loads = cons(items_desc)
            cleanup_empty(bins, loads)
            vnd_improve(bins, loads)
            key = (len(bins), score_solution(loads))
            if best is None or key < best_key:
                best, best_key = (bins, loads), key

        # Randomized RCL starts
        for alpha in (0.15, 0.35, 0.65):
            if time.time() >= deadline:
                break
            bins, loads = construct_rcl_bfd(alpha)
            cleanup_empty(bins, loads)
            vnd_improve(bins, loads)
            key = (len(bins), score_solution(loads))
            if key < best_key:
                best, best_key = (bins, loads), key

        return clone_solution(best[0], best[1])

    best_bins, best_loads = make_initial()
    best_num = len(best_bins)
    best_sc = score_solution(best_loads)

    curr_bins, curr_loads = clone_solution(best_bins, best_loads)

    # ---------- Main VNS loop ----------
    # Use a fixed iteration count but try to spend full time by setting it high.
    # Also allow time checks to terminate early.
    kmax = 9

    # More time => more iterations; still a fixed number for a given call.
    # (No penalty up to 100s; caller may give smaller time_limit.)
    iter_budget = max(2000, 250 * n // 10 + 3500)

    # restart cadence
    restart_every = 120

    iters = 0
    while iters < iter_budget:
        if time.time() >= deadline:
            break
        iters += 1

        k = 1
        while k <= kmax:
            if time.time() >= deadline:
                break
            cand_bins, cand_loads = shake(curr_bins, curr_loads, k)
            vnd_improve(cand_bins, cand_loads)
            cleanup_empty(cand_bins, cand_loads)

            cand_key = (len(cand_bins), score_solution(cand_loads))
            curr_key = (len(curr_bins), score_solution(curr_loads))

            if cand_key < curr_key:
                curr_bins, curr_loads = cand_bins, cand_loads
                k = 1
            else:
                k += 1

            if cand_key < (best_num, best_sc):
                best_bins, best_loads = clone_solution(cand_bins, cand_loads)
                best_num, best_sc = cand_key

        # Periodic restarts / diversification
        if iters % restart_every == 0 and time.time() < deadline:
            # restart from best and apply a larger shake
            curr_bins, curr_loads = clone_solution(best_bins, best_loads)
            curr_bins, curr_loads = shake(curr_bins, curr_loads, random.choice([4, 5, 6, 7, 8, 9]))
            vnd_improve(curr_bins, curr_loads)

        # Occasional fresh randomized construction (multi-start within VNS)
        if iters % (restart_every * 2) == 0 and time.time() < deadline:
            nbins, nloads = construct_rcl_bfd(random.choice([0.2, 0.4, 0.7]))
            vnd_improve(nbins, nloads)
            nkey = (len(nbins), score_solution(nloads))
            if nkey < (best_num, best_sc):
                best_bins, best_loads = clone_solution(nbins, nloads)
                best_num, best_sc = nkey
            curr_bins, curr_loads = clone_solution(best_bins, best_loads)

    cleanup_empty(best_bins, best_loads)
    return {"packing": best_bins, "bin_weights": best_loads}
