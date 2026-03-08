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

    # ---- Helpers ----
    items_by_desc = sorted(range(n), key=lambda i: (-w[i], i))

    def score_solution(loads: List[int]) -> int:
        # Secondary objective for tie-breaking: prefer fuller bins.
        # Use sum of squared slack -> smaller is better.
        # Integer weights => integer score.
        slack2 = 0
        for L in loads:
            s = C - L
            slack2 += s * s
        return slack2

    def bfd_construct(order: List[int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        loads: List[int] = []
        for it in order:
            wi = w[it]
            best_j = -1
            best_rem = C + 1
            # Best-fit: minimal remaining capacity after placing.
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

    def cleanup_empty(bins: List[List[int]], loads: List[int]) -> None:
        # Remove empty bins in-place.
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
        # pos_bin[item] = bin index, pos_idx[item] = index inside bin list
        pos_bin = [-1] * n
        pos_idx = [-1] * n
        for b, lst in enumerate(bins):
            for j, it in enumerate(lst):
                pos_bin[it] = b
                pos_idx[it] = j
        return pos_bin, pos_idx

    def clone_solution(bins: List[List[int]], loads: List[int]) -> Tuple[List[List[int]], List[int]]:
        return [lst[:] for lst in bins], loads[:]

    def try_relocate_one(bins: List[List[int]], loads: List[int], pos_bin: List[int], pos_idx: List[int], it: int, to_b: int) -> bool:
        b_from = pos_bin[it]
        if b_from == -1:
            return False
        if to_b == b_from:
            return False
        wi = w[it]
        if to_b == len(bins):
            # create new bin
            bins.append([])
            loads.append(0)
        if loads[to_b] + wi > C:
            # if we created a new bin, remove it
            if to_b == len(bins) - 1 and not bins[to_b]:
                bins.pop()
                loads.pop()
            return False

        # remove from old bin
        idx = pos_idx[it]
        last_it = bins[b_from][-1]
        bins[b_from][idx] = last_it
        pos_idx[last_it] = idx
        bins[b_from].pop()
        loads[b_from] -= wi

        # add to new bin
        pos_bin[it] = to_b
        pos_idx[it] = len(bins[to_b])
        bins[to_b].append(it)
        loads[to_b] += wi
        return True

    def try_swap(bins: List[List[int]], loads: List[int], pos_bin: List[int], pos_idx: List[int], a: int, b: int) -> bool:
        ba = pos_bin[a]
        bb = pos_bin[b]
        if ba == -1 or bb == -1 or ba == bb:
            return False
        wa = w[a]
        wb = w[b]
        # feasibility after swap
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

    def best_fit_reinsert(bins: List[List[int]], loads: List[int], items: List[int]) -> None:
        # Insert items using best-fit (deterministic), creating bins as needed.
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
                bins.append([it])
                loads.append(wi)
            else:
                bins[best_j].append(it)
                loads[best_j] += wi

    # ---- VND (local improvement) ----
    def vnd_improve(bins: List[List[int]], loads: List[int]) -> None:
        # Standard in VNS: apply neighborhoods deterministically until no improvement.
        # We focus on reducing number of bins, then improving fullness.
        improved = True
        while improved:
            improved = False
            cleanup_empty(bins, loads)
            pos_bin, pos_idx = compute_positions(bins)

            # 1) Try to eliminate bins: iterate bins from lightest to heaviest
            order_bins = sorted(range(len(bins)), key=lambda b: loads[b])
            for b in order_bins:
                if time.time() >= deadline:
                    return
                if not bins[b]:
                    continue
                # attempt to move all items out of bin b
                items = bins[b][:]
                # process larger items first for feasibility
                items.sort(key=lambda i: -w[i])

                # snapshot
                saved_bins, saved_loads = clone_solution(bins, loads)
                saved_pos_bin, saved_pos_idx = pos_bin[:], pos_idx[:]

                ok = True
                for it in items:
                    # best-fit among other bins
                    wi = w[it]
                    best_j = -1
                    best_rem = C + 1
                    for j in range(len(loads)):
                        if j == b:
                            continue
                        rem = C - (loads[j] + wi)
                        if rem >= 0 and rem < best_rem:
                            best_rem = rem
                            best_j = j
                    if best_j == -1:
                        ok = False
                        break
                    if not try_relocate_one(bins, loads, pos_bin, pos_idx, it, best_j):
                        ok = False
                        break

                if ok:
                    # bin b should be empty now => improvement
                    cleanup_empty(bins, loads)
                    improved = True
                    break
                else:
                    # rollback
                    bins[:] = saved_bins
                    loads[:] = saved_loads
                    pos_bin[:] = saved_pos_bin
                    pos_idx[:] = saved_pos_idx

            if improved:
                continue

            # 2) If cannot reduce bins, improve score by relocations/swaps
            base_score = score_solution(loads)
            moved = True
            while moved:
                moved = False
                pos_bin, pos_idx = compute_positions(bins)

                # Relocate attempt: sample items in random order
                cand_items = list(range(n))
                random.shuffle(cand_items)
                for it in cand_items:
                    if time.time() >= deadline:
                        return
                    b_from = pos_bin[it]
                    if b_from == -1:
                        continue
                    wi = w[it]
                    # try best-fit target bins (excluding source)
                    targets = list(range(len(bins)))
                    random.shuffle(targets)
                    best_move = None
                    best_new_score = base_score
                    for tb in targets:
                        if tb == b_from:
                            continue
                        if loads[tb] + wi > C:
                            continue
                        # evaluate delta score cheaply by recomputing affected bins
                        old = (C - loads[b_from]) ** 2 + (C - loads[tb]) ** 2
                        newL_from = loads[b_from] - wi
                        newL_to = loads[tb] + wi
                        new = (C - newL_from) ** 2 + (C - newL_to) ** 2
                        new_score = base_score - old + new
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_move = tb
                    if best_move is not None:
                        if try_relocate_one(bins, loads, pos_bin, pos_idx, it, best_move):
                            cleanup_empty(bins, loads)
                            base_score = best_new_score
                            moved = True
                            break

                if moved:
                    continue

                # Swap attempt: try random pairs
                pos_bin, pos_idx = compute_positions(bins)
                pairs = 200  # bounded effort per round
                for _ in range(pairs):
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
                    new_score = base_score - old + new
                    if new_score < base_score:
                        if try_swap(bins, loads, pos_bin, pos_idx, a, b):
                            base_score = new_score
                            moved = True
                            break

    # ---- Shaking neighborhoods ----
    def shake(bins: List[List[int]], loads: List[int], k: int) -> Tuple[List[List[int]], List[int]]:
        sbins, sloads = clone_solution(bins, loads)
        cleanup_empty(sbins, sloads)
        pos_bin, pos_idx = compute_positions(sbins)

        B = len(sbins)
        if B == 0:
            return sbins, sloads

        def random_item_from_bin(bi: int):
            if not sbins[bi]:
                return None
            return random.choice(sbins[bi])

        if k == 1:
            # move 1 random item to a random feasible bin (or new bin sometimes)
            it = random.randrange(n)
            bf = pos_bin[it]
            if bf == -1:
                return sbins, sloads
            targets = list(range(len(sbins) + 1))
            random.shuffle(targets)
            for tb in targets:
                if tb == bf:
                    continue
                if tb == len(sbins) and random.random() < 0.7:
                    continue
                if try_relocate_one(sbins, sloads, pos_bin, pos_idx, it, tb):
                    break
            cleanup_empty(sbins, sloads)
            return sbins, sloads

        if k == 2:
            # swap two random items from different bins
            for _ in range(30):
                a = random.randrange(n)
                b = random.randrange(n)
                if a == b:
                    continue
                if try_swap(sbins, sloads, pos_bin, pos_idx, a, b):
                    break
            cleanup_empty(sbins, sloads)
            return sbins, sloads

        if k == 3:
            # relocate 2-3 items from a random bin to other bins
            b = random.randrange(len(sbins))
            if len(sbins[b]) == 0:
                return sbins, sloads
            t = 2 if len(sbins[b]) < 3 else random.choice([2, 3])
            chosen = random.sample(sbins[b], k=min(t, len(sbins[b])))
            chosen.sort(key=lambda i: -w[i])
            for it in chosen:
                targets = list(range(len(sbins)))
                random.shuffle(targets)
                for tb in targets:
                    if tb == pos_bin[it]:
                        continue
                    if sloads[tb] + w[it] <= C:
                        try_relocate_one(sbins, sloads, pos_bin, pos_idx, it, tb)
                        break
            cleanup_empty(sbins, sloads)
            return sbins, sloads

        if k == 4:
            # merge attempt: pick one of the lightest bins and try to empty it (partial)
            order_bins = sorted(range(len(sbins)), key=lambda b: sloads[b])
            b = order_bins[0]
            items = sbins[b][:]
            items.sort(key=lambda i: -w[i])
            for it in items:
                targets = list(range(len(sbins)))
                random.shuffle(targets)
                for tb in targets:
                    if tb == pos_bin[it] or tb == b:
                        continue
                    if sloads[tb] + w[it] <= C:
                        try_relocate_one(sbins, sloads, pos_bin, pos_idx, it, tb)
                        break
            cleanup_empty(sbins, sloads)
            return sbins, sloads

        # k >= 5: destroy/repair with r items
        r = min(n, 3 + k)  # increasing destruction
        removed: List[int] = []
        # remove items: prefer from many bins
        cand_items = list(range(n))
        random.shuffle(cand_items)
        for it in cand_items:
            if len(removed) >= r:
                break
            bf = pos_bin[it]
            if bf == -1:
                continue
            # remove it
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
        # reinsert removed by best-fit decreasing of removed
        removed.sort(key=lambda i: -w[i])
        best_fit_reinsert(sbins, sloads, removed)
        cleanup_empty(sbins, sloads)
        return sbins, sloads

    # ---- Initial solution ----
    # Slight randomized perturbation of the sorted order to diversify starts.
    order0 = items_by_desc[:]
    # Randomly swap a few adjacent pairs among equal/close weights
    for _ in range(min(20, n // 5 + 1)):
        if time.time() >= deadline:
            break
        i = random.randrange(n - 1) if n > 1 else 0
        if n > 1 and abs(w[order0[i]] - w[order0[i + 1]]) <= 1 and random.random() < 0.5:
            order0[i], order0[i + 1] = order0[i + 1], order0[i]

    best_bins, best_loads = bfd_construct(order0)
    cleanup_empty(best_bins, best_loads)
    vnd_improve(best_bins, best_loads)
    best_num = len(best_bins)
    best_sc = score_solution(best_loads)

    curr_bins, curr_loads = clone_solution(best_bins, best_loads)

    # ---- Main VNS loop ----
    kmax = 6
    # Fixed iteration budget (still time-checked). Scale with n.
    iter_budget = max(200, 40 * kmax + 10 * n)

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

            cand_num = len(cand_bins)
            cand_sc = score_solution(cand_loads)

            curr_num = len(curr_bins)
            curr_sc = score_solution(curr_loads)

            # Improvement criterion
            if (cand_num < curr_num) or (cand_num == curr_num and cand_sc < curr_sc):
                curr_bins, curr_loads = cand_bins, cand_loads
                k = 1
            else:
                k += 1

            # Update best
            if (cand_num < best_num) or (cand_num == best_num and cand_sc < best_sc):
                best_bins, best_loads = clone_solution(cand_bins, cand_loads)
                best_num = cand_num
                best_sc = cand_sc

        # Occasionally restart from best with a small random shake to diversify
        if time.time() >= deadline:
            break
        if iters % 25 == 0:
            curr_bins, curr_loads = clone_solution(best_bins, best_loads)
            # small diversification
            curr_bins, curr_loads = shake(curr_bins, curr_loads, random.choice([2, 3, 5]))
            vnd_improve(curr_bins, curr_loads)

    cleanup_empty(best_bins, best_loads)

    # Ensure feasibility (debug-safety): loads within capacity
    # (No heavy assertions to keep runtime stable.)

    return {"packing": best_bins, "bin_weights": best_loads}
