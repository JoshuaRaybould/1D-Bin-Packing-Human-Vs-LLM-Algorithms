import time
import random
from typing import List, Tuple, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    effective_limit = min(float(time_limit), 100.0)

    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    total_weight = sum(weights)

    # --------------------------
    # Helpers / Objective
    # --------------------------
    def slack_sq(load: int) -> int:
        s = C - load
        return s * s

    def near_full_bonus(load: int) -> int:
        # count bins with small slack (encourage packing bins tightly)
        # returns 1 if slack <= 10% capacity, else 0
        return 1 if (C - load) * 10 <= C else 0

    def objective(nonempty_count: int, slack_sq_sum: int, near_full_cnt: int) -> Tuple[int, int, int, int]:
        # lexicographic:
        # 1) minimize bin count
        # 2) minimize waste (depends on bin count)
        # 3) maximize number of near-full bins (tight packing)
        # 4) maximize slack concentration (sum slack^2)
        waste = nonempty_count * C - total_weight
        return (nonempty_count, waste, -near_full_cnt, -slack_sq_sum)

    # --------------------------
    # Packing primitives (O(1) removes)
    # --------------------------
    def remove_from_bin(bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int],
                        b: int, it: int):
        idx = pos[it]
        last = bins[b][-1]
        bins[b][idx] = last
        pos[last] = idx
        bins[b].pop()
        pos[it] = -1
        assign[it] = -1
        loads[b] -= weights[it]

    def add_to_bin(bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int],
                   b: int, it: int):
        pos[it] = len(bins[b])
        bins[b].append(it)
        assign[it] = b
        loads[b] += weights[it]

    def relocate(bins, loads, assign, pos, it: int, b_to: int):
        b_from = assign[it]
        if b_from == b_to:
            return
        remove_from_bin(bins, loads, assign, pos, b_from, it)
        add_to_bin(bins, loads, assign, pos, b_to, it)

    def swap_items(bins, loads, assign, pos, i1: int, i2: int):
        b1, b2 = assign[i1], assign[i2]
        if b1 == b2:
            return
        remove_from_bin(bins, loads, assign, pos, b1, i1)
        remove_from_bin(bins, loads, assign, pos, b2, i2)
        add_to_bin(bins, loads, assign, pos, b1, i2)
        add_to_bin(bins, loads, assign, pos, b2, i1)

    # --------------------------
    # Rebuild / Compact
    # --------------------------
    def rebuild_compact(bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int]):
        new_bins: List[List[int]] = []
        new_loads: List[int] = []
        mapping = [-1] * len(bins)
        for b, items in enumerate(bins):
            if items:
                mapping[b] = len(new_bins)
                new_bins.append(items[:])
                new_loads.append(loads[b])
        for it in range(n):
            b = assign[it]
            if b >= 0:
                assign[it] = mapping[b]
        pos[:] = [-1] * n
        slack_sum = 0
        near_full = 0
        for b, items in enumerate(new_bins):
            ld = new_loads[b]
            slack_sum += slack_sq(ld)
            near_full += near_full_bonus(ld)
            for idx, it in enumerate(items):
                pos[it] = idx
        return new_bins, new_loads, len(new_bins), slack_sum, near_full

    # --------------------------
    # Views / insertion
    # --------------------------
    def compute_nonempty(bins: List[List[int]]):
        return [b for b in range(len(bins)) if bins[b]]

    def best_fit_bin_for_item(w: int, bins: List[List[int]], loads: List[int], cand_bins: List[int]) -> int:
        best_b = -1
        best_res = None
        for b in cand_bins:
            rem = C - loads[b]
            if rem >= w:
                res = rem - w
                if best_res is None or res < best_res:
                    best_res = res
                    best_b = b
                    if res == 0:
                        break
        return best_b

    def top_bins_by_free_space(bins: List[List[int]], loads: List[int], limit: int = 40) -> List[int]:
        nonempty = compute_nonempty(bins)
        nonempty.sort(key=lambda b: (C - loads[b]), reverse=True)
        return nonempty[: min(limit, len(nonempty))]

    def bins_by_load_asc(bins: List[List[int]], loads: List[int], limit: int = 40) -> List[int]:
        nonempty = compute_nonempty(bins)
        nonempty.sort(key=lambda b: loads[b])
        return nonempty[: min(limit, len(nonempty))]

    # --------------------------
    # Initial construction (multi-start)
    # --------------------------
    def construct_bfd(order: List[int]) -> Tuple[List[List[int]], List[int], List[int], List[int], int, int, int]:
        bins: List[List[int]] = []
        loads: List[int] = []
        assign = [-1] * n
        pos = [-1] * n

        def open_bin(it: int):
            b = len(bins)
            bins.append([it])
            loads.append(weights[it])
            assign[it] = b
            pos[it] = 0

        for it in order:
            w = weights[it]
            if not bins:
                open_bin(it)
                continue
            nonempty = compute_nonempty(bins)
            b_to = best_fit_bin_for_item(w, bins, loads, nonempty)
            if b_to == -1:
                open_bin(it)
            else:
                add_to_bin(bins, loads, assign, pos, b_to, it)

        slack_sum = 0
        near_full = 0
        for ld in loads:
            slack_sum += slack_sq(ld)
            near_full += near_full_bonus(ld)
        return bins, loads, assign, pos, len(bins), slack_sum, near_full

    def harmonic_order() -> List[int]:
        # Put items into coarse size classes to diversify order
        items = list(range(n))
        # classes by fraction of capacity
        def cls(w: int) -> int:
            if w * 2 > C:
                return 0
            if w * 3 > C:
                return 1
            if w * 4 > C:
                return 2
            if w * 5 > C:
                return 3
            return 4
        items.sort(key=lambda i: (cls(weights[i]), -weights[i]))
        # within class random shuffle
        out = []
        i = 0
        while i < n:
            j = i
            ci = cls(weights[items[i]])
            while j < n and cls(weights[items[j]]) == ci:
                j += 1
            block = items[i:j]
            random.shuffle(block)
            out.extend(block)
            i = j
        return out

    def noisy_desc_order(noise: float) -> List[int]:
        # sort by weight desc + small noise to diversify while preserving structure
        items = list(range(n))
        items.sort(key=lambda i: (-weights[i] + noise * random.random(),))
        return items

    init_deadline = start + min(0.08 * effective_limit, 4.0)
    best_bins = best_loads = best_assign = best_pos = None
    best_nonempty = 10**9
    best_slack = 0
    best_near_full = 0
    best_obj = None

    base_desc = list(range(n))
    base_desc.sort(key=lambda i: weights[i], reverse=True)

    cand = 0
    while time.time() < init_deadline and cand < 60:
        if cand % 5 == 0:
            order = base_desc
        elif cand % 5 == 1:
            order = harmonic_order()
        elif cand % 5 == 2:
            order = noisy_desc_order(noise=0.25)
        elif cand % 5 == 3:
            order = noisy_desc_order(noise=0.60)
        else:
            order = base_desc[:]
            # perturb blocks of equal weights
            i = 0
            while i < n:
                j = i + 1
                wi = weights[order[i]]
                while j < n and weights[order[j]] == wi:
                    j += 1
                if j - i > 1:
                    block = order[i:j]
                    random.shuffle(block)
                    order[i:j] = block
                i = j

        bins, loads, assign, pos, nb, sl, nf = construct_bfd(order)
        o = objective(nb, sl, nf)
        if best_obj is None or o < best_obj:
            best_obj = o
            best_bins, best_loads, best_assign, best_pos = bins, loads, assign, pos
            best_nonempty, best_slack, best_near_full = nb, sl, nf
        cand += 1

    # Current starts at best
    bins = [b[:] for b in best_bins]
    loads = best_loads[:]
    assign = best_assign[:]
    pos_in_bin = best_pos[:]
    nonempty_count = best_nonempty
    slack_sq_sum = best_slack
    near_full_cnt = best_near_full

    # --------------------------
    # Incremental objective updates
    # --------------------------
    def update_nonempty_delta(before: bool, after: bool) -> int:
        if before and not after:
            return -1
        if not before and after:
            return 1
        return 0

    def apply_relocate(it: int, b_to: int) -> bool:
        nonlocal nonempty_count, slack_sq_sum, near_full_cnt
        b_from = assign[it]
        if b_from < 0 or b_from == b_to:
            return False
        w = weights[it]
        if loads[b_to] + w > C:
            return False

        before_from = bool(bins[b_from])
        before_to = bool(bins[b_to])
        old_from_ld = loads[b_from]
        old_to_ld = loads[b_to]
        old_sl = (slack_sq(old_from_ld) if before_from else 0) + (slack_sq(old_to_ld) if before_to else 0)
        old_nf = (near_full_bonus(old_from_ld) if before_from else 0) + (near_full_bonus(old_to_ld) if before_to else 0)

        relocate(bins, loads, assign, pos_in_bin, it, b_to)

        after_from = bool(bins[b_from])
        after_to = bool(bins[b_to])
        nonempty_count += update_nonempty_delta(before_from, after_from)
        nonempty_count += update_nonempty_delta(before_to, after_to)

        new_from_ld = loads[b_from]
        new_to_ld = loads[b_to]
        new_sl = (slack_sq(new_from_ld) if after_from else 0) + (slack_sq(new_to_ld) if after_to else 0)
        new_nf = (near_full_bonus(new_from_ld) if after_from else 0) + (near_full_bonus(new_to_ld) if after_to else 0)

        slack_sq_sum += new_sl - old_sl
        near_full_cnt += new_nf - old_nf
        return True

    def undo_relocate(it: int, b_from: int):
        # relocate current item back to b_from
        nonlocal nonempty_count, slack_sq_sum, near_full_cnt
        b_to = assign[it]
        if b_to == b_from:
            return

        before_from = bool(bins[b_from])
        before_to = bool(bins[b_to])
        old_from_ld = loads[b_from]
        old_to_ld = loads[b_to]
        old_sl = (slack_sq(old_from_ld) if before_from else 0) + (slack_sq(old_to_ld) if before_to else 0)
        old_nf = (near_full_bonus(old_from_ld) if before_from else 0) + (near_full_bonus(old_to_ld) if before_to else 0)

        relocate(bins, loads, assign, pos_in_bin, it, b_from)

        after_from = bool(bins[b_from])
        after_to = bool(bins[b_to])
        nonempty_count += update_nonempty_delta(before_from, after_from)
        nonempty_count += update_nonempty_delta(before_to, after_to)

        new_from_ld = loads[b_from]
        new_to_ld = loads[b_to]
        new_sl = (slack_sq(new_from_ld) if after_from else 0) + (slack_sq(new_to_ld) if after_to else 0)
        new_nf = (near_full_bonus(new_from_ld) if after_from else 0) + (near_full_bonus(new_to_ld) if after_to else 0)

        slack_sq_sum += new_sl - old_sl
        near_full_cnt += new_nf - old_nf

    def apply_swap(i1: int, i2: int) -> bool:
        nonlocal slack_sq_sum, near_full_cnt
        b1, b2 = assign[i1], assign[i2]
        if b1 < 0 or b2 < 0 or b1 == b2:
            return False
        w1, w2 = weights[i1], weights[i2]
        if loads[b1] - w1 + w2 > C:
            return False
        if loads[b2] - w2 + w1 > C:
            return False

        old1, old2 = loads[b1], loads[b2]
        old_sl = slack_sq(old1) + slack_sq(old2)
        old_nf = near_full_bonus(old1) + near_full_bonus(old2)

        swap_items(bins, loads, assign, pos_in_bin, i1, i2)

        new1, new2 = loads[b1], loads[b2]
        new_sl = slack_sq(new1) + slack_sq(new2)
        new_nf = near_full_bonus(new1) + near_full_bonus(new2)

        slack_sq_sum += new_sl - old_sl
        near_full_cnt += new_nf - old_nf
        return True

    def maybe_compact(force: bool = False):
        nonlocal bins, loads, assign, pos_in_bin, nonempty_count, slack_sq_sum, near_full_cnt
        if force:
            bins, loads, nonempty_count, slack_sq_sum, near_full_cnt = rebuild_compact(
                bins, loads, assign, pos_in_bin
            )
            return
        empties = len(bins) - nonempty_count
        if len(bins) >= 25 and empties > max(4, len(bins) // 8):
            bins, loads, nonempty_count, slack_sq_sum, near_full_cnt = rebuild_compact(
                bins, loads, assign, pos_in_bin
            )

    # --------------------------
    # Neighborhood: strong bin elimination with recursive placement
    # --------------------------
    def try_eliminate_bin_strong(b_from: int, time_ctr: List[int]) -> bool:
        if b_from < 0 or b_from >= len(bins) or not bins[b_from]:
            return False

        items = bins[b_from][:]
        items.sort(key=lambda it: weights[it], reverse=True)

        # Destination candidates: prioritize bins with larger slack and also near-full (for exact fits)
        nonempty = compute_nonempty(bins)
        cand = [b for b in nonempty if b != b_from]
        if not cand:
            return False
        cand.sort(key=lambda b: (C - loads[b], loads[b]), reverse=True)

        # recursion tries top choices per item
        # Depth-control: larger for small |items|, smaller otherwise
        if len(items) <= 4:
            per_item_choices = 10
        elif len(items) <= 7:
            per_item_choices = 7
        else:
            per_item_choices = 5

        moved: List[Tuple[int, int]] = []

        def rollback():
            for it, ob in reversed(moved):
                if assign[it] != ob:
                    undo_relocate(it, ob)
            moved.clear()

        def choices_for(it: int) -> List[int]:
            w = weights[it]
            # collect best-fit among top scanK by slack
            scanK = min(len(cand), 50)
            best: List[Tuple[int, int]] = []  # (residual, b)
            for b in cand[:scanK]:
                if not bins[b]:
                    continue
                rem = C - loads[b]
                if rem >= w:
                    res = rem - w
                    best.append((res, b))
            if not best:
                return []
            best.sort()
            return [b for _, b in best[:per_item_choices]]

        # Heuristic ordering: hardest-first by number of choices, recomputed once
        item_choices = [(it, choices_for(it)) for it in items]
        if any(len(ch) == 0 for _, ch in item_choices):
            return False
        item_choices.sort(key=lambda x: (len(x[1]), -weights[x[0]]))
        ordered_items = [it for it, _ in item_choices]
        ordered_choices = {it: ch for it, ch in item_choices}

        def dfs(idx: int) -> bool:
            if idx == len(ordered_items):
                return True
            if time_ctr[0] % 400 == 0 and time.time() - start > effective_limit:
                return False
            time_ctr[0] += 1

            it = ordered_items[idx]
            ob = assign[it]
            if ob != b_from:
                # already moved by earlier recursion (shouldn't happen)
                return dfs(idx + 1)

            ch = ordered_choices[it]
            # small randomization among best few to escape deterministic traps
            if len(ch) > 2:
                first = ch[:2]
                rest = ch[2:]
                random.shuffle(rest)
                ch_try = first + rest
            else:
                ch_try = ch

            for b_to in ch_try:
                if loads[b_to] + weights[it] > C:
                    continue
                if apply_relocate(it, b_to):
                    moved.append((it, b_from))
                    if dfs(idx + 1):
                        return True
                    # undo
                    moved.pop()
                    undo_relocate(it, b_from)
            return False

        ok = dfs(0)
        if not ok:
            rollback()
            return False

        # success: bin should be empty
        if bins[b_from]:
            rollback()
            return False
        return True

    # --------------------------
    # Neighborhood: merge two bins (try to remove one)
    # --------------------------
    def try_merge_two_bins(time_ctr: List[int]) -> bool:
        # pick two small bins, remove their items, try to reinsert to eliminate at least one bin
        small = bins_by_load_asc(bins, loads, limit=18)
        if len(small) < 2:
            return False

        # candidate pairs: few best combinations
        pairs = []
        for i in range(min(8, len(small))):
            for j in range(i + 1, min(10, len(small))):
                a, b = small[i], small[j]
                if not bins[a] or not bins[b]:
                    continue
                pairs.append((loads[a] + loads[b], a, b))
        if not pairs:
            return False
        pairs.sort()  # smallest total first
        pairs = pairs[:10]

        for _, a, b in pairs:
            if time_ctr[0] % 300 == 0 and time.time() - start > effective_limit:
                return False
            time_ctr[0] += 1

            if not bins[a] or not bins[b] or a == b:
                continue

            removed = bins[a][:] + bins[b][:]
            removed.sort(key=lambda it: weights[it], reverse=True)

            # snapshot moves for rollback
            moved: List[Tuple[int, int]] = []

            def rollback():
                for it, ob in reversed(moved):
                    if assign[it] != ob:
                        undo_relocate(it, ob)
                moved.clear()

            # First, pull all items into a (temporarily) then empty both by moving out
            # We'll simply remove them (relocate to a temporary new bin) by opening a scratch bin.
            scratch = len(bins)
            bins.append([])
            loads.append(0)

            # remove all items from a and b into scratch
            for it in removed:
                ob = assign[it]
                if ob not in (a, b):
                    continue
                # scratch always feasible
                apply_relocate(it, scratch)
                moved.append((it, ob))

            maybe_compact(force=False)

            # attempt to reinsert without using both a and b; we allow using existing bins including a/b if present.
            for it in removed:
                if time.time() - start > effective_limit:
                    rollback()
                    return False
                w = weights[it]
                nonempty = compute_nonempty(bins)
                # exclude scratch from destinations (we'll want it empty)
                cand = [bb for bb in nonempty if bb != scratch]
                # prefer best-fit
                b_to = best_fit_bin_for_item(w, bins, loads, cand)
                if b_to == -1:
                    # open new bin (bad for goal). fail.
                    rollback()
                    return False
                ob = assign[it]
                if not apply_relocate(it, b_to):
                    rollback()
                    return False
                moved.append((it, ob))

            # if we managed to reinsert all, scratch should be empty and at least one of (a,b) can be empty now
            if bins[scratch]:
                rollback()
                return False

            # success if we reduced nonempty count (usually by emptying a/b/scratch)
            # scratch is empty: can be compacted away
            maybe_compact(force=True)
            return True

        return False

    # --------------------------
    # Neighborhood: 2-1 exchange (kept, tightened)
    # --------------------------
    def bin_item_candidates(b: int, t: int = 8) -> List[int]:
        items = bins[b]
        if len(items) <= 2 * t:
            return items[:]
        srt = sorted(items, key=lambda it: weights[it])
        return srt[:t] + srt[-t:]

    def try_exchange_2_1(time_ctr: List[int]) -> bool:
        nonempty = compute_nonempty(bins)
        if len(nonempty) < 2:
            return False
        donors = bins_by_load_asc(bins, loads, limit=14)
        receivers = top_bins_by_free_space(bins, loads, limit=22)

        for a in donors:
            if not bins[a] or len(bins[a]) < 2:
                continue
            cand_a = bin_item_candidates(a, t=7)
            # choose pairs biased to heavy+light combos
            cand_a.sort(key=lambda it: weights[it], reverse=True)
            for i in range(min(8, len(cand_a))):
                for j in range(i + 1, min(10, len(cand_a))):
                    it1, it2 = cand_a[i], cand_a[j]
                    if assign[it1] != a or assign[it2] != a:
                        continue
                    w12 = weights[it1] + weights[it2]
                    for b in receivers:
                        if b == a or not bins[b]:
                            continue
                        if time_ctr[0] % 350 == 0 and time.time() - start > effective_limit:
                            return False
                        time_ctr[0] += 1

                        # pick a few candidates from b
                        cand_b = bin_item_candidates(b, t=7)
                        # try to swap out an item k
                        for k in cand_b:
                            if assign[k] != b:
                                continue
                            if loads[b] - weights[k] + w12 > C:
                                continue
                            if loads[a] - w12 + weights[k] > C:
                                continue

                            # execute with rollback
                            if not apply_relocate(k, a):
                                continue
                            if not apply_relocate(it1, b):
                                undo_relocate(k, b)
                                continue
                            if not apply_relocate(it2, b):
                                undo_relocate(it1, a)
                                undo_relocate(k, b)
                                continue
                            return True
        return False

    # --------------------------
    # Neighborhood: small reloc/swap finishing
    # --------------------------
    def try_simple_moves(time_ctr: List[int]) -> bool:
        nonempty = compute_nonempty(bins)
        if len(nonempty) <= 1:
            return False
        donors = bins_by_load_asc(bins, loads, limit=16)
        receivers = top_bins_by_free_space(bins, loads, limit=26)

        # relocations
        for a in donors:
            if not bins[a]:
                continue
            items = bin_item_candidates(a, t=8)
            items.sort(key=lambda it: weights[it], reverse=True)
            for it in items:
                if assign[it] != a:
                    continue
                w = weights[it]
                cand = [b for b in receivers if b != a and bins[b]]
                b_to = best_fit_bin_for_item(w, bins, loads, cand)
                if b_to != -1:
                    if time_ctr[0] % 500 == 0 and time.time() - start > effective_limit:
                        return False
                    time_ctr[0] += 1
                    if apply_relocate(it, b_to):
                        return True

        # random feasible swaps
        for _ in range(120):
            if time_ctr[0] % 500 == 0 and time.time() - start > effective_limit:
                return False
            time_ctr[0] += 1
            i1 = random.randrange(n)
            i2 = random.randrange(n)
            if i1 != i2 and apply_swap(i1, i2):
                return True
        return False

    # --------------------------
    # VND (intensification)
    # --------------------------
    def vnd(move_budget: int):
        nonlocal best_obj, best_bins, best_loads, best_assign, best_pos
        nonlocal best_nonempty, best_slack, best_near_full

        time_ctr = [0]
        moves = 0
        while moves < move_budget:
            if moves % 600 == 0 and time.time() - start > effective_limit:
                return

            cur_o = objective(nonempty_count, slack_sq_sum, near_full_cnt)
            if cur_o < best_obj:
                best_obj = cur_o
                best_bins = [b[:] for b in bins]
                best_loads = loads[:]
                best_assign = assign[:]
                best_pos = pos_in_bin[:]
                best_nonempty = nonempty_count
                best_slack = slack_sq_sum
                best_near_full = near_full_cnt

            improved = False

            # 1) eliminate a few smallest bins
            targets = bins_by_load_asc(bins, loads, limit=14)
            for b_from in targets[:8]:
                if time.time() - start > effective_limit:
                    return
                pre_o = objective(nonempty_count, slack_sq_sum, near_full_cnt)
                if try_eliminate_bin_strong(b_from, time_ctr):
                    maybe_compact(force=True)
                    post_o = objective(nonempty_count, slack_sq_sum, near_full_cnt)
                    if post_o < pre_o:
                        improved = True
                        break
            if improved:
                moves += 1
                continue

            # 2) merge two bins
            pre_o = objective(nonempty_count, slack_sq_sum, near_full_cnt)
            if try_merge_two_bins(time_ctr):
                maybe_compact(force=False)
                if objective(nonempty_count, slack_sq_sum, near_full_cnt) <= pre_o:
                    improved = True
            if improved:
                moves += 1
                continue

            # 3) 2-1 exchange
            pre_o = objective(nonempty_count, slack_sq_sum, near_full_cnt)
            if try_exchange_2_1(time_ctr):
                if objective(nonempty_count, slack_sq_sum, near_full_cnt) <= pre_o:
                    improved = True
            if improved:
                moves += 1
                continue

            # 4) simple moves
            pre_o = objective(nonempty_count, slack_sq_sum, near_full_cnt)
            if try_simple_moves(time_ctr):
                if objective(nonempty_count, slack_sq_sum, near_full_cnt) <= pre_o:
                    improved = True

            if not improved:
                return

            moves += 1
            if moves % 25 == 0:
                maybe_compact(force=False)

    # --------------------------
    # Shaking (ruin & recreate) - VNS component
    # --------------------------
    def ruin_and_recreate(k: int):
        nonlocal nonempty_count, slack_sq_sum, near_full_cnt

        nonempty = compute_nonempty(bins)
        if not nonempty:
            return

        # adaptive removal size
        r = min(n, 10 + 6 * k)

        removed: List[int] = []

        # strategy A: remove whole bins (smallest bins)
        if random.random() < 0.55:
            small_bins = bins_by_load_asc(bins, loads, limit=6)
            for b in small_bins:
                if len(removed) >= r:
                    break
                if not bins[b]:
                    continue
                take = bins[b][:]
                random.shuffle(take)
                removed.extend(take)

        # strategy B: remove items from bins with high slack (to re-pack better)
        if len(removed) < r and random.random() < 0.65:
            slack_bins = top_bins_by_free_space(bins, loads, limit=10)
            for b in slack_bins:
                if len(removed) >= r:
                    break
                if not bins[b]:
                    continue
                items = bins[b][:]
                random.shuffle(items)
                removed.extend(items[: min(len(items), max(1, (r - len(removed)) // 2))])

        # strategy C: random fill
        while len(removed) < r:
            it = random.randrange(n)
            if assign[it] >= 0:
                removed.append(it)

        # unique
        seen = set()
        uniq = []
        for it in removed:
            if it not in seen and assign[it] >= 0:
                seen.add(it)
                uniq.append(it)
        removed = uniq

        # remove
        for it in removed:
            b = assign[it]
            if b < 0:
                continue
            before = bool(bins[b])
            old_ld = loads[b]
            old_sl = slack_sq(old_ld) if before else 0
            old_nf = near_full_bonus(old_ld) if before else 0
            remove_from_bin(bins, loads, assign, pos_in_bin, b, it)
            after = bool(bins[b])
            nonempty_count += update_nonempty_delta(before, after)
            new_sl = slack_sq(loads[b]) if after else 0
            new_nf = near_full_bonus(loads[b]) if after else 0
            slack_sq_sum += new_sl - old_sl
            near_full_cnt += new_nf - old_nf

        maybe_compact(force=False)

        # reinsert: best-fit with limited randomized candidate ordering
        removed.sort(key=lambda it: weights[it], reverse=True)
        for it in removed:
            if time.time() - start > effective_limit:
                return
            w = weights[it]
            nonempty = compute_nonempty(bins)
            # candidate list: prefer near-feasible bins
            nonempty.sort(key=lambda b: (C - loads[b] - w >= 0, -(C - loads[b])), reverse=True)
            b_to = best_fit_bin_for_item(w, bins, loads, nonempty)
            if b_to == -1:
                b_new = len(bins)
                bins.append([])
                loads.append(0)
                add_to_bin(bins, loads, assign, pos_in_bin, b_new, it)
                nonempty_count += 1
                slack_sq_sum += slack_sq(loads[b_new])
                near_full_cnt += near_full_bonus(loads[b_new])
            else:
                old_ld = loads[b_to]
                old_sl = slack_sq(old_ld)
                old_nf = near_full_bonus(old_ld)
                add_to_bin(bins, loads, assign, pos_in_bin, b_to, it)
                new_ld = loads[b_to]
                slack_sq_sum += slack_sq(new_ld) - old_sl
                near_full_cnt += near_full_bonus(new_ld) - old_nf

        maybe_compact(force=False)

    # --------------------------
    # Main VNS loop
    # --------------------------
    # Intensify initial
    vnd(move_budget=5000)

    # Ensure best snapshot exists
    best_obj = objective(best_nonempty, best_slack, best_near_full)

    k_max = 24
    MAX_ITERS = 2600  # fixed iterations as required

    stall = 0
    best_bins_seen = best_obj[0]

    for it in range(MAX_ITERS):
        if it % 40 == 0 and time.time() - start > effective_limit:
            break

        if best_obj[0] < best_bins_seen:
            best_bins_seen = best_obj[0]
            stall = 0
        else:
            stall += 1

        # periodic strong restart from best
        if stall >= 180:
            bins = [b[:] for b in best_bins]
            loads = best_loads[:]
            assign = best_assign[:]
            pos_in_bin = best_pos[:]
            nonempty_count = best_nonempty
            slack_sq_sum = best_slack
            near_full_cnt = best_near_full
            # strong shake
            ruin_and_recreate(k=12)
            vnd(move_budget=4200)
            stall = 0
            continue

        k = 1
        while k <= k_max:
            if time.time() - start > effective_limit:
                break

            # start from best
            bins = [b[:] for b in best_bins]
            loads = best_loads[:]
            assign = best_assign[:]
            pos_in_bin = best_pos[:]
            nonempty_count = best_nonempty
            slack_sq_sum = best_slack
            near_full_cnt = best_near_full

            # shaking intensity depends on k
            ruin_and_recreate(k)

            # intensification budget scales mildly with k
            vnd(move_budget=2600 + 40 * k)

            cur_o = objective(nonempty_count, slack_sq_sum, near_full_cnt)
            if cur_o < best_obj:
                best_obj = cur_o
                best_bins = [b[:] for b in bins]
                best_loads = loads[:]
                best_assign = assign[:]
                best_pos = pos_in_bin[:]
                best_nonempty = nonempty_count
                best_slack = slack_sq_sum
                best_near_full = near_full_cnt
                k = 1
            else:
                k += 1

    # Final: return compact best
    bins = [b[:] for b in best_bins]
    loads = best_loads[:]
    assign = best_assign[:]
    pos_in_bin = best_pos[:]
    bins, loads, _, _, _ = rebuild_compact(bins, loads, assign, pos_in_bin)
    return {"packing": bins, "bin_weights": loads}
