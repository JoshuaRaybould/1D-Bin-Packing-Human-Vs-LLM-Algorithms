import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    effective_limit = min(float(time_limit), 100.0)

    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)
    total_weight = sum(weights)

    # --------------------------
    # Data structures
    # --------------------------
    # bins: List[List[int]] items per bin id
    # loads: List[int] load per bin id
    # assign[item] -> bin id
    # pos_in_bin[item] -> index within bins[assign[item]]
    # nonempty_count maintained incrementally
    # slack_sq_sum = sum((C-load)^2) over non-empty bins

    def slack_sq(load: int) -> int:
        fs = C - load
        return fs * fs

    def objective(nonempty_count: int, slack_sq_sum: int) -> Tuple[int, int, int]:
        # (1) minimize #bins
        # (2) minimize waste = bins*C - total_weight (depends only on #bins)
        # (3) maximize slack concentration => maximize sum(fs^2)
        #   in tuple minimized: use -slack_sq_sum
        waste = nonempty_count * C - total_weight
        return (nonempty_count, waste, -slack_sq_sum)

    # --------------------------
    # Packing primitives (O(1) removes)
    # --------------------------
    def remove_from_bin(bins: List[List[int]], loads: List[int], assign: List[int], pos: List[int],
                        b: int, it: int):
        """Swap-remove item it from bins[b]."""
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
        # Remove both
        remove_from_bin(bins, loads, assign, pos, b1, i1)
        remove_from_bin(bins, loads, assign, pos, b2, i2)
        # Add swapped
        add_to_bin(bins, loads, assign, pos, b1, i2)
        add_to_bin(bins, loads, assign, pos, b2, i1)

    # --------------------------
    # Compaction / rebuild
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
        # rebuild assign, pos
        for it in range(n):
            b = assign[it]
            if b >= 0:
                assign[it] = mapping[b]
        pos[:] = [-1] * n
        for b, items in enumerate(new_bins):
            for idx, it in enumerate(items):
                pos[it] = idx
        # compute slack sum
        slack_sum = 0
        for ld in new_loads:
            slack_sum += slack_sq(ld)
        return new_bins, new_loads, len(new_bins), slack_sum

    # --------------------------
    # Best-fit insertion helpers
    # --------------------------
    def compute_bin_views(bins: List[List[int]], loads: List[int]):
        """Return lists of non-empty bins sorted by free space desc and by load asc."""
        nonempty = [b for b in range(len(bins)) if bins[b]]
        # sort by free space descending (largest slack first)
        by_fs = sorted(nonempty, key=lambda b: (C - loads[b]), reverse=True)
        # sort by load ascending (small bins first)
        by_load = sorted(nonempty, key=lambda b: loads[b])
        return nonempty, by_fs, by_load

    def best_fit_bin_for_item(w: int, bins: List[List[int]], loads: List[int], cand_bins: List[int]):
        best_b = -1
        best_res = None
        for b in cand_bins:
            rem = C - loads[b]
            if rem >= w:
                res = rem - w
                if best_res is None or res < best_res:
                    best_res = res
                    best_b = b
                    if best_res == 0:
                        break
        return best_b

    # --------------------------
    # Initial solutions (multi-start)
    # --------------------------
    def construct_solution(variant: str):
        items = list(range(n))
        items.sort(key=lambda i: weights[i], reverse=True)

        # randomized tie-breaking within equal weights
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

        bins: List[List[int]] = []
        loads: List[int] = []
        assign = [-1] * n
        pos = [-1] * n

        def open_bin_with(it: int):
            b = len(bins)
            bins.append([it])
            loads.append(weights[it])
            assign[it] = b
            pos[it] = 0

        for it in items:
            w = weights[it]
            if not bins:
                open_bin_with(it)
                continue

            nonempty = [b for b in range(len(bins)) if bins[b]]

            chosen = -1
            if variant == "FFD":
                # first fit
                for b in nonempty:
                    if loads[b] + w <= C:
                        chosen = b
                        break
            elif variant == "BFD":
                # best fit
                best_res = None
                for b in nonempty:
                    rem = C - loads[b]
                    if rem >= w:
                        res = rem - w
                        if best_res is None or res < best_res:
                            best_res = res
                            chosen = b
                            if res == 0:
                                break
            else:  # "AWF" almost-worst-fit (diversification)
                # among feasible, pick largest remaining capacity (worst-fit)
                best_rem = None
                for b in nonempty:
                    rem = C - loads[b]
                    if rem >= w:
                        if best_rem is None or rem > best_rem:
                            best_rem = rem
                            chosen = b

            if chosen == -1:
                open_bin_with(it)
            else:
                add_to_bin(bins, loads, assign, pos, chosen, it)

        slack_sum = 0
        for ld in loads:
            slack_sum += slack_sq(ld)
        return bins, loads, assign, pos, len(bins), slack_sum

    # initial construction budget
    init_deadline = start + min(0.05 * effective_limit, 2.0)
    variants = ["BFD", "FFD", "AWF"]
    best_bins = best_loads = best_assign = best_pos = None
    best_nonempty = 0
    best_slack = 0
    best_obj = None

    # build multiple candidates
    cand_count = 0
    while time.time() < init_deadline and cand_count < 30:
        v = variants[cand_count % len(variants)]
        bins, loads, assign, pos, nb, slack_sum = construct_solution(v)
        o = objective(nb, slack_sum)
        if best_obj is None or o < best_obj:
            best_obj = o
            best_bins, best_loads, best_assign, best_pos = bins, loads, assign, pos
            best_nonempty, best_slack = nb, slack_sum
        cand_count += 1

    # current state starts from best initial
    bins = [b[:] for b in best_bins]
    loads = best_loads[:]
    assign = best_assign[:]
    pos_in_bin = best_pos[:]
    nonempty_count = best_nonempty
    slack_sq_sum = best_slack

    # counters for rebuild and time checks
    accepted_moves = 0
    last_improve_iter = 0

    def update_bin_nonempty_before_after(b: int, before_nonempty: bool, after_nonempty: bool):
        nonlocal nonempty_count
        if before_nonempty and not after_nonempty:
            nonempty_count -= 1
        elif not before_nonempty and after_nonempty:
            nonempty_count += 1

    def try_relocate_move(it: int, b_to: int):
        nonlocal slack_sq_sum, accepted_moves
        b_from = assign[it]
        if b_from == b_to or b_from < 0:
            return False
        w = weights[it]
        if loads[b_to] + w > C:
            return False

        before_from_nonempty = bool(bins[b_from])
        before_to_nonempty = bool(bins[b_to])

        old_from = loads[b_from]
        old_to = loads[b_to]
        old_slack_from = slack_sq(old_from) if before_from_nonempty else 0
        old_slack_to = slack_sq(old_to) if before_to_nonempty else 0

        relocate(bins, loads, assign, pos_in_bin, it, b_to)

        after_from_nonempty = bool(bins[b_from])
        after_to_nonempty = bool(bins[b_to])

        update_bin_nonempty_before_after(b_from, before_from_nonempty, after_from_nonempty)
        update_bin_nonempty_before_after(b_to, before_to_nonempty, after_to_nonempty)

        new_slack_from = slack_sq(loads[b_from]) if after_from_nonempty else 0
        new_slack_to = slack_sq(loads[b_to]) if after_to_nonempty else 0

        slack_sq_sum += (new_slack_from + new_slack_to - old_slack_from - old_slack_to)
        accepted_moves += 1
        return True

    def undo_relocate_move(it: int, b_from: int):
        # relocate it back to b_from
        nonlocal slack_sq_sum, accepted_moves
        b_to = assign[it]
        if b_to == b_from:
            return

        before_from_nonempty = bool(bins[b_from])
        before_to_nonempty = bool(bins[b_to])

        old_from = loads[b_from]
        old_to = loads[b_to]
        old_slack_from = slack_sq(old_from) if before_from_nonempty else 0
        old_slack_to = slack_sq(old_to) if before_to_nonempty else 0

        relocate(bins, loads, assign, pos_in_bin, it, b_from)

        after_from_nonempty = bool(bins[b_from])
        after_to_nonempty = bool(bins[b_to])

        update_bin_nonempty_before_after(b_from, before_from_nonempty, after_from_nonempty)
        update_bin_nonempty_before_after(b_to, before_to_nonempty, after_to_nonempty)

        new_slack_from = slack_sq(loads[b_from]) if after_from_nonempty else 0
        new_slack_to = slack_sq(loads[b_to]) if after_to_nonempty else 0

        slack_sq_sum += (new_slack_from + new_slack_to - old_slack_from - old_slack_to)
        accepted_moves += 1

    def try_swap_move(i1: int, i2: int):
        nonlocal slack_sq_sum, accepted_moves
        b1, b2 = assign[i1], assign[i2]
        if b1 < 0 or b2 < 0 or b1 == b2:
            return False
        w1, w2 = weights[i1], weights[i2]
        if loads[b1] - w1 + w2 > C:
            return False
        if loads[b2] - w2 + w1 > C:
            return False

        before_b1_nonempty = bool(bins[b1])
        before_b2_nonempty = bool(bins[b2])
        old1 = loads[b1]
        old2 = loads[b2]
        old_slack1 = slack_sq(old1) if before_b1_nonempty else 0
        old_slack2 = slack_sq(old2) if before_b2_nonempty else 0

        swap_items(bins, loads, assign, pos_in_bin, i1, i2)

        # bins remain non-empty in swap if they were (sizes unchanged)
        new_slack1 = slack_sq(loads[b1]) if before_b1_nonempty else 0
        new_slack2 = slack_sq(loads[b2]) if before_b2_nonempty else 0
        slack_sq_sum += (new_slack1 + new_slack2 - old_slack1 - old_slack2)
        accepted_moves += 1
        return True

    def maybe_compact(force: bool = False):
        nonlocal bins, loads, assign, pos_in_bin, nonempty_count, slack_sq_sum
        if force:
            bins, loads, nonempty_count, slack_sq_sum = rebuild_compact(bins, loads, assign, pos_in_bin)
            return
        # trigger if too many empties
        empties = len(bins) - nonempty_count
        if len(bins) >= 20 and empties > max(3, len(bins) // 10):
            bins, loads, nonempty_count, slack_sq_sum = rebuild_compact(bins, loads, assign, pos_in_bin)

    # --------------------------
    # Candidate selection utilities
    # --------------------------
    def pick_target_bins_for_elimination(limit: int = 20):
        nonempty, by_fs, by_load = compute_bin_views(bins, loads)
        # prioritize smallest-load bins (easier to eliminate), plus a few random
        targets = by_load[: min(limit, len(by_load))]
        if len(nonempty) > 0:
            for _ in range(min(3, len(nonempty))):
                targets.append(random.choice(nonempty))
        # unique while preserving order
        seen = set()
        out = []
        for b in targets:
            if b not in seen and bins[b]:
                seen.add(b)
                out.append(b)
        return out, by_fs

    def bin_item_candidates(b: int, t: int = 8):
        items = bins[b]
        if len(items) <= 2 * t:
            return items[:]
        # heaviest t + lightest t
        sorted_items = sorted(items, key=lambda it: weights[it])
        return sorted_items[:t] + sorted_items[-t:]

    # --------------------------
    # Neighborhood 1: Bin elimination (destroy/repair with limited backtracking)
    # --------------------------
    def try_eliminate_bin(b_from: int, by_fs_bins: List[int], time_check_counter: List[int]) -> bool:
        """Attempt to move all items from b_from into other bins; commit only if succeeds."""
        if not bins[b_from]:
            return False
        items = bins[b_from][:]
        items.sort(key=lambda it: weights[it], reverse=True)

        # Candidate destination bins: largest free space first, excluding b_from
        cand_bins = [b for b in by_fs_bins if b != b_from and bins[b]]
        if not cand_bins:
            return False

        moved: List[Tuple[int, int]] = []  # (item, original_bin=b_from) for rollback order

        def place_item_with_alternatives(it: int, depth: int) -> bool:
            w = weights[it]
            # consider top R best-fit among bins with enough free space
            # to avoid sorting, scan first K bins in by_fs list and track best residuals
            R = 8
            bests: List[Tuple[int, int]] = []  # (residual, bin)
            scanK = min(len(cand_bins), 40)
            for b in cand_bins[:scanK]:
                rem = C - loads[b]
                if rem >= w:
                    res = rem - w
                    if len(bests) < R:
                        bests.append((res, b))
                        if len(bests) == R:
                            bests.sort()
                    else:
                        if res < bests[-1][0]:
                            bests[-1] = (res, b)
                            bests.sort()
                    if res == 0:
                        break

            if not bests:
                return False

            # depth-1 greedy: try first; depth>1: try a few alternatives
            tries = bests if depth > 0 else bests[:1]
            for _, b_to in tries:
                if time_check_counter[0] % 200 == 0 and time.time() - start > effective_limit:
                    return False
                time_check_counter[0] += 1

                if try_relocate_move(it, b_to):
                    moved.append((it, b_from))
                    return True
            return False

        # greedy first pass
        for it in items:
            if time_check_counter[0] % 200 == 0 and time.time() - start > effective_limit:
                # rollback
                for rit, ob in reversed(moved):
                    undo_relocate_move(rit, ob)
                return False
            time_check_counter[0] += 1

            if not place_item_with_alternatives(it, depth=0):
                # limited backtracking: for this item, rollback a few last moves and retry with alternatives
                success = False
                backtrack_steps = min(3, len(moved))
                # try rolling back 0..backtrack_steps moves and attempt with deeper alternatives
                for bt in range(0, backtrack_steps + 1):
                    # rollback bt last moves
                    rolled = []
                    for _ in range(bt):
                        rit, ob = moved.pop()
                        undo_relocate_move(rit, ob)
                        rolled.append((rit, ob))

                    if place_item_with_alternatives(it, depth=1):
                        # now re-place rolled items greedily
                        ok = True
                        for rit, _ in rolled:
                            if not place_item_with_alternatives(rit, depth=0):
                                ok = False
                                break
                        if ok:
                            success = True
                            break

                    # rollback attempt (including it if moved)
                    # ensure item it is back in b_from
                    if assign[it] != b_from:
                        undo_relocate_move(it, b_from)
                        if moved and moved[-1][0] == it:
                            moved.pop()
                    for rit, ob in reversed(rolled):
                        # put them back into b_from (original)
                        if assign[rit] != ob:
                            undo_relocate_move(rit, ob)
                    # restore moved list by re-adding rolled? not needed since we're fully rolled back
                    # rebuild moved list to what it was before this bt
                    # easiest: recompute moved by scanning items already placed is expensive; instead, we
                    # accept that we rolled everything back. To keep it simple, we break if not success.
                    moved.clear()

                    # re-apply earlier items greedily up to current it index
                    for prev in items:
                        if prev == it:
                            break
                        if assign[prev] != b_from:
                            continue
                        if not place_item_with_alternatives(prev, depth=0):
                            success = False
                            break
                    if success:
                        break

                if not success:
                    # rollback everything
                    for rit, ob in reversed(moved):
                        undo_relocate_move(rit, ob)
                    return False

        # success: bin should be empty now
        if bins[b_from]:
            # rollback if something inconsistent
            for rit, ob in reversed(moved):
                undo_relocate_move(rit, ob)
            return False

        return True

    # --------------------------
    # Neighborhood 2: 2-1 exchange
    # --------------------------
    def try_exchange_2_1(time_check_counter: List[int]) -> bool:
        nonempty, by_fs, by_load = compute_bin_views(bins, loads)
        if len(nonempty) < 2:
            return False

        donors = by_load[: min(12, len(by_load))]
        receivers = by_fs[: min(18, len(by_fs))]

        for a in donors:
            if not bins[a] or len(bins[a]) < 2:
                continue
            cand_a = bin_item_candidates(a, t=7)
            # pairs from cand_a
            for i in range(len(cand_a)):
                for j in range(i + 1, len(cand_a)):
                    it1, it2 = cand_a[i], cand_a[j]
                    w12 = weights[it1] + weights[it2]
                    for b in receivers:
                        if a == b or not bins[b]:
                            continue
                        # quick filter: receiver must have slack for w12 after removing one item
                        cand_b = bin_item_candidates(b, t=7)
                        for k in cand_b:
                            if time_check_counter[0] % 250 == 0 and time.time() - start > effective_limit:
                                return False
                            time_check_counter[0] += 1

                            if assign[it1] != a or assign[it2] != a or assign[k] != b:
                                continue
                            # feasibility
                            if loads[b] - weights[k] + w12 > C:
                                continue
                            if loads[a] - w12 + weights[k] > C:
                                continue

                            # apply: move it1,it2 -> b; move k -> a
                            # do via relocations: relocate k to a, then relocate it1,it2 to b
                            # Need rollback support
                            old_bins = (assign[it1], assign[it2], assign[k])
                            if not try_relocate_move(k, a):
                                continue
                            if not try_relocate_move(it1, b):
                                undo_relocate_move(k, b)
                                continue
                            if not try_relocate_move(it2, b):
                                undo_relocate_move(it1, a)
                                undo_relocate_move(k, b)
                                continue
                            return True
        return False

    # --------------------------
    # Neighborhood 3: Ejection chain (depth-limited)
    # --------------------------
    def try_ejection_chain(time_check_counter: List[int], max_depth: int = 5) -> bool:
        nonempty, by_fs, by_load = compute_bin_views(bins, loads)
        if len(nonempty) < 2:
            return False

        # pick a few start items from small bins to promote elimination
        start_bins = by_load[: min(10, len(by_load))]
        start_items = []
        for b in start_bins:
            start_items.extend(bin_item_candidates(b, t=6))
        if not start_items:
            return False
        random.shuffle(start_items)
        start_items = start_items[: min(25, len(start_items))]

        # candidate destination bins: near-feasible (small deficit) or feasible best-fit
        dest_bins = by_fs[: min(25, len(by_fs))]

        # stack of performed relocations for rollback (it, original_bin)
        moves: List[Tuple[int, int]] = []

        def rollback_all():
            for it, ob in reversed(moves):
                if assign[it] != ob:
                    undo_relocate_move(it, ob)
            moves.clear()

        def chain(it: int, depth: int) -> bool:
            if depth > max_depth:
                return False

            b_from = assign[it]
            w = weights[it]

            # choose a destination bin biased to best-fit among feasible, else smallest deficit
            best_feas = None
            best_def = None
            scanK = min(len(dest_bins), 25)
            for b in dest_bins[:scanK]:
                if b == b_from or not bins[b]:
                    continue
                rem = C - loads[b]
                if rem >= w:
                    res = rem - w
                    if best_feas is None or res < best_feas[0]:
                        best_feas = (res, b)
                        if res == 0:
                            break
                else:
                    deficit = w - rem
                    if best_def is None or deficit < best_def[0]:
                        best_def = (deficit, b)

            if best_feas is not None:
                b_to = best_feas[1]
                if try_relocate_move(it, b_to):
                    moves.append((it, b_from))
                    return True
                return False

            if best_def is None:
                return False

            b_to = best_def[1]
            deficit = best_def[0]

            # need to eject something from b_to to make space
            # try a couple of candidates whose removal helps cover deficit
            cand = bin_item_candidates(b_to, t=6)
            # prioritize heavier items if deficit big; else lighter
            cand.sort(key=lambda x: weights[x], reverse=(deficit > C // 4))
            cand = cand[:3]

            for eject in cand:
                if time_check_counter[0] % 250 == 0 and time.time() - start > effective_limit:
                    return False
                time_check_counter[0] += 1

                if assign[eject] != b_to or assign[it] != b_from:
                    continue
                # first move eject out to somewhere else via recursive chain
                if not try_relocate_move(eject, b_from):
                    # put it temporarily into b_from to free b_to? Too disruptive.
                    # Instead: attempt to place eject somewhere feasible directly.
                    # We'll attempt recursion by placing eject first.
                    pass

                # We'll implement: move eject to some feasible bin (best-fit), then move it into b_to.
                # Find a feasible bin for eject excluding b_to.
                rem_bins = [b for b in dest_bins if b != b_to and b != b_from and bins[b]]
                b_eject = best_fit_bin_for_item(weights[eject], bins, loads, rem_bins)
                if b_eject == -1:
                    continue

                # execute eject relocate
                ob_e = b_to
                if not try_relocate_move(eject, b_eject):
                    continue
                moves.append((eject, ob_e))

                # now try to relocate it into b_to (should fit)
                if loads[b_to] + w <= C:
                    if try_relocate_move(it, b_to):
                        moves.append((it, b_from))
                        return True

                # if still not fit, rollback this eject and continue
                rollback_all()

            return False

        for it in start_items:
            if time_check_counter[0] % 250 == 0 and time.time() - start > effective_limit:
                return False
            time_check_counter[0] += 1

            moves.clear()
            if chain(it, 0):
                # keep moves committed
                moves.clear()
                return True
            else:
                rollback_all()
        return False

    # --------------------------
    # Neighborhood 4/5: 2-2 swap and cheap relocate/swap (finishing)
    # --------------------------
    def try_pair_swap_2_2(time_check_counter: List[int]) -> bool:
        nonempty, by_fs, by_load = compute_bin_views(bins, loads)
        if len(nonempty) < 2:
            return False
        A = by_load[: min(10, len(by_load))]
        B = by_fs[: min(14, len(by_fs))]
        for a in A:
            if len(bins[a]) < 2:
                continue
            cand_a = bin_item_candidates(a, t=6)
            for b in B:
                if a == b or len(bins[b]) < 2:
                    continue
                cand_b = bin_item_candidates(b, t=6)
                # enumerate a-pairs and b-pairs small
                for i in range(len(cand_a)):
                    for j in range(i + 1, len(cand_a)):
                        a1, a2 = cand_a[i], cand_a[j]
                        wa = weights[a1] + weights[a2]
                        for u in range(len(cand_b)):
                            for v in range(u + 1, len(cand_b)):
                                if time_check_counter[0] % 300 == 0 and time.time() - start > effective_limit:
                                    return False
                                time_check_counter[0] += 1

                                b1, b2 = cand_b[u], cand_b[v]
                                if assign[a1] != a or assign[a2] != a or assign[b1] != b or assign[b2] != b:
                                    continue
                                wb = weights[b1] + weights[b2]
                                if loads[a] - wa + wb > C:
                                    continue
                                if loads[b] - wb + wa > C:
                                    continue
                                # execute: move b1,b2 -> a and a1,a2 -> b
                                # do via relocations with rollback on failure
                                if not try_relocate_move(a1, b):
                                    continue
                                if not try_relocate_move(a2, b):
                                    undo_relocate_move(a1, a)
                                    continue
                                if not try_relocate_move(b1, a):
                                    undo_relocate_move(a2, a)
                                    undo_relocate_move(a1, a)
                                    continue
                                if not try_relocate_move(b2, a):
                                    undo_relocate_move(b1, b)
                                    undo_relocate_move(a2, a)
                                    undo_relocate_move(a1, a)
                                    continue
                                return True
        return False

    def try_simple_relocate_or_swap(time_check_counter: List[int]) -> bool:
        nonempty, by_fs, by_load = compute_bin_views(bins, loads)
        if not nonempty:
            return False

        # relocate: attempt to move from small bins into large-slack bins
        donors = by_load[: min(12, len(by_load))]
        receivers = by_fs[: min(18, len(by_fs))]
        for a in donors:
            if not bins[a]:
                continue
            items = bin_item_candidates(a, t=8)
            items.sort(key=lambda it: weights[it], reverse=True)
            for it in items:
                b_from = assign[it]
                if b_from != a:
                    continue
                w = weights[it]
                # best-fit among receivers
                best_b = best_fit_bin_for_item(w, bins, loads, [b for b in receivers if b != a and bins[b]])
                if best_b != -1:
                    if time_check_counter[0] % 300 == 0 and time.time() - start > effective_limit:
                        return False
                    time_check_counter[0] += 1

                    if try_relocate_move(it, best_b):
                        return True

        # fallback: a few random feasible swaps
        for _ in range(80):
            if time_check_counter[0] % 300 == 0 and time.time() - start > effective_limit:
                return False
            time_check_counter[0] += 1

            i1 = random.randrange(n)
            i2 = random.randrange(n)
            if i1 != i2 and try_swap_move(i1, i2):
                return True

        return False

    # --------------------------
    # VND with limited sideways acceptance
    # --------------------------
    def vnd(move_budget: int = 2500):
        nonlocal best_obj, best_bins, best_loads, best_assign, best_pos, best_nonempty, best_slack

        # aspiration: allow limited sideways moves if they improve bin-closure potential
        sideways_left = 18

        time_check_counter = [0]
        attempts = 0

        while attempts < move_budget:
            if attempts % 400 == 0 and time.time() - start > effective_limit:
                return

            o_cur = objective(nonempty_count, slack_sq_sum)

            # keep best-so-far
            if o_cur < best_obj:
                best_obj = o_cur
                best_bins = [b[:] for b in bins]
                best_loads = loads[:]
                best_assign = assign[:]
                best_pos = pos_in_bin[:]
                best_nonempty = nonempty_count
                best_slack = slack_sq_sum
                sideways_left = 18

            # refresh candidate views lazily
            targets, by_fs = pick_target_bins_for_elimination(limit=16)

            # Neighborhood 1: bin elimination (commit only on success)
            eliminated = False
            for b_from in targets[: min(10, len(targets))]:
                if attempts % 200 == 0 and time.time() - start > effective_limit:
                    return
                attempts += 1
                pre_bins = nonempty_count
                pre_slack = slack_sq_sum
                pre_state = (best_obj,)  # dummy to avoid linter

                if try_eliminate_bin(b_from, by_fs, time_check_counter):
                    eliminated = True
                    maybe_compact(force=True)
                    break
                else:
                    # ensure state is consistent; if elimination failed it rolled back internally
                    pass

            if eliminated:
                continue

            # Other neighborhoods attempt a move; accept if improves objective,
            # or (sideways) if increases slack_sq_sum or reduces #items in a target small bin.
            moved = False

            # Try 2-1 exchange
            pre_o = objective(nonempty_count, slack_sq_sum)
            pre_slack_local = slack_sq_sum
            if try_exchange_2_1(time_check_counter):
                moved = True
            else:
                # ejection chain
                if try_ejection_chain(time_check_counter, max_depth=5):
                    moved = True
                else:
                    # 2-2
                    if try_pair_swap_2_2(time_check_counter):
                        moved = True
                    else:
                        # simple relocate/swap
                        if try_simple_relocate_or_swap(time_check_counter):
                            moved = True

            if not moved:
                break

            post_o = objective(nonempty_count, slack_sq_sum)
            if post_o < pre_o:
                # keep
                attempts += 1
            else:
                # aspiration: allow limited sideways if slack concentration improved
                if sideways_left > 0 and slack_sq_sum > pre_slack_local:
                    sideways_left -= 1
                    attempts += 1
                else:
                    # revert to best snapshot (cheap and safe within VND)
                    bins[:] = [b[:] for b in best_bins]
                    loads[:] = best_loads[:]
                    assign[:] = best_assign[:]
                    pos_in_bin[:] = best_pos[:]
                    nonlocal_nonempty = best_nonempty
                    nonlocal_slack = best_slack
                    # assign back
                    nonlocal_vars = (nonlocal_nonempty, nonlocal_slack)
                    # actually set
                    nonlocal_nonempty, nonlocal_slack = nonlocal_vars
                    # Python scoping: set explicitly
                    nonlocal nonempty_count, slack_sq_sum
                    nonempty_count = best_nonempty
                    slack_sq_sum = best_slack
                    break

            # periodic compaction
            if accepted_moves % 500 == 0:
                maybe_compact(force=False)

    # --------------------------
    # Shaking: ruin-and-recreate
    # --------------------------
    def ruin_and_recreate(k: int):
        nonlocal slack_sq_sum

        nonempty, by_fs, by_load = compute_bin_views(bins, loads)
        if not nonempty:
            return

        # r grows with k
        r = min(60, 8 + 4 * k)
        removed: List[int] = []

        # removal strategy mixture
        # 1) from 1-3 smallest bins
        small_bins = by_load[: min(3, len(by_load))]
        for b in small_bins:
            if len(removed) >= r:
                break
            items = bins[b][:]
            random.shuffle(items)
            take = min(len(items), max(1, r // 3))
            removed.extend(items[:take])

        # 2) random bins
        if len(removed) < r:
            for _ in range(3):
                if len(removed) >= r:
                    break
                b = random.choice(nonempty)
                items = bins[b][:]
                if items:
                    random.shuffle(items)
                    take = min(len(items), max(1, (r - len(removed)) // 3))
                    removed.extend(items[:take])

        # 3) fill remainder with random items
        while len(removed) < r:
            it = random.randrange(n)
            if assign[it] >= 0:
                removed.append(it)

        # unique
        rem_set = set()
        uniq_removed = []
        for it in removed:
            if it not in rem_set and assign[it] >= 0:
                rem_set.add(it)
                uniq_removed.append(it)
        removed = uniq_removed

        # Remove them
        # update slack sums by adjusting bins impacted: easiest by incremental removal per item
        for it in removed:
            b = assign[it]
            if b < 0:
                continue
            before_nonempty = bool(bins[b])
            old_ld = loads[b]
            old_sl = slack_sq(old_ld) if before_nonempty else 0
            remove_from_bin(bins, loads, assign, pos_in_bin, b, it)
            after_nonempty = bool(bins[b])
            update_bin_nonempty_before_after(b, before_nonempty, after_nonempty)
            new_sl = slack_sq(loads[b]) if after_nonempty else 0
            slack_sq_sum += (new_sl - old_sl)

        maybe_compact(force=False)

        # Reinsert removed with BFD
        removed.sort(key=lambda it: weights[it], reverse=True)
        for it in removed:
            if time.time() - start > effective_limit:
                return
            w = weights[it]
            nonempty, by_fs, by_load = compute_bin_views(bins, loads)
            # best-fit among all non-empty bins (scan limited top by_fs and then full if needed)
            cand = nonempty
            b_to = best_fit_bin_for_item(w, bins, loads, cand)
            if b_to == -1:
                # open new bin
                b_new = len(bins)
                bins.append([])
                loads.append(0)
                before_nonempty = False
                old_sl = 0
                add_to_bin(bins, loads, assign, pos_in_bin, b_new, it)
                update_bin_nonempty_before_after(b_new, before_nonempty, True)
                slack_sq_sum += slack_sq(loads[b_new]) - old_sl
            else:
                before_nonempty = bool(bins[b_to])
                old_ld = loads[b_to]
                old_sl = slack_sq(old_ld) if before_nonempty else 0
                add_to_bin(bins, loads, assign, pos_in_bin, b_to, it)
                after_nonempty = True
                new_sl = slack_sq(loads[b_to])
                slack_sq_sum += (new_sl - old_sl)

        maybe_compact(force=False)

    def light_random_shake(k: int):
        # keep small probability of simple random moves
        steps = 2 * k
        for _ in range(steps):
            if time.time() - start > effective_limit:
                return
            if random.random() < 0.7:
                it = random.randrange(n)
                b_from = assign[it]
                if b_from < 0:
                    continue
                nonempty, by_fs, by_load = compute_bin_views(bins, loads)
                if len(nonempty) <= 1:
                    continue
                b_to = random.choice(nonempty)
                if b_to != b_from and loads[b_to] + weights[it] <= C:
                    try_relocate_move(it, b_to)
            else:
                i1 = random.randrange(n)
                i2 = random.randrange(n)
                if i1 != i2:
                    try_swap_move(i1, i2)

    # --------------------------
    # Main VNS loop
    # --------------------------
    k_max = 20
    MAX_ITERS = 2000

    # initial intensification
    vnd(move_budget=3000)

    # store best snapshot already
    no_bin_improve = 0
    best_bins_count_seen = best_obj[0]

    for it in range(MAX_ITERS):
        if time.time() - start > effective_limit:
            break

        # periodic restart if stalled in bin-count
        if best_obj[0] < best_bins_count_seen:
            best_bins_count_seen = best_obj[0]
            no_bin_improve = 0
        else:
            no_bin_improve += 1

        if no_bin_improve >= 120:
            # restart current from best plus strong shake
            bins = [b[:] for b in best_bins]
            loads = best_loads[:]
            assign = best_assign[:]
            pos_in_bin = best_pos[:]
            nonempty_count = best_nonempty
            slack_sq_sum = best_slack
            ruin_and_recreate(k=10)
            vnd(move_budget=2600)
            no_bin_improve = 0
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

            # shaking
            if random.random() < 0.85:
                ruin_and_recreate(k)
            else:
                light_random_shake(k)

            # intensification
            vnd(move_budget=2200)

            cur_o = objective(nonempty_count, slack_sq_sum)
            if cur_o < best_obj:
                best_obj = cur_o
                best_bins = [b[:] for b in bins]
                best_loads = loads[:]
                best_assign = assign[:]
                best_pos = pos_in_bin[:]
                best_nonempty = nonempty_count
                best_slack = slack_sq_sum
                k = 1
            else:
                k += 1

    # Final output: compact best solution
    bins = [b[:] for b in best_bins]
    loads = best_loads[:]
    assign = best_assign[:]
    pos_in_bin = best_pos[:]
    nonempty_count = best_nonempty
    slack_sq_sum = best_slack

    bins, loads, nonempty_count, slack_sq_sum = rebuild_compact(bins, loads, assign, pos_in_bin)
    return {"packing": bins, "bin_weights": loads}
