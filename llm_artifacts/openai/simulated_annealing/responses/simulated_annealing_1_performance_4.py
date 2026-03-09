import time
import math
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    C = int(bin_capacity)
    w = weights
    n = len(w)

    start = time.perf_counter()
    deadline = start + float(time_limit)

    def time_up() -> bool:
        return time.perf_counter() >= deadline

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ---------------------- helpers ----------------------
    # Lower bound used only for internal guidance
    sum_w = sum(w)
    lb = (sum_w + C - 1) // C

    # A tightness-like secondary score: smaller is better.
    # We use sum(rem^2) + alpha * (#very_small_bins) to discourage fragmentation.
    small_thresh = 0.18 * C

    def compute_secondary(bw_list: List[int]) -> int:
        s = 0
        tiny = 0
        for load in bw_list:
            r = C - load
            s += r * r
            if load < small_thresh:
                tiny += 1
        # alpha scaled to be meaningful but far smaller than bin-count weight
        return s + int(0.25 * C * C) * tiny

    def pack_greedy(order: List[int], best_fit: bool, tie_noise: bool) -> Tuple[List[List[int]], List[int], List[int]]:
        bins: List[List[int]] = []
        bw: List[int] = []
        item_bin = [-1] * n

        for i in order:
            wi = w[i]
            chosen = -1
            best_metric = None
            for b in range(len(bins)):
                r = C - bw[b]
                if wi <= r:
                    if best_fit:
                        metric = r - wi
                        if tie_noise:
                            metric = metric * 1000000 + random.randrange(1000)
                        if chosen == -1 or metric < best_metric:
                            chosen = b
                            best_metric = metric
                            if metric == 0:
                                break
                    else:
                        chosen = b
                        break
            if chosen == -1:
                chosen = len(bins)
                bins.append([i])
                bw.append(wi)
            else:
                bins[chosen].append(i)
                bw[chosen] += wi
            item_bin[i] = chosen
        return bins, bw, item_bin

    def randomized_decreasing() -> List[int]:
        order = list(range(n))
        # shuffle ties by random
        order.sort(key=lambda i: (-w[i], random.random()))
        return order

    # ---------------------- initial multi-start ----------------------
    # Build several candidates; keep the best by (B, secondary)
    order_dec = list(range(n))
    order_dec.sort(key=lambda i: w[i], reverse=True)

    candidates: List[Tuple[List[List[int]], List[int], List[int]]] = []
    # Deterministic baselines
    candidates.append(pack_greedy(order_dec, best_fit=False, tie_noise=False))
    candidates.append(pack_greedy(order_dec, best_fit=True, tie_noise=False))

    # Randomized starts budget
    init_budget = min(1.2, 0.08 * float(time_limit))
    init_end = min(deadline, start + init_budget)

    while time.perf_counter() < init_end and len(candidates) < 18:
        best_fit = (random.random() < 0.8)
        candidates.append(pack_greedy(randomized_decreasing(), best_fit=best_fit, tie_noise=True))

    best_pack = None
    best_key = None
    for cbins, cbw, citem_bin in candidates:
        key = (len(cbw), compute_secondary(cbw))
        if best_key is None or key < best_key:
            best_key = key
            best_pack = (cbins, cbw, citem_bin)

    bins, bw, item_bin = best_pack  # type: ignore

    # ---------------------- SA state ----------------------
    B = len(bins)
    rem = [C - x for x in bw]

    # Energy: bin count dominates
    # Secondary scale: roughly O(B*C^2)
    M = 10 * C * C

    def energy(Bv: int, bw_list: List[int]) -> int:
        return Bv * M + compute_secondary(bw_list)

    curE = energy(B, bw)

    best_bins = [b[:] for b in bins]
    best_bw = bw[:]
    best_B = B
    best_E = curE

    # ---------------------- move primitives + undo ----------------------
    # We keep bins as lists; for O(1) remove we swap-with-last.
    def remove_from_bin(b: int, pos: int) -> int:
        lst = bins[b]
        x = lst[pos]
        last = lst[-1]
        lst[pos] = last
        lst.pop()
        return x

    def add_to_bin(b: int, item: int) -> None:
        bins[b].append(item)

    def open_bin_with(item: int) -> int:
        b = len(bins)
        bins.append([item])
        bw.append(w[item])
        rem.append(C - w[item])
        item_bin[item] = b
        return b

    def close_empty_bin(b: int) -> Tuple[int, List[int], int]:
        # remove bin b by swapping last into b
        last = len(bins) - 1
        moved_items = []
        moved_from = last
        if b != last:
            moved_items = bins[last]
            bins[b] = bins[last]
            bw[b] = bw[last]
            rem[b] = rem[last]
            for it in bins[b]:
                item_bin[it] = b
        # pop last
        bins.pop(); bw.pop(); rem.pop()
        return moved_from, moved_items, last

    # change log entries:
    # ('reloc', item, src, dst, src_pos)
    # ('swap', b1, p1, b2, p2, i, j)
    # ('open', bnew)
    # ('close', bclosed, b_was_last, saved_items, saved_bw, saved_rem)

    def undo(changes: List[Tuple]) -> None:
        # undo in reverse
        for ch in reversed(changes):
            tag = ch[0]
            if tag == 'reloc':
                _, item, src, dst, src_pos = ch
                wi = w[item]
                # remove from dst (last)
                bins[dst].pop()
                bw[dst] -= wi
                rem[dst] += wi
                # add back to src at src_pos
                bins[src].append(item)
                back = len(bins[src]) - 1
                bins[src][back], bins[src][src_pos] = bins[src][src_pos], bins[src][back]
                bw[src] += wi
                rem[src] -= wi
                item_bin[item] = src

            elif tag == 'swap':
                _, b1, p1, b2, p2, i, j = ch
                wi, wj = w[i], w[j]
                # swap back
                bins[b1][p1] = i
                bins[b2][p2] = j
                item_bin[i] = b1
                item_bin[j] = b2
                bw[b1] = bw[b1] - wj + wi
                bw[b2] = bw[b2] - wi + wj
                rem[b1] = C - bw[b1]
                rem[b2] = C - bw[b2]

            elif tag == 'open':
                _, bnew = ch
                # bnew must be last
                for it in bins[bnew]:
                    item_bin[it] = -1
                bins.pop(); bw.pop(); rem.pop()

            elif tag == 'close':
                _, bclosed, b_was_last, saved_items, saved_bw, saved_rem = ch
                # restore by appending and swapping if needed
                bins.append(saved_items)
                bw.append(saved_bw)
                rem.append(saved_rem)
                restored_last = len(bins) - 1
                for it in saved_items:
                    item_bin[it] = restored_last

                if not b_was_last:
                    # swap restored_last into bclosed, and move current bclosed to last
                    # current bclosed holds some other bin; capture it
                    cur_items = bins[bclosed]
                    cur_bw = bw[bclosed]
                    cur_rem = rem[bclosed]

                    bins[bclosed] = bins[restored_last]
                    bw[bclosed] = bw[restored_last]
                    rem[bclosed] = rem[restored_last]
                    for it in bins[bclosed]:
                        item_bin[it] = bclosed

                    bins[restored_last] = cur_items
                    bw[restored_last] = cur_bw
                    rem[restored_last] = cur_rem
                    for it in bins[restored_last]:
                        item_bin[it] = restored_last

            else:
                # should not happen
                pass

    # ---------------------- destination selection ----------------------
    # Best-fit among random samples biased toward tight bins
    def pick_dest(item: int, exclude: int, k: int = 18) -> int:
        wi = w[item]
        best = -1
        best_after = C + 1

        Bn = len(bins)
        if Bn <= 1:
            return -1

        for _ in range(k):
            b = random.randrange(Bn)
            if b == exclude:
                continue
            if rem[b] >= wi:
                after = rem[b] - wi
                if after < best_after:
                    best_after = after
                    best = b
                    if after == 0:
                        break
        return best

    # ---------------------- neighborhoods (SA moves) ----------------------
    def do_relocate() -> Tuple[bool, List[Tuple]]:
        if len(bins) <= 0:
            return False, []
        src = random.randrange(len(bins))
        if not bins[src]:
            return False, []
        src_pos = random.randrange(len(bins[src]))
        item = bins[src][src_pos]
        wi = w[item]

        dst = pick_dest(item, exclude=src, k=22)
        changes: List[Tuple] = []

        if dst == -1:
            # optionally open a new bin only if it helps later; usually not.
            return False, []

        # apply
        remove_from_bin(src, src_pos)
        bw[src] -= wi
        rem[src] += wi

        add_to_bin(dst, item)
        bw[dst] += wi
        rem[dst] -= wi
        item_bin[item] = dst

        changes.append(('reloc', item, src, dst, src_pos))

        # close src if empty
        if not bins[src]:
            # save last bin snapshot before closure for undo
            last = len(bins) - 1
            b_was_last = (src == last)
            saved_items = bins[last][:]
            saved_bw = bw[last]
            saved_rem = rem[last]
            changes.append(('close', src, b_was_last, saved_items, saved_bw, saved_rem))
            close_empty_bin(src)

        return True, changes

    def do_swap() -> Tuple[bool, List[Tuple]]:
        if len(bins) < 2:
            return False, []
        b1 = random.randrange(len(bins))
        b2 = random.randrange(len(bins) - 1)
        if b2 >= b1:
            b2 += 1
        if not bins[b1] or not bins[b2]:
            return False, []
        p1 = random.randrange(len(bins[b1]))
        p2 = random.randrange(len(bins[b2]))
        i = bins[b1][p1]
        j = bins[b2][p2]
        wi, wj = w[i], w[j]

        new1 = bw[b1] - wi + wj
        new2 = bw[b2] - wj + wi
        if new1 > C or new2 > C:
            return False, []

        # apply
        bins[b1][p1] = j
        bins[b2][p2] = i
        item_bin[i] = b2
        item_bin[j] = b1
        bw[b1] = new1
        bw[b2] = new2
        rem[b1] = C - new1
        rem[b2] = C - new2

        return True, [('swap', b1, p1, b2, p2, i, j)]

    def do_two_move() -> Tuple[bool, List[Tuple]]:
        # Move two items from a source bin into two (possibly same) destination bins.
        if len(bins) < 2:
            return False, []
        src = random.randrange(len(bins))
        if len(bins[src]) < 2:
            return False, []

        # pick two items
        p1 = random.randrange(len(bins[src]))
        p2 = random.randrange(len(bins[src]) - 1)
        if p2 >= p1:
            p2 += 1
        i1 = bins[src][p1]
        i2 = bins[src][p2]
        wi1, wi2 = w[i1], w[i2]

        # choose destinations (best-fit samples)
        d1 = pick_dest(i1, exclude=src, k=20)
        d2 = pick_dest(i2, exclude=src, k=20)
        if d1 == -1 or d2 == -1:
            return False, []

        # ensure feasibility sequentially
        if rem[d1] < wi1:
            return False, []
        # after placing i1, rem changes if d1==d2
        if d1 == d2:
            if rem[d1] - wi1 < wi2:
                return False, []
        else:
            if rem[d2] < wi2:
                return False, []

        changes: List[Tuple] = []

        # remove higher index first to keep positions valid
        if p1 > p2:
            first = (i1, p1, d1, wi1)
            second = (i2, p2, d2, wi2)
        else:
            first = (i2, p2, d2, wi2)
            second = (i1, p1, d1, wi1)

        # apply first relocation
        item, pos, dst, wi = first
        remove_from_bin(src, pos)
        bw[src] -= wi
        rem[src] += wi
        add_to_bin(dst, item)
        bw[dst] += wi
        rem[dst] -= wi
        item_bin[item] = dst
        changes.append(('reloc', item, src, dst, pos))

        # apply second relocation (src positions still valid due to remove ordering)
        item, pos, dst, wi = second
        remove_from_bin(src, pos)
        bw[src] -= wi
        rem[src] += wi
        add_to_bin(dst, item)
        bw[dst] += wi
        rem[dst] -= wi
        item_bin[item] = dst
        changes.append(('reloc', item, src, dst, pos))

        # close if emptied
        if not bins[src]:
            last = len(bins) - 1
            b_was_last = (src == last)
            saved_items = bins[last][:]
            saved_bw = bw[last]
            saved_rem = rem[last]
            changes.append(('close', src, b_was_last, saved_items, saved_bw, saved_rem))
            close_empty_bin(src)

        return True, changes

    def do_empty_bin() -> Tuple[bool, List[Tuple]]:
        # Classic high-impact SA move: try to empty a light bin by re-inserting its items.
        if len(bins) < 2:
            return False, []

        # pick a likely victim: small load or few items
        # sample a few and choose best victim score
        trials = 8
        best_v = -1
        best_score = None
        for _ in range(trials):
            b = random.randrange(len(bins))
            if not bins[b]:
                continue
            if b == -1:
                continue
            # score: prefer low load and few items
            score = bw[b] + 0.35 * C * len(bins[b])
            if best_score is None or score < best_score:
                best_score = score
                best_v = b

        v = best_v
        if v == -1 or not bins[v] or len(bins) == 1:
            return False, []

        items = bins[v][:]
        # move larger items first
        items.sort(key=lambda it: w[it], reverse=True)

        # plan destinations
        plan: List[Tuple[int, int]] = []
        for it in items:
            d = pick_dest(it, exclude=v, k=28)
            if d == -1:
                return False, []
            # check feasibility using current rem snapshot + planned adds
            plan.append((it, d))

        # verify feasibility with a shadow rem
        shadow = rem[:]
        shadow[v] = rem[v]
        for it, d in plan:
            wi = w[it]
            if shadow[d] < wi:
                return False, []
            shadow[d] -= wi
            shadow[v] += wi

        changes: List[Tuple] = []

        # execute using position map for victim with swap-removals
        pos = {bins[v][k]: k for k in range(len(bins[v]))}
        for it, d in plan:
            p = pos[it]
            last_it = bins[v][-1]
            wi = w[it]

            remove_from_bin(v, p)
            bw[v] -= wi
            rem[v] += wi

            add_to_bin(d, it)
            bw[d] += wi
            rem[d] -= wi
            item_bin[it] = d

            changes.append(('reloc', it, v, d, p))

            if it != last_it:
                pos[last_it] = p
            pos.pop(it, None)

        # close victim
        if not bins[v]:
            last = len(bins) - 1
            b_was_last = (v == last)
            saved_items = bins[last][:]
            saved_bw = bw[last]
            saved_rem = rem[last]
            changes.append(('close', v, b_was_last, saved_items, saved_bw, saved_rem))
            close_empty_bin(v)
        else:
            # should not happen
            return False, []

        return True, changes

    # ---------------------- temperature calibration ----------------------
    # Collect a few positive deltas to set T0
    def try_any_move() -> Tuple[bool, List[Tuple]]:
        r = random.random()
        if r < 0.52:
            return do_relocate()
        elif r < 0.68:
            return do_swap()
        elif r < 0.83:
            return do_two_move()
        else:
            return do_empty_bin()

    # quick probe
    probe_end = min(deadline, time.perf_counter() + min(0.7, 0.03 * float(time_limit) + 0.15))
    pos_dE: List[int] = []
    probe_iters = 0
    while time.perf_counter() < probe_end and probe_iters < 4000 and len(pos_dE) < 250:
        e0 = energy(len(bins), bw)
        ok, changes = try_any_move()
        if not ok:
            probe_iters += 1
            continue
        e1 = energy(len(bins), bw)
        dE = e1 - e0
        if dE > 0:
            pos_dE.append(dE)
        # always undo during probe to not drift too far
        undo(changes)
        probe_iters += 1

    if pos_dE:
        pos_dE.sort()
        med = pos_dE[len(pos_dE) // 2]
        # want acceptance exp(-med/T0) ~ 0.55
        T0 = max(1e-9, -med / math.log(0.55))
    else:
        T0 = max(1.0, 0.15 * C)

    Tmin = T0 * 1e-4
    T = T0

    # ---------------------- iteration budgeting ----------------------
    # Calibrate iteration rate with real SA (with occasional undos)
    cal_start = time.perf_counter()
    cal_end = min(deadline, cal_start + min(1.0, 0.04 * float(time_limit) + 0.2))
    iters = 0
    acc = 0
    while time.perf_counter() < cal_end and iters < 20000:
        e0 = energy(len(bins), bw)
        ok, changes = try_any_move()
        if not ok:
            iters += 1
            continue
        e1 = energy(len(bins), bw)
        dE = e1 - e0
        if dE <= 0 or random.random() < math.exp(-dE / max(T, 1e-12)):
            acc += 1
            # keep
        else:
            undo(changes)
        iters += 1

    elapsed = max(1e-6, time.perf_counter() - cal_start)
    iters_per_sec = iters / elapsed

    # plan to run until time limit (with periodic checks)
    remaining = max(0.0, deadline - time.perf_counter())
    max_iters = int(min(80_000_000, max(50_000, remaining * iters_per_sec * 0.995)))

    # ---------------------- SA main loop ----------------------
    epoch = max(1200, min(8000, 800 + 8 * n))
    target_acc_low = 0.18
    target_acc_high = 0.45

    accepted = 0
    attempted = 0
    no_improve_epochs = 0

    # Move mix shifts toward empty-bin and 2-move later
    def move_choice(progress: float) -> int:
        r = random.random()
        if progress < 0.35:
            # explore
            if r < 0.55:
                return 0  # relocate
            if r < 0.72:
                return 1  # swap
            if r < 0.88:
                return 2  # two-move
            return 3      # empty-bin
        elif progress < 0.80:
            if r < 0.42:
                return 0
            if r < 0.55:
                return 1
            if r < 0.78:
                return 2
            return 3
        else:
            if r < 0.25:
                return 0
            if r < 0.33:
                return 1
            if r < 0.63:
                return 2
            return 3

    # periodic time checks
    check_mask = 2047

    for it in range(max_iters):
        if (it & check_mask) == 0 and time_up():
            break

        progress = it / max(1, max_iters)
        mv = move_choice(progress)

        e0 = energy(len(bins), bw)
        if mv == 0:
            ok, changes = do_relocate()
        elif mv == 1:
            ok, changes = do_swap()
        elif mv == 2:
            ok, changes = do_two_move()
        else:
            ok, changes = do_empty_bin()

        attempted += 1
        if not ok:
            continue

        e1 = energy(len(bins), bw)
        dE = e1 - e0

        if dE <= 0 or random.random() < math.exp(-dE / max(T, 1e-12)):
            accepted += 1
            curE = e1
        else:
            undo(changes)
            curE = e0

        # record best by lexicographic (B, E)
        curB = len(bins)
        if curB < best_B or (curB == best_B and curE < best_E):
            best_B = curB
            best_E = curE
            best_bins = [b[:] for b in bins]
            best_bw = bw[:]
            no_improve_epochs = 0

            # if we hit LB, keep intensifying secondary objective
            if best_B == lb:
                # lower temp to polish
                T = max(Tmin, T * 0.7)

        # epoch adaptation
        if (it + 1) % epoch == 0:
            acc_rate = accepted / max(1, attempted)

            # adaptive temperature to keep acceptance in band
            if acc_rate < target_acc_low:
                T *= 1.25
            elif acc_rate > target_acc_high:
                T *= 0.85
            else:
                T *= 0.95

            # gentle cooling with progress
            T *= (0.995 - 0.15 * progress * 0.01)

            T = min(T0 * 3.0, max(Tmin, T))

            accepted = 0
            attempted = 0

            no_improve_epochs += 1
            # reheat if stuck
            if no_improve_epochs >= 6:
                # stronger reheat later to escape deep basins
                T = min(T0 * 3.0, T * (1.8 if progress < 0.7 else 2.2))
                no_improve_epochs = 0

    return {"packing": best_bins, "bin_weights": best_bw}
