import time
import random
from typing import List, Tuple, Dict, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = int(bin_capacity)
    w = weights
    n = len(w)
    start = time.time()

    def now() -> float:
        return time.time() - start

    # -------------------- Construction (multi-start randomized BFD) --------------------
    def rbfd_initial(trials: int) -> Tuple[List[List[int]], List[int], List[int]]:
        # Randomized Best-Fit Decreasing: among best bins (min remaining space) choose randomly.
        best = None
        best_m = 10**9
        order_base = sorted(range(n), key=lambda i: w[i], reverse=True)

        for _ in range(trials):
            # small randomization: shuffle within equal-weight blocks
            order = order_base[:]
            i = 0
            while i < n:
                j = i + 1
                wi = w[order[i]]
                while j < n and w[order[j]] == wi:
                    j += 1
                if j - i > 1:
                    block = order[i:j]
                    random.shuffle(block)
                    order[i:j] = block
                i = j

            bins: List[List[int]] = []
            loads: List[int] = []
            assign = [-1] * n

            for item in order:
                wi = w[item]
                best_bins = []
                best_rem = C + 1
                for b, lb in enumerate(loads):
                    rem = C - lb
                    if wi <= rem:
                        new_rem = rem - wi
                        if new_rem < best_rem:
                            best_rem = new_rem
                            best_bins = [b]
                        elif new_rem == best_rem:
                            best_bins.append(b)
                if not best_bins:
                    b = len(bins)
                    bins.append([item])
                    loads.append(wi)
                    assign[item] = b
                else:
                    b = random.choice(best_bins)
                    bins[b].append(item)
                    loads[b] += wi
                    assign[item] = b

            m = len(bins)
            if m < best_m:
                best_m = m
                best = (bins, loads, assign)

        assert best is not None
        return best

    # -------------------- Utilities --------------------
    def clone_solution(bins: List[List[int]], loads: List[int], assign: List[int]):
        return [b[:] for b in bins], loads[:], assign[:]

    def remove_empty_bins(bins: List[List[int]], loads: List[int], assign: List[int]) -> None:
        mapping = [-1] * len(bins)
        new_bins: List[List[int]] = []
        new_loads: List[int] = []
        for b, items in enumerate(bins):
            if items:
                mapping[b] = len(new_bins)
                new_bins.append(items)
                new_loads.append(loads[b])
        if len(new_bins) != len(bins):
            for i in range(n):
                bi = assign[i]
                if bi != -1:
                    assign[i] = mapping[bi]
            bins[:] = new_bins
            loads[:] = new_loads

    def compute_slacks(loads: List[int]) -> Tuple[int, int]:
        s = 0
        s2 = 0
        for lb in loads:
            t = C - lb
            s += t
            s2 += t * t
        return s, s2

    def score(m: int, slack: int, slack2: int) -> Tuple[int, int, int]:
        return (m, slack, slack2)

    # -------------------- Moves (with incremental slack maintenance) --------------------
    def apply_relocate(item: int, src: int, dst: int,
                       bins: List[List[int]], loads: List[int], assign: List[int],
                       slack: int, slack2: int) -> Tuple[int, int]:
        wi = w[item]
        # update slack terms for src and dst (before)
        s_src = C - loads[src]
        s_dst = C - loads[dst]

        # do move
        bins[src].remove(item)
        loads[src] -= wi
        bins[dst].append(item)
        loads[dst] += wi
        assign[item] = dst

        # after
        s_src2 = C - loads[src]
        s_dst2 = C - loads[dst]

        slack += (s_src2 + s_dst2) - (s_src + s_dst)
        slack2 += (s_src2 * s_src2 + s_dst2 * s_dst2) - (s_src * s_src + s_dst * s_dst)

        if not bins[src]:
            # remove empty bin => remove its slack contribution (which is C) and slack2 (C^2)
            # but loads[src] is 0 so s_src2 == C already included; removing bin removes that
            slack -= C
            slack2 -= C * C
            remove_empty_bins(bins, loads, assign)

        return slack, slack2

    def apply_swap(a: int, b: int, bin_a: int, bin_b: int,
                   bins: List[List[int]], loads: List[int], assign: List[int],
                   slack: int, slack2: int) -> Tuple[int, int]:
        wa, wb = w[a], w[b]
        s_a = C - loads[bin_a]
        s_b = C - loads[bin_b]

        bins[bin_a].remove(a)
        bins[bin_b].remove(b)
        bins[bin_a].append(b)
        bins[bin_b].append(a)
        loads[bin_a] += wb - wa
        loads[bin_b] += wa - wb
        assign[a] = bin_b
        assign[b] = bin_a

        s_a2 = C - loads[bin_a]
        s_b2 = C - loads[bin_b]
        slack += (s_a2 + s_b2) - (s_a + s_b)  # should remain same
        slack2 += (s_a2 * s_a2 + s_b2 * s_b2) - (s_a * s_a + s_b * s_b)
        return slack, slack2

    # 2-1 exchange: move two items (x,y) from bin_t -> bin_j, and one item z from bin_j -> bin_t
    def apply_two_one(x: int, y: int, z: int, bin_t: int, bin_j: int,
                      bins: List[List[int]], loads: List[int], assign: List[int],
                      slack: int, slack2: int) -> Tuple[int, int]:
        s_t = C - loads[bin_t]
        s_j = C - loads[bin_j]

        bins[bin_t].remove(x)
        bins[bin_t].remove(y)
        bins[bin_j].remove(z)

        bins[bin_j].append(x)
        bins[bin_j].append(y)
        bins[bin_t].append(z)

        loads[bin_t] += w[z] - w[x] - w[y]
        loads[bin_j] += w[x] + w[y] - w[z]

        assign[x] = bin_j
        assign[y] = bin_j
        assign[z] = bin_t

        s_t2 = C - loads[bin_t]
        s_j2 = C - loads[bin_j]
        slack += (s_t2 + s_j2) - (s_t + s_j)
        slack2 += (s_t2 * s_t2 + s_j2 * s_j2) - (s_t * s_t + s_j * s_j)

        if not bins[bin_t]:
            slack -= C
            slack2 -= C * C
            remove_empty_bins(bins, loads, assign)
        return slack, slack2

    # -------------------- Tabu machinery --------------------
    # Tabu on (item, destination_bin) like before; also store reverse for swaps implicitly.
    tabu: Dict[Tuple[int, int], int] = {}

    def is_tabu(item: int, dst: int, it: int) -> bool:
        return tabu.get((item, dst), -1) > it

    def set_tabu(item: int, dst: int, it: int, tenure: int) -> None:
        tabu[(item, dst)] = it + tenure

    # -------------------- Target selection / sampling --------------------
    def light_bins(loads: List[int], k: int) -> List[int]:
        idx = list(range(len(loads)))
        idx.sort(key=lambda b: loads[b])
        return idx[: min(k, len(idx))]

    def heavy_bins(loads: List[int], k: int) -> List[int]:
        idx = list(range(len(loads)))
        idx.sort(key=lambda b: loads[b], reverse=True)
        return idx[: min(k, len(idx))]

    def pick_items_from_bin(items: List[int], topk: int) -> List[int]:
        if not items:
            return []
        if len(items) <= topk:
            return items[:]
        # return top by weight
        tmp = sorted(items, key=lambda i: w[i], reverse=True)
        return tmp[:topk]

    # -------------------- Greedy emptying (intensification step) --------------------
    def try_greedy_empty_bin(tbin: int,
                             bins: List[List[int]], loads: List[int], assign: List[int],
                             slack: int, slack2: int,
                             it: int, tenure: int,
                             best_sc: Tuple[int, int, int]) -> Tuple[bool, int, int]:
        # Attempt to relocate all items of tbin into other bins (one by one best-fit).
        if tbin < 0 or tbin >= len(bins) or not bins[tbin]:
            return False, slack, slack2
        items = sorted(bins[tbin][:], key=lambda i: w[i], reverse=True)
        # snapshot for rollback
        snap_bins, snap_loads, snap_assign = clone_solution(bins, loads, assign)
        snap_slack, snap_slack2 = slack, slack2

        for item in items:
            src = assign[item]
            if src == -1 or src >= len(bins):
                continue
            wi = w[item]
            # best-fit destination among all except src
            best_dst = -1
            best_rem = C + 1
            for dst in range(len(bins)):
                if dst == src:
                    continue
                rem = C - loads[dst]
                if wi <= rem:
                    r2 = rem - wi
                    if r2 < best_rem:
                        best_rem = r2
                        best_dst = dst
                        if best_rem == 0:
                            break
            if best_dst == -1:
                # rollback
                bins[:], loads[:], assign[:] = snap_bins, snap_loads, snap_assign
                return False, snap_slack, snap_slack2
            slack, slack2 = apply_relocate(item, src, best_dst, bins, loads, assign, slack, slack2)
            set_tabu(item, best_dst, it, tenure)

        # if improved bins count (or even equal but better slack), keep
        cur_sc = score(len(bins), slack, slack2)
        if cur_sc < best_sc:
            return True, slack, slack2
        return True, slack, slack2

    # -------------------- Initial best --------------------
    # More trials for larger instances, still cheap.
    trials = 10 if n <= 200 else 18 if n <= 800 else 26
    bins, loads, assign = rbfd_initial(trials)
    remove_empty_bins(bins, loads, assign)
    slack, slack2 = compute_slacks(loads)

    best_bins, best_loads, best_assign = clone_solution(bins, loads, assign)
    best_slack, best_slack2 = slack, slack2
    best_score = score(len(best_bins), best_slack, best_slack2)

    # -------------------- Parameters (quality-oriented) --------------------
    # Iteration budget: allow long runs; also stop by time.
    max_iters = max(20000, min(400000, 2500 * max(1, n // 10)))

    # Candidate limits (more thorough)
    cand_reloc = max(200, min(2500, 18 * n))
    cand_swap = max(150, min(2000, 12 * n))
    cand_two_one = max(120, min(1600, 10 * n))

    # Tabu tenure adaptive
    base_tenure = 8
    span_tenure = 18

    stagnation_limit = max(1200, 25 * n)
    restart_limit = max(5000, 90 * n)

    # -------------------- Search loop --------------------
    it = 0
    last_improve = 0
    last_restart = 0

    while it < max_iters:
        it += 1
        # periodic time checks
        if (it & 0x3F) == 0 and now() >= time_limit:
            break

        m = len(bins)
        if m <= 1:
            break

        tenure = base_tenure + random.randint(0, span_tenure) + (m // 10)

        # Intensification occasionally: try to empty a very light bin greedily
        if (it % 50) == 0:
            tlist = light_bins(loads, 3)
            if tlist:
                ok, slack, slack2 = try_greedy_empty_bin(
                    tlist[0], bins, loads, assign, slack, slack2,
                    it, tenure, best_score
                )
                if ok:
                    cur_sc = score(len(bins), slack, slack2)
                    if cur_sc < best_score:
                        best_score = cur_sc
                        best_bins, best_loads, best_assign = clone_solution(bins, loads, assign)
                        best_slack, best_slack2 = slack, slack2
                        last_improve = it
                        continue

        # Focus bins: try to eliminate one of the lightest bins
        focus_bins = light_bins(loads, max(6, m // 4))
        focus = focus_bins[0] if focus_bins else random.randrange(m)

        # Precompute lists
        all_bins = list(range(m))

        best_move = None
        best_move_sc = None

        # ---------- Relocate neighborhood (biased to empty focus bin) ----------
        # Prefer moving larger items out of focus; destinations prefer high load / tight fit
        focus_items = pick_items_from_bin(bins[focus], topk=min(8, len(bins[focus])))
        if not focus_items and bins[focus]:
            focus_items = [random.choice(bins[focus])]

        heavy = heavy_bins(loads, max(10, m // 3))

        for _ in range(cand_reloc):
            if now() >= time_limit:
                break
            if not focus_items:
                break
            item = random.choice(focus_items)
            src = assign[item]
            if src == -1:
                continue

            # pick dst from heavy bins with some randomness
            dst = random.choice(heavy) if heavy else random.randrange(m)
            if dst == src:
                continue
            wi = w[item]
            if loads[dst] + wi > C:
                # try a few alternatives
                ok = False
                for _t in range(5):
                    dst2 = random.choice(all_bins)
                    if dst2 != src and loads[dst2] + wi <= C:
                        dst = dst2
                        ok = True
                        break
                if not ok:
                    continue

            # tabu / aspiration
            new_m = m - 1 if (src != dst and len(bins[src]) == 1) else m
            # slack unchanged unless empty bin removed
            new_slack = slack - C if (src != dst and len(bins[src]) == 1) else slack
            # slack2 delta for src/dst
            s_src = C - loads[src]
            s_dst = C - loads[dst]
            s_src2 = s_src + wi
            s_dst2 = s_dst - wi
            new_slack2 = slack2 + (s_src2 * s_src2 + s_dst2 * s_dst2) - (s_src * s_src + s_dst * s_dst)
            if len(bins[src]) == 1:
                new_slack2 -= C * C

            new_sc = score(new_m, new_slack, new_slack2)

            tabu_flag = is_tabu(item, dst, it)
            if tabu_flag and not (new_sc < best_score):
                continue
            if best_move_sc is None or new_sc < best_move_sc:
                best_move_sc = new_sc
                best_move = ("reloc", item, src, dst)

        # ---------- 2-1 exchange neighborhood (powerful for emptying focus) ----------
        # Choose two items from focus and a donor bin j with one item back.
        if m >= 2 and bins[focus]:
            focus_pool = sorted(bins[focus], key=lambda i: w[i], reverse=True)
            top_pairs = focus_pool[: min(10, len(focus_pool))]

            for _ in range(cand_two_one):
                if now() >= time_limit:
                    break
                if len(top_pairs) < 2:
                    break

                x = random.choice(top_pairs)
                y = random.choice(top_pairs)
                if x == y:
                    continue
                wx, wy = w[x], w[y]

                bin_t = assign[x]
                if bin_t != focus:
                    bin_t = focus
                if bin_t < 0 or bin_t >= m:
                    continue

                bin_j = random.choice(all_bins)
                if bin_j == bin_t or not bins[bin_j]:
                    continue

                # need a z in bin_j
                z_candidates = pick_items_from_bin(bins[bin_j], topk=min(8, len(bins[bin_j])))
                z = random.choice(z_candidates)

                # Feasibility after exchange
                # bin_j gets x,y and loses z
                if loads[bin_j] - w[z] + wx + wy > C:
                    continue
                # bin_t loses x,y and gets z
                if loads[bin_t] - wx - wy + w[z] > C:
                    continue

                # score delta for two bins
                new_m = m - 1 if len(bins[bin_t]) == 2 else m
                new_slack = slack - C if len(bins[bin_t]) == 2 else slack

                s_t = C - loads[bin_t]
                s_j = C - loads[bin_j]
                s_t2 = s_t + wx + wy - w[z]
                s_j2 = s_j - wx - wy + w[z]
                new_slack2 = slack2 + (s_t2 * s_t2 + s_j2 * s_j2) - (s_t * s_t + s_j * s_j)
                if len(bins[bin_t]) == 2:
                    new_slack2 -= C * C

                new_sc = score(new_m, new_slack, new_slack2)

                tabu_flag = is_tabu(x, bin_j, it) or is_tabu(y, bin_j, it) or is_tabu(z, bin_t, it)
                if tabu_flag and not (new_sc < best_score):
                    continue

                if best_move_sc is None or new_sc < best_move_sc:
                    best_move_sc = new_sc
                    best_move = ("two_one", x, y, z, bin_t, bin_j)

        # ---------- Swap neighborhood (for slack2 improvement / enabling future relocations) ----------
        for _ in range(cand_swap):
            if now() >= time_limit:
                break
            b1 = random.choice(focus_bins) if focus_bins else random.randrange(m)
            b2 = random.randrange(m)
            if b1 == b2 or not bins[b1] or not bins[b2]:
                continue
            a = random.choice(pick_items_from_bin(bins[b1], topk=min(6, len(bins[b1]))))
            bitem = random.choice(pick_items_from_bin(bins[b2], topk=min(6, len(bins[b2]))))
            if a == bitem:
                continue
            wa, wb = w[a], w[bitem]

            if loads[b1] - wa + wb > C:
                continue
            if loads[b2] - wb + wa > C:
                continue

            s1 = C - loads[b1]
            s2_ = C - loads[b2]
            s1b = s1 + wa - wb
            s2b = s2_ + wb - wa
            new_slack2 = slack2 + (s1b * s1b + s2b * s2b) - (s1 * s1 + s2_ * s2_)
            new_sc = score(m, slack, new_slack2)

            tabu_flag = is_tabu(a, b2, it) or is_tabu(bitem, b1, it)
            if tabu_flag and not (new_sc < best_score):
                continue

            if best_move_sc is None or new_sc < best_move_sc:
                best_move_sc = new_sc
                best_move = ("swap", a, bitem, b1, b2)

        # ---------- If no move found, diversify (tabu-guided perturbation) ----------
        if best_move is None:
            # random relocations from light bins
            lb = light_bins(loads, 5)
            for _ in range(8):
                if now() >= time_limit:
                    break
                if not lb:
                    break
                src = random.choice(lb)
                if not bins[src]:
                    continue
                item = random.choice(bins[src])
                wi = w[item]
                # find random feasible dst
                perm = all_bins[:]
                random.shuffle(perm)
                for dst in perm:
                    if dst != src and loads[dst] + wi <= C:
                        slack, slack2 = apply_relocate(item, src, dst, bins, loads, assign, slack, slack2)
                        set_tabu(item, dst, it, tenure)
                        break
            continue

        # ---------- Apply best move ----------
        typ = best_move[0]
        if typ == "reloc":
            _, item, src, dst = best_move
            # if bins were compacted earlier, src/dst could be off; re-read from assign
            src2 = assign[item]
            if src2 != -1:
                src = src2
            if src != dst and loads[dst] + w[item] <= C:
                slack, slack2 = apply_relocate(item, src, dst, bins, loads, assign, slack, slack2)
                set_tabu(item, dst, it, tenure)

        elif typ == "swap":
            _, a, bitem, b1, b2 = best_move
            b1 = assign[a]
            b2 = assign[bitem]
            if b1 != -1 and b2 != -1 and b1 != b2:
                if loads[b1] - w[a] + w[bitem] <= C and loads[b2] - w[bitem] + w[a] <= C:
                    slack, slack2 = apply_swap(a, bitem, b1, b2, bins, loads, assign, slack, slack2)
                    set_tabu(a, b2, it, tenure)
                    set_tabu(bitem, b1, it, tenure)

        else:  # two_one
            _, x, y, z, bin_t, bin_j = best_move
            bin_t = assign[x]
            if bin_t == -1:
                bin_t = focus
            bin_j = assign[z]
            if bin_j == -1:
                continue
            if bin_t == bin_j:
                continue
            # still feasible?
            if loads[bin_j] - w[z] + w[x] + w[y] <= C and loads[bin_t] - w[x] - w[y] + w[z] <= C:
                slack, slack2 = apply_two_one(x, y, z, bin_t, bin_j, bins, loads, assign, slack, slack2)
                set_tabu(x, bin_j, it, tenure)
                set_tabu(y, bin_j, it, tenure)
                set_tabu(z, bin_t, it, tenure)

        # ---------- Best update / stagnation handling ----------
        cur_sc = score(len(bins), slack, slack2)
        if cur_sc < best_score:
            best_score = cur_sc
            best_bins, best_loads, best_assign = clone_solution(bins, loads, assign)
            best_slack, best_slack2 = slack, slack2
            last_improve = it
        else:
            # adaptive diversification via partial restart
            if it - last_restart > restart_limit and it - last_improve > stagnation_limit:
                last_restart = it
                # restart from best then perturb
                bins, loads, assign = clone_solution(best_bins, best_loads, best_assign)
                slack, slack2 = best_slack, best_slack2

                # perturb: move some items from a couple light bins to random feasible bins
                m = len(bins)
                if m > 1:
                    perturb_bins = light_bins(loads, min(4, m))
                    moved = 0
                    for sb in perturb_bins:
                        if not bins[sb]:
                            continue
                        # move up to 3 items
                        items = bins[sb][:]
                        random.shuffle(items)
                        for item in items[:3]:
                            src = assign[item]
                            if src == -1:
                                continue
                            perm = list(range(len(bins)))
                            random.shuffle(perm)
                            for dst in perm:
                                if dst != src and loads[dst] + w[item] <= C:
                                    slack, slack2 = apply_relocate(item, src, dst, bins, loads, assign, slack, slack2)
                                    set_tabu(item, dst, it, tenure)
                                    moved += 1
                                    break
                        if moved >= 8:
                            break

        if (it & 0xFF) == 0 and now() >= time_limit:
            break

    # -------------------- Return best --------------------
    remove_empty_bins(best_bins, best_loads, best_assign)
    bw = [sum(w[i] for i in items) for items in best_bins]
    return {"packing": best_bins, "bin_weights": bw}
