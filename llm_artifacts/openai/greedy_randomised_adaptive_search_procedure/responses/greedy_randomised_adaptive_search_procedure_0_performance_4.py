import time
import random
from typing import List, Tuple, Optional, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.perf_counter()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # Feasibility guard
    if any(w > bin_capacity for w in weights):
        packing = [[i] for i in range(n)]
        bin_w = [weights[i] for i in range(n)]
        return {"packing": packing, "bin_weights": bin_w}

    # Use up to 100 seconds (no penalty for longer; hard stop at 100)
    hard_cap = 100.0
    T = min(time_limit, hard_cap)

    def now() -> float:
        return time.perf_counter()

    def elapsed() -> float:
        return now() - start

    # ---------------- Lower bounds (cheap but tighter than before) ----------------
    total_w = sum(weights)
    lb_sum = (total_w + bin_capacity - 1) // bin_capacity

    # LB large-items: if w > C/k then at most (k-1) items per bin
    def lb_large_items() -> int:
        best = 0
        C = bin_capacity
        for k in (2, 3, 4, 5, 6, 7):
            thr = C / k
            cnt = 0
            for w in weights:
                if w > thr:
                    cnt += 1
            # at most (k-1) such items in a bin
            best = max(best, (cnt + (k - 2)) // (k - 1))
        return best

    # LB for > C/2 items: they cannot pair together
    def lb_half() -> int:
        C = bin_capacity
        cnt = 0
        for w in weights:
            if w > C / 2:
                cnt += 1
        return cnt

    # Cheap Martello-Toth style bound:
    # sort decreasing; for each prefix compute how many bins forced by heavy items and leftover.
    # This is not full MT, but helps on instances with many large items.
    w_sorted = sorted(weights, reverse=True)
    prefix = [0] * (n + 1)
    for i, w in enumerate(w_sorted, 1):
        prefix[i] = prefix[i - 1] + w

    def lb_prefix() -> int:
        C = bin_capacity
        best = 0
        # only check first limited i for speed
        limit = min(n, 2000)
        for i in range(1, limit + 1):
            # i items, each must go somewhere; trivial
            # forced bins if these i items all > C/3? too expensive to test.
            # use: at most floor(C / w_i) items of size >= w_i per bin
            wi = w_sorted[i - 1]
            if wi == 0:
                continue
            max_per_bin = C // wi
            if max_per_bin <= 0:
                return n
            forced = (i + max_per_bin - 1) // max_per_bin
            # also total weight bound on prefix alone
            forced = max(forced, (prefix[i] + C - 1) // C)
            if forced > best:
                best = forced
        return best

    LB = max(lb_sum, lb_large_items(), lb_half(), lb_prefix())

    # ---------------- Metrics / comparison ----------------
    def compute_metrics(bin_w: List[int]) -> Tuple[int, int, int]:
        # Primary: number of bins
        # Secondary: maximize sum of squares of fills (encourages compactness)
        # Tertiary: maximize minimum fill (avoid very empty bins)
        s_sq = 0
        min_fill = bin_capacity
        for bw in bin_w:
            s_sq += bw * bw
            if bw < min_fill:
                min_fill = bw
        return (len(bin_w), -s_sq, -min_fill)

    def better(m1: Tuple[int, int, int], m2: Optional[Tuple[int, int, int]]) -> bool:
        return m2 is None or m1 < m2

    # ---------------- Core data helpers ----------------
    def assignment_from_packing(packing: List[List[int]]) -> List[int]:
        where = [-1] * n
        for b, items in enumerate(packing):
            for it in items:
                where[it] = b
        return where

    def rebuild_rem(bin_w: List[int]) -> List[int]:
        return [bin_capacity - bw for bw in bin_w]

    # ---------------- Construction (GRASP) ----------------
    # We deliberately DO NOT allow opening a new bin if any feasible existing placement exists.
    # This fixes a major quality leak in the previous code.

    MODE_BEST = 0
    MODE_FIRST = 1
    MODE_WORST = 2

    def construct(alpha: float, item_rcl: int, place_rcl: int, mode: int) -> Tuple[List[List[int]], List[int], List[int]]:
        items = list(range(n))
        items.sort(key=lambda i: (-weights[i], random.random()))

        # bins
        packing: List[List[int]] = []
        bin_w: List[int] = []
        rem: List[int] = []
        where = [-1] * n

        ops = 0
        TIME_CHECK_EVERY = 512

        remaining = items
        while remaining:
            ops += 1
            if (ops % TIME_CHECK_EVERY) == 0 and elapsed() >= T:
                break

            k = min(item_rcl, len(remaining))
            pos = random.randrange(k)
            it = remaining.pop(pos)
            w = weights[it]

            # gather feasible bins
            feas: List[Tuple[float, int]] = []
            if mode == MODE_FIRST:
                # consider prefix of bins
                for b in range(len(rem)):
                    rb = rem[b]
                    if rb >= w:
                        slack = rb - w
                        # slightly prioritize earlier bins but keep slack awareness
                        feas.append((float(b) + 1e-3 * slack + 1e-9 * random.random(), b))
                        if len(feas) >= max(8, place_rcl):
                            break
            else:
                for b, rb in enumerate(rem):
                    if rb >= w:
                        slack = rb - w
                        if mode == MODE_BEST:
                            score = slack
                        else:
                            score = -slack
                        feas.append((score + 1e-9 * random.random(), b))

            if feas:
                feas.sort(key=lambda x: x[0])
                m = min(place_rcl, len(feas))
                # GRASP thresholded selection
                cut = max(1, int((0.2 + alpha) * m))
                _, bsel = random.choice(feas[:cut])
                packing[bsel].append(it)
                bin_w[bsel] += w
                rem[bsel] -= w
                where[it] = bsel
            else:
                # open new bin (only when necessary)
                packing.append([it])
                bin_w.append(w)
                rem.append(bin_capacity - w)
                where[it] = len(packing) - 1

        # finish any items if time cut
        if remaining:
            for it in remaining:
                w = weights[it]
                best_b = -1
                best_s = None
                for b, rb in enumerate(rem):
                    if rb >= w:
                        s = rb - w
                        if best_s is None or s < best_s:
                            best_s = s
                            best_b = b
                if best_b == -1:
                    packing.append([it])
                    bin_w.append(w)
                    rem.append(bin_capacity - w)
                    where[it] = len(packing) - 1
                else:
                    packing[best_b].append(it)
                    bin_w[best_b] += w
                    rem[best_b] -= w
                    where[it] = best_b

        return packing, bin_w, where

    # ---------------- Local search (GRASP essential) ----------------
    # Focus: bin elimination with bounded reinsertion; with light ejection (swap-out) for hard cases.

    def try_reinsert_item_bestfit(rem: List[int], forbid_bin: int, w: int) -> int:
        best_b = -1
        best_s = None
        for b, rb in enumerate(rem):
            if b == forbid_bin:
                continue
            if rb >= w:
                s = rb - w
                if best_s is None or s < best_s:
                    best_s = s
                    best_b = b
        return best_b

    def try_empty_bin(
        packing: List[List[int]], bin_w: List[int], rem: List[int], where: List[int],
        bidx: int, max_eject: int
    ) -> bool:
        # Attempt to eliminate bin bidx by relocating its items.
        # Strategy:
        # 1) deterministic order: large-first
        # 2) greedy best-fit reinsertion
        # 3) if fail, allow limited ejections: move one (or two) items out of a destination bin to make room.

        items = packing[bidx][:]
        if not items:
            return False
        items.sort(key=lambda it: weights[it], reverse=True)

        # plan arrays
        tmp_rem = rem[:]
        moves: List[Tuple[int, int]] = []

        # quick helper to find ejection candidates from a bin
        def eject_from_bin(bin_items: List[int], needed: int) -> Optional[List[int]]:
            # find up to max_eject items whose total weight >= needed
            # use small candidates: take a few largest to reduce branching
            if not bin_items:
                return None
            cand = sorted(bin_items, key=lambda it: weights[it], reverse=True)[:10]
            # 1-item eject
            for it2 in cand:
                if weights[it2] >= needed:
                    return [it2]
            if max_eject >= 2:
                # 2-item eject
                L = len(cand)
                for i in range(L):
                    wi = weights[cand[i]]
                    for j in range(i + 1, L):
                        if wi + weights[cand[j]] >= needed:
                            return [cand[i], cand[j]]
            return None

        # record tentative placements including ejections
        # ejections are represented as (ejected_item, new_bin)
        ejection_moves: List[Tuple[int, int, int]] = []  # (ejected_item, from_bin, to_bin)

        for it in items:
            w = weights[it]
            dest = try_reinsert_item_bestfit(tmp_rem, bidx, w)
            if dest != -1:
                tmp_rem[dest] -= w
                moves.append((it, dest))
                continue

            # try ejection chain: pick a bin, eject small set, place it, then reinsert ejected elsewhere
            # choose candidate bins with largest slack (most promising to manipulate)
            order_bins = list(range(len(tmp_rem)))
            order_bins.sort(key=lambda b: tmp_rem[b], reverse=True)
            tried = 0
            success = False
            for b in order_bins:
                if b == bidx:
                    continue
                tried += 1
                if tried > 12:
                    break

                rb = tmp_rem[b]
                if rb >= w:
                    continue  # would have been found by best-fit already
                need = w - rb
                ej = eject_from_bin(packing[b], need)
                if ej is None:
                    continue

                # simulate ejecting ej from bin b
                tmp2 = tmp_rem[:]
                tmp2[b] += sum(weights[x] for x in ej)
                if tmp2[b] < w:
                    continue
                tmp2[b] -= w

                # try to reinsert ejected items elsewhere (best-fit, sequential)
                ok = True
                ej_places: List[Tuple[int, int]] = []
                for x in sorted(ej, key=lambda t: weights[t], reverse=True):
                    dx = try_reinsert_item_bestfit(tmp2, bidx, weights[x])
                    if dx == -1 or dx == b:
                        ok = False
                        break
                    tmp2[dx] -= weights[x]
                    ej_places.append((x, dx))

                if not ok:
                    continue

                # accept this simulated set
                tmp_rem = tmp2
                moves.append((it, b))
                for x, dx in ej_places:
                    ejection_moves.append((x, b, dx))
                success = True
                break

            if not success:
                return False

        # Apply moves atomically
        # First apply ejections (remove from their bins), then move target items, then place ejected elsewhere.
        # Remove duplicates in ejection list (can happen if same item considered twice; avoid)
        if ejection_moves:
            seen = set()
            filtered = []
            for x, fb, tb in ejection_moves:
                if x in seen:
                    continue
                seen.add(x)
                filtered.append((x, fb, tb))
            ejection_moves = filtered

        for x, fb, _ in ejection_moves:
            if x in packing[fb]:
                packing[fb].remove(x)
                bin_w[fb] -= weights[x]
                rem[fb] += weights[x]
                where[x] = -1

        # move items out of bidx
        for it, dest in moves:
            if it in packing[bidx]:
                packing[bidx].remove(it)
                bin_w[bidx] -= weights[it]
                rem[bidx] += weights[it]
                where[it] = -1
            packing[dest].append(it)
            bin_w[dest] += weights[it]
            rem[dest] -= weights[it]
            where[it] = dest

        # place ejected items
        for x, _, tb in ejection_moves:
            if where[x] != -1:
                continue
            packing[tb].append(x)
            bin_w[tb] += weights[x]
            rem[tb] -= weights[x]
            where[x] = tb

        if packing[bidx]:
            return False

        # remove empty bin and fix indices
        packing.pop(bidx)
        bin_w.pop(bidx)
        rem.pop(bidx)
        for it in range(n):
            b = where[it]
            if b > bidx:
                where[it] = b - 1

        return True

    def local_search(
        packing: List[List[int]], bin_w: List[int], where: List[int],
        passes: int, tries_per_pass: int, max_eject: int
    ) -> Tuple[List[List[int]], List[int], List[int]]:
        rem = rebuild_rem(bin_w)
        TIME_MASK = 63

        for p in range(passes):
            if elapsed() >= T:
                break

            improved = False

            # prioritize emptiest / lightest bins
            order = list(range(len(packing)))
            order.sort(key=lambda b: (bin_w[b], len(packing[b])))
            target_bins = order[: min(len(order), tries_per_pass)]

            for t, bidx in enumerate(target_bins):
                if (t & TIME_MASK) == 0 and elapsed() >= T:
                    break
                if bidx >= len(packing) or len(packing) <= 1:
                    continue
                if try_empty_bin(packing, bin_w, rem, where, bidx, max_eject=max_eject):
                    improved = True
                    break

            if not improved:
                break

        return packing, bin_w, where

    # ---------------- Elite set + Path relinking (GRASP enhancement) ----------------
    def signature(bin_w: List[int]) -> Tuple[int, ...]:
        return tuple(sorted(bin_w, reverse=True))

    def add_elite(
        elite: List[Tuple[Tuple[int, int, int], Tuple[int, ...], List[List[int]], List[int]]],
        m: Tuple[int, int, int], packing: List[List[int]], bin_w: List[int],
        E: int
    ) -> None:
        sig = signature(bin_w)
        for _, s, _, _ in elite:
            if s == sig:
                return
        entry = (m, sig, [b[:] for b in packing], bin_w[:])
        elite.append(entry)
        elite.sort(key=lambda x: x[0])
        if len(elite) > E:
            elite.pop()

    def relabel_bins_by_load(packing: List[List[int]], bin_w: List[int]) -> Tuple[List[int], List[int]]:
        order = list(range(len(bin_w)))
        order.sort(key=lambda b: (bin_w[b], len(packing[b])), reverse=True)
        pos = [0] * len(order)
        for r, b in enumerate(order):
            pos[b] = r
        return pos, order

    def path_relink(
        pack_a: List[List[int]], bw_a: List[int],
        pack_b: List[List[int]], bw_b: List[int],
        max_moves: int
    ) -> Optional[Tuple[List[List[int]], List[int]]]:
        if elapsed() >= T:
            return None

        packing = [x[:] for x in pack_a]
        bin_w = bw_a[:]
        rem = rebuild_rem(bin_w)
        where = assignment_from_packing(packing)

        pos_a, order_a = relabel_bins_by_load(packing, bin_w)
        pos_b, _order_b = relabel_bins_by_load(pack_b, bw_b)
        where_b = assignment_from_packing(pack_b)

        target_rank = [-1] * n
        for it in range(n):
            bb = where_b[it]
            if bb >= 0 and bb < len(pos_b):
                target_rank[it] = pos_b[bb]

        cand = [it for it in range(n) if target_rank[it] >= 0]
        random.shuffle(cand)

        best_m = compute_metrics(bin_w)
        best_pack = None
        best_bw = None

        TIME_MASK = 31
        moves = 0

        while cand and moves < max_moves:
            moves += 1
            if (moves & TIME_MASK) == 0 and elapsed() >= T:
                break

            it = cand.pop()
            cur_b = where[it]
            if cur_b < 0:
                continue

            if moves % 25 == 0:
                pos_a, order_a = relabel_bins_by_load(packing, bin_w)

            tr = target_rank[it]
            if tr < 0:
                continue
            if pos_a[cur_b] == tr:
                continue

            if not order_a:
                continue

            desired_idx = min(tr, len(order_a) - 1)
            w = weights[it]

            dest = -1
            # try a few ranks around desired
            for delta in (0, 1, -1, 2, -2, 3, -3, 4, -4):
                rr = desired_idx + delta
                if 0 <= rr < len(order_a):
                    b2 = order_a[rr]
                    if b2 != cur_b and rem[b2] >= w:
                        dest = b2
                        break
            if dest == -1:
                continue

            # apply move
            packing[cur_b].remove(it)
            bin_w[cur_b] -= w
            rem[cur_b] += w

            packing[dest].append(it)
            bin_w[dest] += w
            rem[dest] -= w
            where[it] = dest

            # remove empty bin
            if not packing[cur_b]:
                packing.pop(cur_b)
                bin_w.pop(cur_b)
                rem.pop(cur_b)
                for j in range(n):
                    if where[j] > cur_b:
                        where[j] -= 1

            m = compute_metrics(bin_w)
            if m < best_m:
                best_m = m
                best_pack = [b[:] for b in packing]
                best_bw = bin_w[:]
                if best_m[0] == LB:
                    break

        if best_pack is None:
            return None
        return best_pack, best_bw

    # ---------------- Main GRASP loop ----------------
    # Fixed iterations but time-checked; we try to saturate the time budget.
    MAX_ITER = 2_000_000
    E = 20

    best_packing: List[List[int]] = []
    best_bin_w: List[int] = []
    best_m: Optional[Tuple[int, int, int]] = None

    elite: List[Tuple[Tuple[int, int, int], Tuple[int, ...], List[List[int]], List[int]]] = []

    alpha = 0.10
    no_improve = 0

    # base RCL sizes
    base_item_rcl = min(40, max(8, int(n ** 0.5)))
    base_place_rcl = 30

    for it in range(MAX_ITER):
        if (it & 1023) == 0 and elapsed() >= T:
            break

        # Diversify under stagnation
        if no_improve >= 250:
            alpha = min(0.55, alpha + 0.05)
        else:
            alpha = max(0.05, alpha * 0.9995)

        item_rcl = base_item_rcl
        place_rcl = base_place_rcl
        if no_improve >= 250:
            item_rcl = min(60, max(item_rcl, 20))
            place_rcl = min(45, max(place_rcl, 35))

        # mode mix
        r = random.random()
        if no_improve < 120:
            mode = MODE_BEST if r < 0.85 else MODE_FIRST
        else:
            if r < 0.60:
                mode = MODE_BEST
            elif r < 0.85:
                mode = MODE_FIRST
            else:
                mode = MODE_WORST

        packing, bin_w, where = construct(alpha, item_rcl=item_rcl, place_rcl=place_rcl, mode=mode)
        m0 = compute_metrics(bin_w)

        close = (m0[0] <= LB + 1)
        passes = 6 if close else 3
        tries = 26 if close else 16
        max_eject = 2 if close else (1 if n > 2000 else 2)

        packing, bin_w, where = local_search(packing, bin_w, where, passes=passes, tries_per_pass=tries, max_eject=max_eject)
        m = compute_metrics(bin_w)

        if better(m, best_m):
            best_m = m
            best_packing = [b[:] for b in packing]
            best_bin_w = bin_w[:]
            no_improve = 0
            add_elite(elite, m, packing, bin_w, E)

            # intensify with a couple of relinkings
            if elite and elapsed() < T:
                for _ in range(min(3, len(elite))):
                    em, esig, epack, ebw = random.choice(elite)
                    if esig == signature(best_bin_w):
                        continue
                    pr = path_relink(best_packing, best_bin_w, epack, ebw, max_moves=260)
                    if pr is None or elapsed() >= T:
                        continue
                    pr_pack, pr_bw = pr
                    pr_where = assignment_from_packing(pr_pack)
                    pr_pack, pr_bw, pr_where = local_search(pr_pack, pr_bw, pr_where, passes=3, tries_per_pass=18, max_eject=2)
                    pr_m = compute_metrics(pr_bw)
                    add_elite(elite, pr_m, pr_pack, pr_bw, E)
                    if better(pr_m, best_m):
                        best_m = pr_m
                        best_packing = [b[:] for b in pr_pack]
                        best_bin_w = pr_bw[:]
                        if best_m[0] == LB:
                            break

            if best_m[0] == LB:
                break
        else:
            no_improve += 1
            add_elite(elite, m, packing, bin_w, E)

            # periodic relinking during stagnation
            if elite and (no_improve >= 180) and (it % 80 == 0) and elapsed() < T:
                em, esig, epack, ebw = random.choice(elite)
                pr1 = path_relink(packing, bin_w, epack, ebw, max_moves=220)
                pr2 = path_relink(epack, ebw, packing, bin_w, max_moves=220)
                for pr in (pr1, pr2):
                    if pr is None or elapsed() >= T:
                        continue
                    pr_pack, pr_bw = pr
                    pr_where = assignment_from_packing(pr_pack)
                    pr_pack, pr_bw, pr_where = local_search(pr_pack, pr_bw, pr_where, passes=2, tries_per_pass=14, max_eject=2)
                    pr_m = compute_metrics(pr_bw)
                    add_elite(elite, pr_m, pr_pack, pr_bw, E)
                    if better(pr_m, best_m):
                        best_m = pr_m
                        best_packing = [b[:] for b in pr_pack]
                        best_bin_w = pr_bw[:]
                        no_improve = 0
                        if best_m[0] == LB:
                            break

    # Fallback
    if best_m is None:
        best_packing, best_bin_w, _ = construct(0.10, item_rcl=min(30, n), place_rcl=25, mode=MODE_BEST)

    final_bin_w = [sum(weights[i] for i in b) for b in best_packing]
    return {"packing": best_packing, "bin_weights": final_bin_w}
