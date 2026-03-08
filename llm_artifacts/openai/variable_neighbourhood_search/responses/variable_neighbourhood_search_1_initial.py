import time
import random
from typing import List, Tuple, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = bin_capacity
    n = len(weights)
    start = time.perf_counter()
    deadline = start + max(0.0, time_limit)

    # ---------- Helper functions ----------
    def time_up() -> bool:
        return time.perf_counter() >= deadline

    def compute_obj(bin_w: List[int]) -> Tuple[int, int]:
        # (bins, sum squared slack)
        s2 = 0
        for w in bin_w:
            slack = C - w
            s2 += slack * slack
        return (len(bin_w), s2)

    def normalize(packing: List[List[int]], bin_w: List[int]) -> Tuple[List[List[int]], List[int]]:
        # Remove empty bins
        new_p, new_w = [], []
        for b, w in zip(packing, bin_w):
            if b:
                new_p.append(b)
                new_w.append(w)
        return new_p, new_w

    def build_mapping(packing: List[List[int]]) -> List[Tuple[int, int]]:
        # item -> (bin, pos)
        loc = [(-1, -1)] * n
        for bi, b in enumerate(packing):
            for pi, it in enumerate(b):
                loc[it] = (bi, pi)
        return loc

    def bfd_initial() -> Tuple[List[List[int]], List[int]]:
        # Best-Fit Decreasing with mild randomization among best bins
        order = list(range(n))
        order.sort(key=lambda i: weights[i], reverse=True)
        packing: List[List[int]] = []
        bin_w: List[int] = []

        for it in order:
            w = weights[it]
            best_bins = []
            best_rem = None
            for bi, bw in enumerate(bin_w):
                if bw + w <= C:
                    rem = C - (bw + w)
                    if best_rem is None or rem < best_rem:
                        best_rem = rem
                        best_bins = [bi]
                    elif rem == best_rem:
                        best_bins.append(bi)
            if best_bins:
                # random tie break
                bi = random.choice(best_bins)
                packing[bi].append(it)
                bin_w[bi] += w
            else:
                packing.append([it])
                bin_w.append(w)
        return packing, bin_w

    def try_close_bins(packing: List[List[int]], bin_w: List[int], loc: List[Tuple[int, int]]) -> None:
        # Attempt to eliminate bins by re-inserting their items elsewhere (greedy), starting from lightest bins
        idx = list(range(len(bin_w)))
        idx.sort(key=lambda b: bin_w[b])
        for b in idx:
            if time_up():
                return
            if b >= len(packing) or not packing[b]:
                continue
            items = packing[b][:]
            # Try to move all items
            feasible = True
            # Remove items from b temporarily
            packing[b].clear()
            old_w = bin_w[b]
            bin_w[b] = 0

            # Reinsert each item best-fit
            reinserts = []
            for it in items:
                w = weights[it]
                best = -1
                best_rem = None
                for bj, bw in enumerate(bin_w):
                    if bj == b or not packing[bj]:
                        continue
                    if bw + w <= C:
                        rem = C - (bw + w)
                        if best_rem is None or rem < best_rem:
                            best_rem = rem
                            best = bj
                if best == -1:
                    feasible = False
                    break
                reinserts.append((it, best))
                bin_w[best] += w
                packing[best].append(it)
                loc[it] = (best, len(packing[best]) - 1)

            if not feasible:
                # rollback: restore bin b and remove any inserted items
                for it, bj in reinserts:
                    # remove it from bj
                    pb, pos = loc[it]
                    # pos might be stale if swaps happened; remove by search (small)
                    try:
                        packing[bj].remove(it)
                    except ValueError:
                        pass
                    bin_w[bj] -= weights[it]
                packing[b] = items
                bin_w[b] = old_w
                for pi, it in enumerate(items):
                    loc[it] = (b, pi)
            else:
                # bin b eliminated
                pass

        # cleanup empties + rebuild loc for safety
        packing[:], bin_w[:] = normalize(packing, bin_w)
        new_loc = build_mapping(packing)
        loc[:] = new_loc

    def candidate_bins_for_item(w: int, bin_w: List[int], exclude: int = -1, sample: int = 24) -> List[int]:
        m = len(bin_w)
        if m <= 1:
            return [0] if m == 1 else []
        # include a few best by remaining capacity (tightest fit) + random sample
        candidates = []
        # random sample
        if m > sample:
            seen = set()
            for _ in range(sample):
                bj = random.randrange(m)
                if bj == exclude:
                    continue
                if bj in seen:
                    continue
                seen.add(bj)
                candidates.append(bj)
        else:
            candidates = [bj for bj in range(m) if bj != exclude]

        # add top tight fits among all (cheap: scan and keep small list)
        best = []  # list of (rem, bj)
        for bj, bw in enumerate(bin_w):
            if bj == exclude:
                continue
            if bw + w <= C:
                rem = C - (bw + w)
                if len(best) < 8:
                    best.append((rem, bj))
                    best.sort()
                else:
                    if rem < best[-1][0]:
                        best[-1] = (rem, bj)
                        best.sort()
        for _, bj in best:
            if bj not in candidates:
                candidates.append(bj)
        return candidates

    # ---------- Local search (VND) ----------
    def vnd(packing: List[List[int]], bin_w: List[int], loc: List[Tuple[int, int]]) -> None:
        # Descent over neighborhoods: relocate, swap, 2-1.
        improved = True
        while improved and not time_up():
            improved = False

            # 1) Relocate single item
            for it in range(n):
                if time_up():
                    return
                bi, _ = loc[it]
                if bi < 0:
                    continue
                w = weights[it]
                # Try to move into other bins to enable emptying or reduce slack^2
                best_move = None
                cur_bins, cur_s2 = compute_obj(bin_w)
                for bj in candidate_bins_for_item(w, bin_w, exclude=bi):
                    if bj == bi or not packing[bj]:
                        continue
                    if bin_w[bj] + w > C:
                        continue
                    # delta objective
                    # bins count changes if source becomes empty
                    src_empty_after = (len(packing[bi]) == 1)
                    new_bins = cur_bins - (1 if src_empty_after else 0)
                    # compute delta s2 exactly for involved bins
                    s2_delta = 0
                    # src bin
                    s_old = (C - bin_w[bi])
                    s_new = (C - (bin_w[bi] - w))
                    s2_delta += s_new * s_new - s_old * s_old
                    # dst bin
                    d_old = (C - bin_w[bj])
                    d_new = (C - (bin_w[bj] + w))
                    s2_delta += d_new * d_new - d_old * d_old
                    new_s2 = cur_s2 + s2_delta
                    if (new_bins, new_s2) < (cur_bins, cur_s2):
                        best_move = (it, bi, bj)
                        break

                if best_move is not None:
                    it, bi, bj = best_move
                    # apply
                    packing[bi].remove(it)
                    bin_w[bi] -= w
                    packing[bj].append(it)
                    bin_w[bj] += w
                    # cleanup possible empty
                    packing[:], bin_w[:] = normalize(packing, bin_w)
                    loc[:] = build_mapping(packing)
                    improved = True
                    break

            if improved:
                continue

            # 2) Swap 1-1
            m = len(packing)
            # pick some bins to examine
            bins_to_check = list(range(m))
            random.shuffle(bins_to_check)
            bins_to_check = bins_to_check[: min(m, 16)]
            base_obj = compute_obj(bin_w)
            for bi in bins_to_check:
                if time_up():
                    return
                for bj in bins_to_check:
                    if bj <= bi:
                        continue
                    if not packing[bi] or not packing[bj]:
                        continue
                    # sample some items from each
                    items_i = packing[bi]
                    items_j = packing[bj]
                    samp_i = items_i if len(items_i) <= 6 else random.sample(items_i, 6)
                    samp_j = items_j if len(items_j) <= 6 else random.sample(items_j, 6)
                    for a in samp_i:
                        wa = weights[a]
                        for b in samp_j:
                            wb = weights[b]
                            new_wi = bin_w[bi] - wa + wb
                            new_wj = bin_w[bj] - wb + wa
                            if new_wi > C or new_wj > C:
                                continue
                            # bin count unchanged
                            # delta s2
                            s2 = base_obj[1]
                            s2_delta = 0
                            si_old = C - bin_w[bi]
                            sj_old = C - bin_w[bj]
                            si_new = C - new_wi
                            sj_new = C - new_wj
                            s2_delta += si_new * si_new - si_old * si_old
                            s2_delta += sj_new * sj_new - sj_old * sj_old
                            if (base_obj[0], s2 + s2_delta) < base_obj:
                                # apply swap
                                packing[bi].remove(a)
                                packing[bj].remove(b)
                                packing[bi].append(b)
                                packing[bj].append(a)
                                bin_w[bi] = new_wi
                                bin_w[bj] = new_wj
                                loc[:] = build_mapping(packing)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

            # 3) 2-1 ejection (move two items from A into B by moving one from B into A)
            m = len(packing)
            if m >= 2:
                base_obj = compute_obj(bin_w)
                bins = list(range(m))
                random.shuffle(bins)
                bins = bins[: min(m, 14)]
                for bi in bins:
                    if time_up():
                        return
                    if len(packing[bi]) < 2:
                        continue
                    for bj in bins:
                        if bj == bi:
                            continue
                        if not packing[bj]:
                            continue
                        # sample candidates
                        Ai = packing[bi]
                        Bj = packing[bj]
                        sampA = Ai if len(Ai) <= 7 else random.sample(Ai, 7)
                        sampB = Bj if len(Bj) <= 7 else random.sample(Bj, 7)
                        # precompute pairs from A
                        pairs = []
                        for x in range(len(sampA)):
                            for y in range(x + 1, len(sampA)):
                                a1, a2 = sampA[x], sampA[y]
                                pairs.append((a1, a2, weights[a1] + weights[a2]))
                        random.shuffle(pairs)
                        pairs = pairs[:20]
                        for (a1, a2, wsum) in pairs:
                            for b in sampB:
                                wb = weights[b]
                                # After move: B receives a1,a2 and loses b; A receives b and loses a1,a2
                                new_wA = bin_w[bi] - wsum + wb
                                new_wB = bin_w[bj] - wb + wsum
                                if new_wA > C or new_wB > C:
                                    continue
                                # bins unchanged
                                s2 = base_obj[1]
                                si_old = C - bin_w[bi]
                                sj_old = C - bin_w[bj]
                                si_new = C - new_wA
                                sj_new = C - new_wB
                                s2_new = s2 + (si_new * si_new - si_old * si_old) + (sj_new * sj_new - sj_old * sj_old)
                                if (base_obj[0], s2_new) < base_obj:
                                    # apply
                                    packing[bi].remove(a1)
                                    packing[bi].remove(a2)
                                    packing[bj].remove(b)
                                    packing[bi].append(b)
                                    packing[bj].append(a1)
                                    packing[bj].append(a2)
                                    bin_w[bi] = new_wA
                                    bin_w[bj] = new_wB
                                    loc[:] = build_mapping(packing)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

    # ---------- Shaking (diversification) ----------
    def shake(cur_p: List[List[int]], cur_w: List[int], k: int) -> Tuple[List[List[int]], List[int]]:
        # Remove r items and reinsert greedily with randomness.
        p = [b[:] for b in cur_p]
        bw = cur_w[:]
        loc = build_mapping(p)

        m = len(p)
        if n == 0:
            return p, bw

        # choose number of removed items
        r = min(n, 2 + k + (k // 2))
        # bias towards taking items from lighter bins to encourage elimination, but add randomness
        bins = list(range(m))
        bins.sort(key=lambda b: bw[b])
        pool = []
        for b in bins[: max(1, m // 2)]:
            pool.extend(p[b])
        if len(pool) < r:
            pool = list(range(n))
        removed = random.sample(pool, r)

        # remove
        for it in removed:
            bi, _ = loc[it]
            if bi < 0:
                continue
            try:
                p[bi].remove(it)
            except ValueError:
                continue
            bw[bi] -= weights[it]

        p, bw = normalize(p, bw)

        # reinsert with best-fit, randomized among top candidates; sometimes open new bin.
        for it in removed:
            if time_up():
                break
            w = weights[it]
            best = []  # (rem, bi)
            for bi, bww in enumerate(bw):
                if bww + w <= C:
                    rem = C - (bww + w)
                    best.append((rem, bi))
            if best:
                best.sort(key=lambda x: x[0])
                top = best[: min(4, len(best))]
                _, chosen = random.choice(top)
                p[chosen].append(it)
                bw[chosen] += w
            else:
                p.append([it])
                bw.append(w)

        return p, bw

    # ---------- Main VNS loop ----------
    packing, bin_w = bfd_initial()
    packing, bin_w = normalize(packing, bin_w)
    loc = build_mapping(packing)

    # Quick improvement by attempting to close bins
    try_close_bins(packing, bin_w, loc)
    vnd(packing, bin_w, loc)

    best_p = [b[:] for b in packing]
    best_w = bin_w[:]
    best_obj = compute_obj(best_w)

    # VNS parameters
    k_max = 10
    iters = 0
    # fixed iteration cap as a safeguard; time is primary stop
    max_iters = 10_000_000

    while iters < max_iters and not time_up():
        iters += 1
        k = 1
        cur_p = [b[:] for b in packing]
        cur_w = bin_w[:]
        cur_loc = build_mapping(cur_p)

        while k <= k_max and not time_up():
            # Shaking
            sp, sw = shake(cur_p, cur_w, k)
            sp, sw = normalize(sp, sw)
            sloc = build_mapping(sp)

            # Local search (VND)
            vnd(sp, sw, sloc)
            try_close_bins(sp, sw, sloc)

            sobj = compute_obj(sw)
            if sobj < best_obj:
                best_obj = sobj
                best_p = [b[:] for b in sp]
                best_w = sw[:]

            # Acceptance: if improved over current, move to it and restart k
            cobj = compute_obj(cur_w)
            if sobj < cobj:
                cur_p, cur_w, cur_loc = sp, sw, sloc
                k = 1
            else:
                k += 1

        # Update incumbent to best found in this outer iteration (intensification)
        # Use best overall as the base if it is better than current packing.
        if best_obj < compute_obj(bin_w):
            packing = [b[:] for b in best_p]
            bin_w = best_w[:]
        else:
            packing = cur_p
            bin_w = cur_w

        # occasional perturbation of k_max based on progress
        if iters % 200 == 0 and not time_up():
            # adapt k_max mildly
            k_max = 8 + (iters // 200) % 6

        # periodic cleanup
        if iters % 50 == 0:
            packing, bin_w = normalize(packing, bin_w)

    # Ensure final best is returned
    packing = best_p
    bin_w = best_w

    # Final normalization and aligned weights
    packing, bin_w = normalize(packing, bin_w)

    return {"packing": packing, "bin_weights": bin_w}
