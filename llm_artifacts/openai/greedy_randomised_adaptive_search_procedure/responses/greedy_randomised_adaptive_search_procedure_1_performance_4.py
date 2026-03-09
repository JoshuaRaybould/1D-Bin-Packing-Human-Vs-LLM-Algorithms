import time
import random
from typing import List, Tuple, Dict


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    items = list(range(n))

    def time_exceeded() -> bool:
        return (time.time() - start) >= time_limit

    def solution_key(packing: List[List[int]], loads: List[int]) -> Tuple[int, int]:
        # Minimize #bins, then total waste
        waste = 0
        C = bin_capacity
        for w in loads:
            waste += (C - w) if w <= C else 0
        return (len(packing), waste)

    def deep_copy(p: List[List[int]], w: List[int]) -> Tuple[List[List[int]], List[int]]:
        return [b[:] for b in p], w[:]

    def compact(p: List[List[int]], loads: List[int], item_bin: List[int]) -> Tuple[List[List[int]], List[int]]:
        new_p: List[List[int]] = []
        new_w: List[int] = []
        remap = {}
        for bi, b in enumerate(p):
            if b:
                remap[bi] = len(new_p)
                new_p.append(b)
                new_w.append(loads[bi])
        for it in range(n):
            bi = item_bin[it]
            if bi != -1:
                item_bin[it] = remap.get(bi, -1)
        return new_p, new_w

    # -------------------- Baseline: FFD (deterministic) --------------------
    def ffd(order: List[int]) -> Tuple[List[List[int]], List[int]]:
        p: List[List[int]] = []
        loads: List[int] = []
        C = bin_capacity
        for it in order:
            w = weights[it]
            if w > C:
                p.append([it])
                loads.append(w)
                continue
            placed = False
            for bi in range(len(p)):
                if loads[bi] + w <= C:
                    p[bi].append(it)
                    loads[bi] += w
                    placed = True
                    break
            if not placed:
                p.append([it])
                loads.append(w)
        return p, loads

    base_order = items[:]
    base_order.sort(key=lambda i: weights[i], reverse=True)
    best_p, best_w = ffd(base_order)
    best_key = solution_key(best_p, best_w)

    # -------------------- GRASP construction --------------------
    # Build bins using best-fit, but choose among an RCL of best bins.
    # Also randomize item order slightly (BFD-like) to diversify.
    def construct(alpha: float, item_rcl_prob: float) -> Tuple[List[List[int]], List[int]]:
        C = bin_capacity

        # Randomized decreasing order (controlled noise)
        order = items[:]
        # Noise proportional to weight to keep mostly-decreasing
        order.sort(key=lambda i: weights[i] + (random.random() - 0.5) * 0.12 * weights[i], reverse=True)

        # Optional RCL on next item among top-k in order (classic GRASP flavor)
        if random.random() < item_rcl_prob and n >= 10:
            # Make a new order by repeatedly selecting among top-k largest remaining
            rem = order
            order2 = []
            k = 6
            while rem:
                kk = k if len(rem) > k else len(rem)
                pick = random.randrange(kk)
                order2.append(rem.pop(pick))
            order = order2

        p: List[List[int]] = []
        loads: List[int] = []

        for it in order:
            w = weights[it]
            if w > C:
                p.append([it])
                loads.append(w)
                continue

            # Collect feasible bins, scored by slack after placement (best-fit)
            cand = []  # (slack_after, load_before, bi)
            for bi, bw in enumerate(loads):
                if bw + w <= C:
                    cand.append((C - (bw + w), bw, bi))

            if not cand:
                p.append([it])
                loads.append(w)
                continue

            cand.sort(key=lambda x: (x[0], -x[1]))
            min_slack = cand[0][0]
            max_slack = cand[-1][0]
            thr = min_slack + alpha * (max_slack - min_slack)
            # RCL = bins whose slack is within threshold
            rcl = [c for c in cand if c[0] <= thr]
            # Mild bias: among RCL, pick one of the best few more often
            if len(rcl) > 4 and random.random() < 0.75:
                chosen = rcl[random.randrange(min(4, len(rcl)))]
            else:
                chosen = random.choice(rcl)

            bi = chosen[2]
            p[bi].append(it)
            loads[bi] += w

        return p, loads

    # -------------------- Local search (essential GRASP component) --------------------
    # Primary goal: reduce #bins by emptying bins (classic for bin packing GRASP).
    # We do repeated bin-emptying attempts using:
    #   - direct relocations (best-fit)
    #   - limited-depth ejection (move an item out of destination to make room)
    # Plus a small enabling phase (relocate/swap) to create tighter bins.
    def local_search(p: List[List[int]], loads: List[int]) -> Tuple[List[List[int]], List[int]]:
        C = bin_capacity

        # item->bin map
        item_bin = [-1] * n
        for bi, b in enumerate(p):
            for it in b:
                if 0 <= it < n:
                    item_bin[it] = bi

        # Helper: attempt to move item it from src to dst (assume feasible)
        def do_move(it: int, src: int, dst: int):
            w = weights[it]
            p[src].remove(it)
            loads[src] -= w
            p[dst].append(it)
            loads[dst] += w
            item_bin[it] = dst

        # Find best-fit destination bin for an item (excluding src)
        def best_fit_bin(src: int, it: int) -> int:
            w = weights[it]
            best = -1
            best_slack = None
            for bi, bw in enumerate(loads):
                if bi == src or not p[bi]:
                    continue
                nb = bw + w
                if nb <= C:
                    slack = C - nb
                    if best_slack is None or slack < best_slack:
                        best_slack = slack
                        best = bi
                        if slack == 0:
                            break
            return best

        # Limited ejection: try to place it into dst by moving out one item from dst to some other bin.
        def try_ejection(src: int, it: int, dst: int, depth: int) -> bool:
            # depth counts remaining ejection levels
            w_it = weights[it]
            if loads[dst] + w_it <= C:
                do_move(it, src, dst)
                return True
            if depth <= 0:
                return False

            # Need to free at least need = loads[dst] + w_it - C
            need = loads[dst] + w_it - C
            # Candidates to eject: items with weight >= need, try heavier first
            # limit scanning
            cand = p[dst][:]
            cand.sort(key=lambda x: weights[x], reverse=True)

            # Try a few ejection candidates
            trials = 0
            for out in cand:
                w_out = weights[out]
                if w_out < need:
                    continue

                # Find a new place for 'out'
                # Prefer best-fit, but allow random among a small RCL of feasible bins
                feasible = []  # (slack, bi)
                for bi, bw in enumerate(loads):
                    if bi == dst or bi == src or not p[bi]:
                        continue
                    if bw + w_out <= C:
                        feasible.append((C - (bw + w_out), bi))
                if not feasible:
                    continue
                feasible.sort(key=lambda x: x[0])
                rcl = feasible[: min(5, len(feasible))]
                _, newb = rcl[random.randrange(len(rcl))]

                # Execute ejection move out of dst -> newb
                p[dst].remove(out)
                loads[dst] -= w_out
                p[newb].append(out)
                loads[newb] += w_out
                item_bin[out] = newb

                # Now try to place it
                if loads[dst] + w_it <= C:
                    do_move(it, src, dst)
                    return True

                # Still not enough, recurse one more level by trying to eject again
                if try_ejection(src, it, dst, depth - 1):
                    return True

                # Rollback ejection if unsuccessful
                p[newb].remove(out)
                loads[newb] -= w_out
                p[dst].append(out)
                loads[dst] += w_out
                item_bin[out] = dst

                trials += 1
                if trials >= 6:
                    break

            return False

        # Attempt to empty a bin completely
        def try_empty_bin(src: int, max_depth: int) -> bool:
            if not p[src]:
                return False
            # Work on a copy of items ordering (large-first improves feasibility)
            src_items = p[src][:]
            src_items.sort(key=lambda x: weights[x], reverse=True)

            moved_stack = []  # record moves for rollback: ('move', it, old, new) and ('ej', out, old, new)

            # For rollback we need to capture *all* modifications. Our ejection routine already modifies bins;
            # to keep it simple, we only use ejection in a way that we can rollback by snapshotting loads and bins.
            # Snapshot the involved bins (cheap because src bin small-ish typically); but to be safe, snapshot all.
            # Since this is called selectively, OK.
            p_snap = [b[:] for b in p]
            w_snap = loads[:]
            ib_snap = item_bin[:]

            for it in src_items:
                # direct best-fit
                dst = best_fit_bin(src, it)
                if dst != -1:
                    do_move(it, src, dst)
                    continue

                # ejection attempts: try a few candidate dst bins (tight bins first)
                # Build candidate dst bins by smallest slack of current load (tighter bins better)
                dst_cand = []
                w_it = weights[it]
                for bi, bw in enumerate(loads):
                    if bi == src or not p[bi]:
                        continue
                    if bw < C:  # only bins that can possibly accept after ejection
                        slack_now = C - bw
                        # smaller slack means tighter bin; prioritize those
                        dst_cand.append((slack_now, bi))
                dst_cand.sort(key=lambda x: x[0])
                dst_list = [bi for _, bi in dst_cand[: min(10, len(dst_cand))]]
                random.shuffle(dst_list)

                placed = False
                for dst2 in dst_list:
                    if try_ejection(src, it, dst2, max_depth):
                        placed = True
                        break
                if not placed:
                    # rollback and fail
                    for i in range(len(p)):
                        p[i] = p_snap[i]
                    for i in range(len(loads)):
                        loads[i] = w_snap[i]
                    for i in range(n):
                        item_bin[i] = ib_snap[i]
                    return False

                if time_exceeded():
                    break

            return (len(p[src]) == 0)

        # Enabling phase: small number of best-improving reloc/swap moves to tighten bins
        def tighten_phase(iter_cap: int) -> bool:
            if len(p) <= 1:
                return False
            improved = False

            # sample bins biased to those with higher slack
            B = len(p)
            idx = list(range(B))
            idx.sort(key=lambda bi: (C - loads[bi]) if p[bi] else C, reverse=True)
            sample_bins = idx[: min(16, B)]
            if len(sample_bins) >= 6:
                # add some randomness
                rest = idx[min(16, B):]
                random.shuffle(rest)
                sample_bins += rest[: min(6, len(rest))]

            moves_done = 0
            while moves_done < iter_cap and not time_exceeded():
                best_delta = 0
                best = None

                # relocations (make a destination bin tighter)
                for src in sample_bins:
                    if not p[src]:
                        continue
                    # prefer moving small/medium items out of slacky bins
                    its = p[src][:]
                    its.sort(key=lambda x: weights[x])
                    for it in its[: min(10, len(its))]:
                        w_it = weights[it]
                        for dst in sample_bins:
                            if dst == src or not p[dst]:
                                continue
                            if loads[dst] + w_it <= C:
                                old = (C - loads[dst])
                                new = (C - (loads[dst] + w_it))
                                delta = old - new
                                if delta > best_delta:
                                    best_delta = delta
                                    best = ('reloc', it, src, dst)

                # swaps (tighten two bins)
                for b1 in sample_bins:
                    if not p[b1]:
                        continue
                    for b2 in sample_bins:
                        if b2 <= b1 or not p[b2]:
                            continue
                        a1 = p[b1][: min(8, len(p[b1]))]
                        a2 = p[b2][: min(8, len(p[b2]))]
                        for it1 in a1:
                            w1 = weights[it1]
                            for it2 in a2:
                                w2 = weights[it2]
                                nb1 = loads[b1] - w1 + w2
                                nb2 = loads[b2] - w2 + w1
                                if nb1 <= C and nb2 <= C:
                                    delta = (abs((C - loads[b1]) - (C - nb1)) + abs((C - loads[b2]) - (C - nb2)))
                                    if delta > best_delta:
                                        best_delta = delta
                                        best = ('swap', it1, b1, it2, b2)

                if best is None or best_delta <= 0:
                    break

                if best[0] == 'reloc':
                    _, it, src, dst = best
                    do_move(it, src, dst)
                    if not p[src]:
                        # compact immediately
                        nonlocal_p, nonlocal_w = p, loads
                        # compact updates item_bin
                        new_p, new_w = compact(nonlocal_p, nonlocal_w, item_bin)
                        p[:] = new_p
                        loads[:] = new_w
                        # recompute samples quickly
                        B = len(p)
                        idx = list(range(B))
                        idx.sort(key=lambda bi: (C - loads[bi]) if p[bi] else C, reverse=True)
                        sample_bins = idx[: min(16, B)]
                    improved = True
                else:
                    _, it1, b1, it2, b2 = best
                    w1, w2 = weights[it1], weights[it2]
                    p[b1].remove(it1)
                    p[b2].remove(it2)
                    p[b1].append(it2)
                    p[b2].append(it1)
                    loads[b1] = loads[b1] - w1 + w2
                    loads[b2] = loads[b2] - w2 + w1
                    item_bin[it1] = b2
                    item_bin[it2] = b1
                    improved = True

                moves_done += 1

            return improved

        # Main LS loop: alternate emptying attempts and tightening
        # Try to eliminate multiple bins.
        outer = 0
        while outer < 60 and not time_exceeded():
            outer += 1
            improved = False

            # Choose candidate source bins (lightest often easiest)
            bin_idx = [i for i in range(len(p)) if p[i]]
            if not bin_idx:
                break
            bin_idx.sort(key=lambda bi: loads[bi])
            # also sometimes try a random heavy bin
            if len(bin_idx) > 6 and random.random() < 0.25:
                heavy = sorted(bin_idx, key=lambda bi: loads[bi], reverse=True)[:3]
                bin_idx = heavy + bin_idx

            tries = 0
            for src in bin_idx:
                if time_exceeded():
                    break
                if not p[src]:
                    continue
                # Depth selection: shallow usually, sometimes deeper
                depth = 1 if random.random() < 0.75 else 2
                if len(p[src]) > 20 and random.random() < 0.7:
                    depth = 1

                if try_empty_bin(src, depth):
                    # eliminate src
                    p[src] = []
                    loads[src] = 0
                    p[:], loads[:] = compact(p, loads, item_bin)
                    improved = True
                    break

                tries += 1
                if tries >= 10:
                    break

            if improved:
                continue

            # No elimination: try tightening moves to enable future elimination
            if not tighten_phase(iter_cap=2):
                break

        # final compact
        p, loads = compact(p, loads, item_bin)
        return p, loads

    # -------------------- Main GRASP loop --------------------
    # Fixed iteration budget, time checked. Use more iterations than before; LS is stronger but controlled.
    # Use alpha schedule biased to small values (bin packing benefits from greediness) with occasional randomness.
    max_iter = 1200 + int(60 * (n ** 0.5))

    alphas = [0.0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5]

    # Periodic intensification: run LS on current best again (can help after improvements)
    intensify_every = 35

    for it in range(max_iter):
        if time_exceeded():
            break

        alpha = alphas[it % len(alphas)]
        # slightly increase randomness later
        item_rcl_prob = 0.05 if it < max_iter // 3 else 0.12
        if it > 2 * max_iter // 3:
            item_rcl_prob = 0.18

        p, w = construct(alpha, item_rcl_prob)
        if time_exceeded():
            break

        p, w = local_search(p, w)

        k = solution_key(p, w)
        if k < best_key:
            best_key = k
            best_p, best_w = deep_copy(p, w)

        # Intensify occasionally on the best found so far
        if (it + 1) % intensify_every == 0 and not time_exceeded():
            bp, bw = deep_copy(best_p, best_w)
            bp, bw = local_search(bp, bw)
            bk = solution_key(bp, bw)
            if bk < best_key:
                best_key = bk
                best_p, best_w = deep_copy(bp, bw)

    # finalize
    final_p = [b for b in best_p if b]
    final_w = [sum(weights[i] for i in b) for b in final_p]
    return {"packing": final_p, "bin_weights": final_w}
