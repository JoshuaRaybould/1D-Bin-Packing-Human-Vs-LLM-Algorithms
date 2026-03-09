import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    C = bin_capacity
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # If infeasible items exist, we still return a packing (may violate). Problem typically assumes feasibility.
    # Keep going but construction/LS will treat them as standalone bins.

    # --- Helpers: scoring bin-count first ---
    total_weight = sum(weights)

    def score_from_residuals(residuals: List[int]) -> Tuple[int, int, int]:
        # (bins, sum(res^2), max_res)
        if not residuals:
            return (0, 0, 0)
        s2 = 0
        mx = 0
        for r in residuals:
            s2 += r * r
            if r > mx:
                mx = r
        return (len(residuals), s2, mx)

    # --- Item order diversification: bucket shuffle by weight each construction ---
    # Pre-bucket indices by weight, descending weights.
    by_w: Dict[int, List[int]] = {}
    for i, w in enumerate(weights):
        by_w.setdefault(w, []).append(i)
    distinct_weights_desc = sorted(by_w.keys(), reverse=True)

    def iter_items_bucket_shuffled() -> List[int]:
        seq: List[int] = []
        for w in distinct_weights_desc:
            bucket = by_w[w]
            if len(bucket) <= 1:
                seq.extend(bucket)
            else:
                tmp = bucket[:]  # shuffle within same weight
                random.shuffle(tmp)
                seq.extend(tmp)
        return seq

    # --- Baseline: First-Fit Decreasing (best-fit tie) ---
    def ffd(items_seq: List[int]) -> Tuple[List[List[int]], List[int]]:
        packing: List[List[int]] = []
        bw: List[int] = []
        rem: List[int] = []
        for it in items_seq:
            w = weights[it]
            if w > C:
                packing.append([it])
                bw.append(w)
                rem.append(0)
                continue
            best_b = -1
            best_res = None
            for b in range(len(rem)):
                if rem[b] >= w:
                    r = rem[b] - w
                    if best_res is None or r < best_res:
                        best_res = r
                        best_b = b
            if best_b >= 0:
                packing[best_b].append(it)
                bw[best_b] += w
                rem[best_b] -= w
            else:
                packing.append([it])
                bw.append(w)
                rem.append(C - w)
        return packing, bw

    # --- GRASP construction with improved bin evaluation and percentile RCL ---
    def residual_penalty(res_after: int) -> int:
        # Small piecewise penalty to avoid hard-to-fill medium residuals.
        # Lower is better.
        if res_after <= 1:
            return 0
        # discourage medium residuals roughly around (C/3, C/2)
        if C > 0:
            if (C // 3) < res_after <= (C // 2):
                return 3
            if (C // 5) < res_after <= (C // 3):
                return 2
        return 1

    def alpha_to_k(alpha: float, m: int) -> int:
        if m <= 1:
            return 1
        # Map alpha in [0,1] to k in [1,m]
        k = 1 + int(alpha * (m - 1))
        if k < 1:
            k = 1
        if k > m:
            k = m
        return k

    def construct(alpha: float) -> Tuple[List[List[int]], List[int], List[int]]:
        items_seq = iter_items_bucket_shuffled()
        packing: List[List[int]] = []
        bw: List[int] = []
        rem: List[int] = []

        for it in items_seq:
            w = weights[it]
            if w > C:
                packing.append([it])
                bw.append(w)
                rem.append(0)
                continue

            # collect feasible bins with key
            feas: List[Tuple[Tuple[int, int], int]] = []  # ((res_after, penalty), bin)
            for b, r in enumerate(rem):
                if r >= w:
                    ra = r - w
                    feas.append(((ra, residual_penalty(ra)), b))

            if not feas:
                packing.append([it])
                bw.append(w)
                rem.append(C - w)
                continue

            # sort by desirability
            feas.sort(key=lambda x: x[0])
            k = alpha_to_k(alpha, len(feas))
            rcl = feas[: max(1, k)]
            chosen_b = random.choice(rcl)[1]

            packing[chosen_b].append(it)
            bw[chosen_b] += w
            rem[chosen_b] -= w

        return packing, bw, rem

    # --- Local search: bin elimination with efficient data structures ---
    class State:
        __slots__ = ("packing", "bw", "res", "item_to_bin", "pos_in_bin")

        def __init__(self, packing: List[List[int]], bw: List[int]):
            self.packing = [lst[:] for lst in packing]
            self.bw = bw[:]
            self.res = [max(0, C - x) for x in self.bw]
            self.item_to_bin = [-1] * n
            self.pos_in_bin = [-1] * n
            for b, bin_items in enumerate(self.packing):
                for pos, it in enumerate(bin_items):
                    self.item_to_bin[it] = b
                    self.pos_in_bin[it] = pos

        def bins(self) -> int:
            return len(self.packing)

        def remove_item_from_bin(self, it: int, b: int) -> None:
            # swap-delete from packing[b]
            pos = self.pos_in_bin[it]
            last_it = self.packing[b][-1]
            self.packing[b][pos] = last_it
            self.pos_in_bin[last_it] = pos
            self.packing[b].pop()
            self.pos_in_bin[it] = -1
            self.item_to_bin[it] = -1

        def add_item_to_bin(self, it: int, b: int) -> None:
            self.pos_in_bin[it] = len(self.packing[b])
            self.packing[b].append(it)
            self.item_to_bin[it] = b

        def move_item(self, it: int, b_from: int, b_to: int) -> None:
            w = weights[it]
            self.remove_item_from_bin(it, b_from)
            self.add_item_to_bin(it, b_to)
            self.bw[b_from] -= w
            self.res[b_from] = max(0, C - self.bw[b_from])
            self.bw[b_to] += w
            self.res[b_to] = max(0, C - self.bw[b_to])

        def swap_items(self, ia: int, ba: int, ib: int, bb: int) -> None:
            # remove both then add swapped
            wa, wb = weights[ia], weights[ib]
            self.remove_item_from_bin(ia, ba)
            self.remove_item_from_bin(ib, bb)
            self.add_item_to_bin(ia, bb)
            self.add_item_to_bin(ib, ba)
            self.bw[ba] = self.bw[ba] - wa + wb
            self.bw[bb] = self.bw[bb] - wb + wa
            self.res[ba] = max(0, C - self.bw[ba])
            self.res[bb] = max(0, C - self.bw[bb])

        def remove_bin(self, b: int) -> None:
            # swap-with-last bin removal to avoid shifting O(m)
            last = len(self.packing) - 1
            if b != last:
                # move last bin into b
                self.packing[b] = self.packing[last]
                self.bw[b] = self.bw[last]
                self.res[b] = self.res[last]
                # update mappings for items moved
                for pos, it in enumerate(self.packing[b]):
                    self.item_to_bin[it] = b
                    self.pos_in_bin[it] = pos
            self.packing.pop()
            self.bw.pop()
            self.res.pop()

    def best_fit_targets(state: State, w: int, exclude_bin: int, k: int) -> List[int]:
        # Return up to k bins that can accept weight w, best-fit (min residual after).
        best: List[Tuple[int, int]] = []  # (res_after, bin)
        for b, r in enumerate(state.res):
            if b == exclude_bin:
                continue
            if r >= w:
                ra = r - w
                best.append((ra, b))
        if not best:
            return []
        best.sort(key=lambda x: x[0])
        return [b for _, b in best[:k]]

    def try_empty_bin_direct(state: State, b0: int, targets_per_item: int = 6) -> bool:
        # Attempt to empty bin b0 by moving its items one-by-one, but commit only if all can be placed.
        if b0 >= state.bins() or not state.packing[b0]:
            return False
        items0 = state.packing[b0][:]
        # place heavier first
        items0.sort(key=lambda it: weights[it], reverse=True)

        # Plan moves without committing
        planned: List[Tuple[int, int]] = []  # (item, target_bin)
        # Temporary residuals
        tmp_res = state.res[:]
        tmp_res[b0] = state.res[b0]  # unchanged

        for it in items0:
            w = weights[it]
            if w > C:
                return False
            # find best-fit target using tmp_res
            best_bins: List[Tuple[int, int]] = []
            for b, r in enumerate(tmp_res):
                if b == b0:
                    continue
                if r >= w:
                    ra = r - w
                    best_bins.append((ra, b))
            if not best_bins:
                return False
            best_bins.sort(key=lambda x: x[0])
            cand = best_bins[: max(1, min(targets_per_item, len(best_bins)))]
            ra, b_to = random.choice(cand)
            planned.append((it, b_to))
            tmp_res[b_to] -= w

        # Commit moves
        for it, b_to in planned:
            # b0 may change only at the end because we do swap-with-last removal;
            # but we avoid removing b0 until all moved.
            state.move_item(it, b0, b_to)

        # now b0 empty -> remove
        if state.bw[b0] == 0:
            state.remove_bin(b0)
            return True
        return False

    def subset_sum_move_from_bin(state: State, b0: int, max_k: int = 3, consider_k_items: int = 30) -> bool:
        # Try to move 1/2/3 items from b0 to fill residuals of other bins exactly.
        if b0 >= state.bins() or not state.packing[b0]:
            return False

        pool = state.packing[b0][:]
        if len(pool) > consider_k_items:
            # mix of heavy and light
            pool.sort(key=lambda it: weights[it])
            k = consider_k_items // 2
            pool = pool[:k] + pool[-(consider_k_items - k):]

        # index singles
        single: Dict[int, int] = {}
        for it in pool:
            single.setdefault(weights[it], it)

        # try fill bins with largest residual first (harder)
        targets = [(r, b) for b, r in enumerate(state.res) if b != b0 and r > 0]
        if not targets:
            return False
        targets.sort(reverse=True)

        # pairs map
        pair: Dict[int, Tuple[int, int]] = {}
        if max_k >= 2:
            m = len(pool)
            for i in range(m):
                wi = weights[pool[i]]
                for j in range(i + 1, m):
                    s = wi + weights[pool[j]]
                    if s not in pair:
                        pair[s] = (pool[i], pool[j])

        for r, b in targets[: min(25, len(targets))]:
            # 1-item exact
            it1 = single.get(r)
            if it1 is not None and state.item_to_bin[it1] == b0 and state.res[b] >= r:
                state.move_item(it1, b0, b)
                if state.bw[b0] == 0:
                    state.remove_bin(b0)
                return True

            # 2-item exact
            if max_k >= 2:
                pr = pair.get(r)
                if pr is not None:
                    a, c = pr
                    if state.item_to_bin[a] == b0 and state.item_to_bin[c] == b0 and a != c and state.res[b] >= r:
                        state.move_item(a, b0, b)
                        # b0 may have swapped contents positions; b0 index unchanged until removal
                        state.move_item(c, b0, b)
                        if state.bw[b0] == 0:
                            state.remove_bin(b0)
                        return True

            # 3-item exact using pair sums
            if max_k >= 3:
                # find x + (y+z) = r
                for x in pool[: min(len(pool), 24)]:
                    if state.item_to_bin[x] != b0:
                        continue
                    wx = weights[x]
                    rest = r - wx
                    pr2 = pair.get(rest)
                    if pr2 is None:
                        continue
                    y, z = pr2
                    if x == y or x == z or y == z:
                        continue
                    if state.item_to_bin[y] != b0 or state.item_to_bin[z] != b0:
                        continue
                    if state.res[b] >= r:
                        state.move_item(x, b0, b)
                        state.move_item(y, b0, b)
                        state.move_item(z, b0, b)
                        if state.bw[b0] == 0:
                            state.remove_bin(b0)
                        return True

        return False

    def ejection_chain_empty(state: State, b0: int, depth: int = 2) -> bool:
        # Attempt to empty b0 by moving one item x into target bin by ejecting one item y,
        # placing y elsewhere (optionally with one more ejection).
        if b0 >= state.bins() or not state.packing[b0]:
            return False

        # choose a few candidate items from b0 (heaviest first)
        items0 = state.packing[b0][:]
        items0.sort(key=lambda it: weights[it], reverse=True)
        items0 = items0[: min(10, len(items0))]

        m = state.bins()
        # candidate target bins: tight bins are good
        target_bins = list(range(m))
        target_bins = [b for b in target_bins if b != b0 and state.res[b] < C]
        target_bins.sort(key=lambda b: state.res[b])
        target_bins = target_bins[: min(12, len(target_bins))]
        if not target_bins:
            return False

        for x in items0:
            wx = weights[x]
            for bt in target_bins:
                # if fits directly, skip (direct method handles); here focus on non-fitting
                if state.res[bt] >= wx:
                    continue
                need = wx - state.res[bt]
                # try eject y from bt with weight >= need
                cand_y = state.packing[bt][:]
                cand_y.sort(key=lambda it: weights[it], reverse=True)
                cand_y = cand_y[: min(8, len(cand_y))]

                for y in cand_y:
                    wy = weights[y]
                    if wy < need:
                        continue
                    # after ejecting y, x fits into bt if:
                    # new_res_bt = res_bt + wy - wx >= 0
                    if state.res[bt] + wy < wx:
                        continue

                    # try relocate y somewhere else
                    # best-fit bins for y excluding bt and b0
                    targets_y = best_fit_targets(state, wy, exclude_bin=bt, k=8)
                    targets_y = [b for b in targets_y if b != b0]

                    for by in targets_y:
                        # Commit swap x<->y via move and relocate
                        # Steps:
                        # - move y from bt to by
                        # - move x from b0 to bt
                        # This changes state; if fails to ultimately empty b0 we still made a tightening move.
                        # But acceptance requires bin reduction, so we will only keep if it helps empty b0 now.
                        # To keep it simple, we attempt to finish emptying b0 with direct test after the ejection.
                        # Save snapshot for rollback (small):
                        snapshot = ([(lst[:]) for lst in state.packing], state.bw[:], state.res[:], state.item_to_bin[:], state.pos_in_bin[:])

                        # perform y -> by
                        state.move_item(y, bt, by)
                        # perform x -> bt
                        state.move_item(x, b0, bt)

                        # try finish emptying b0 directly now
                        if state.bw[b0] == 0:
                            state.remove_bin(b0)
                            return True

                        if try_empty_bin_direct(state, b0):
                            return True

                        # depth-2: attempt one more subset-sum or direct on same b0
                        if depth >= 2:
                            if subset_sum_move_from_bin(state, b0, max_k=2) and state.bins() < len(snapshot[1]):
                                return True

                        # rollback
                        state.packing = snapshot[0]
                        state.bw = snapshot[1]
                        state.res = snapshot[2]
                        state.item_to_bin = snapshot[3]
                        state.pos_in_bin = snapshot[4]

        return False

    def repack_pool_min_bins(pool: List[int], max_bins: int) -> Optional[List[List[int]]]:
        # Deterministic best-fit decreasing repack into <= max_bins bins; return packing if success else None.
        pool_sorted = sorted(pool, key=lambda it: weights[it], reverse=True)
        bins: List[List[int]] = [[] for _ in range(max_bins)]
        bw: List[int] = [0] * max_bins
        rem: List[int] = [C] * max_bins
        for it in pool_sorted:
            w = weights[it]
            if w > C:
                return None
            best_b = -1
            best_ra = None
            for b in range(max_bins):
                if rem[b] >= w:
                    ra = rem[b] - w
                    if best_ra is None or ra < best_ra:
                        best_ra = ra
                        best_b = b
            if best_b < 0:
                return None
            bins[best_b].append(it)
            bw[best_b] += w
            rem[best_b] -= w
        # remove empty bins at end (should be none if pool non-empty, but possible)
        out = [b for b in bins if b]
        return out

    def lns_repack_bins(state: State, k: int = 5) -> bool:
        m = state.bins()
        if m <= 1:
            return False
        k = min(k, m)

        # include lightest bin + a few largest residual bins
        idxs = list(range(m))
        idxs.sort(key=lambda b: state.bw[b])
        chosen = [idxs[0]]
        others = [b for b in range(m) if b != chosen[0]]
        others.sort(key=lambda b: state.res[b], reverse=True)
        chosen.extend(others[: k - 1])
        chosen = list(dict.fromkeys(chosen))

        # pool items
        pool: List[int] = []
        for b in chosen:
            pool.extend(state.packing[b])

        before_bins = len(chosen)
        # try repack into before_bins-1 to eliminate one
        target_bins = before_bins - 1
        if target_bins >= 1:
            rep = repack_pool_min_bins(pool, target_bins)
            if rep is not None:
                # Apply: remove chosen bins (descending to avoid index issues with swap-remove)
                chosen_sorted = sorted(chosen, reverse=True)
                for b in chosen_sorted:
                    state.remove_bin(b)
                # add repacked bins
                for b_items in rep:
                    bw_new = sum(weights[it] for it in b_items)
                    state.packing.append([])
                    state.bw.append(0)
                    state.res.append(C)
                    new_b = state.bins() - 1
                    for it in b_items:
                        state.add_item_to_bin(it, new_b)
                        state.bw[new_b] += weights[it]
                    state.res[new_b] = max(0, C - state.bw[new_b])
                return True

        return False

    def tighten_swaps(state: State, attempts: int = 60) -> bool:
        # Within same bin count, do swaps that improve sum(res^2)
        m = state.bins()
        if m <= 1:
            return False
        improved = False
        for _ in range(attempts):
            a = random.randrange(m)
            b = random.randrange(m)
            if a == b or not state.packing[a] or not state.packing[b]:
                continue
            ia = random.choice(state.packing[a])
            ib = random.choice(state.packing[b])
            wa, wb = weights[ia], weights[ib]
            new_bwa = state.bw[a] - wa + wb
            new_bwb = state.bw[b] - wb + wa
            if new_bwa <= C and new_bwb <= C:
                ra0, rb0 = state.res[a], state.res[b]
                ra1, rb1 = C - new_bwa, C - new_bwb
                if (ra1 * ra1 + rb1 * rb1) < (ra0 * ra0 + rb0 * rb0):
                    state.swap_items(ia, a, ib, b)
                    improved = True
        return improved

    def local_search(state: State, best_bins: int, heavy: bool, t_slice: float) -> State:
        end_t = min(start + time_limit, time.time() + t_slice)
        no_improve_rounds = 0
        max_no_improve = 10 if heavy else 6

        while time.time() < end_t and no_improve_rounds < max_no_improve:
            improved = False
            m = state.bins()
            if m <= 1:
                break

            # Candidate bins to eliminate: few lightest, randomized
            cand = list(range(m))
            cand.sort(key=lambda b: state.bw[b])
            cand = cand[: min(12 if heavy else 8, m)]
            random.shuffle(cand)

            for b0 in cand:
                if b0 >= state.bins():
                    continue
                if state.bw[b0] == 0:
                    state.remove_bin(b0)
                    improved = True
                    break

                # 1) direct emptying
                if try_empty_bin_direct(state, b0):
                    improved = True
                    break

                # 2) subset-sum group moves
                if subset_sum_move_from_bin(state, b0, max_k=3 if heavy else 2):
                    # may or may not eliminate; if eliminated, great
                    if state.bins() < m:
                        improved = True
                        break

                # 3) ejection chains
                if heavy and ejection_chain_empty(state, b0, depth=2):
                    improved = True
                    break

                # 4) LNS repack
                if heavy and lns_repack_bins(state, k=5):
                    improved = True
                    break

            if improved:
                no_improve_rounds = 0
                # continue trying to eliminate more bins
                continue

            # If no elimination, do some tightening swaps to help future eliminations
            if tighten_swaps(state, attempts=80 if heavy else 40):
                no_improve_rounds = 0
                continue

            no_improve_rounds += 1

        return state

    # --- Reactive GRASP alpha selection ---
    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    a_idx = {a: i for i, a in enumerate(alphas)}
    # performance score: lower is better; use running mean of achieved bin count primarily
    cnt = [0] * len(alphas)
    mean_bins = [0.0] * len(alphas)

    def choose_alpha(phase2: bool) -> float:
        # Roulette biased toward lower mean_bins; if no data, uniform.
        # In phase2, sharpen selection.
        seen = sum(cnt)
        if seen < len(alphas):
            return random.choice(alphas)
        # compute weights
        best_mb = min(mean_bins)
        weights_prob = []
        for mb in mean_bins:
            # exp(-beta*(mb-best))
            beta = 2.2 if phase2 else 1.2
            # clamp exponent range
            x = -beta * (mb - best_mb)
            if x < -20:
                x = -20
            weights_prob.append(pow(2.718281828, x))
        s = sum(weights_prob)
        r = random.random() * s
        acc = 0.0
        for a, wp in zip(alphas, weights_prob):
            acc += wp
            if acc >= r:
                return a
        return alphas[-1]

    # --- Initialize best with FFD + local search ---
    base_items = iter_items_bucket_shuffled()
    bp, bbw = ffd(base_items)
    st0 = State(bp, bbw)
    st0 = local_search(st0, best_bins=st0.bins(), heavy=True, t_slice=min(0.15, time_limit))
    best_state = st0
    best_score = score_from_residuals(best_state.res)

    # --- Iteration budgeting (fixed iterations + time checks) ---
    # Higher fixed budget; still obey time.
    if n <= 200:
        max_iter = 12000
    elif n <= 1000:
        max_iter = 9000
    else:
        max_iter = 6500

    # Two-phase: last 35% iterations intensify
    phase2_start = int(max_iter * 0.65)

    it = 0
    while it < max_iter:
        it += 1
        if time.time() - start >= time_limit:
            break

        phase2 = it >= phase2_start
        a = choose_alpha(phase2)

        p, bw, _rem = construct(a)
        st = State(p, bw)

        # Adaptive LS depth based on proximity to best bin count
        heavy = st.bins() <= best_state.bins() + 1
        # time slice: short in phase1 unless near-best
        if phase2:
            t_slice = 0.05 if not heavy else 0.12
        else:
            t_slice = 0.02 if not heavy else 0.06
        # also cap by remaining time
        remaining = start + time_limit - time.time()
        if remaining <= 0:
            break
        if t_slice > remaining:
            t_slice = remaining

        st = local_search(st, best_bins=best_state.bins(), heavy=heavy, t_slice=t_slice)

        sc = score_from_residuals(st.res)
        # Update reactive stats with achieved bin count
        i = a_idx[a]
        cnt[i] += 1
        # running mean
        mean_bins[i] += (sc[0] - mean_bins[i]) / cnt[i]

        if sc < best_score:
            best_score = sc
            best_state = st

    # Build final packing output
    final_packing = [bin_items[:] for bin_items in best_state.packing]
    final_bw = best_state.bw[:]  # already maintained

    # Safety recompute weights for alignment/correctness
    final_bw = [sum(weights[i] for i in b) for b in final_packing]
    return {"packing": final_packing, "bin_weights": final_bw}
