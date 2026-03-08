import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    # allow using up to 100s (per statement)
    time_budget = min(100.0, max(0.0, float(time_limit)))

    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = int(bin_capacity)

    # ---------- low-overhead time checks ----------
    _tick = 0

    def time_exceeded() -> bool:
        return (time.time() - start) >= time_budget

    def tick(mask: int = 2047) -> bool:
        # mask must be 2^k - 1
        nonlocal _tick
        _tick += 1
        if (_tick & mask) == 0:
            return time_exceeded()
        return False

    # ---------- separate overweight items ----------
    normal_items: List[int] = []
    overweight_bins: List[List[int]] = []
    overweight_w: List[int] = []
    for i, w in enumerate(weights):
        if w > C:
            overweight_bins.append([i])
            overweight_w.append(w)
        else:
            normal_items.append(i)

    if not normal_items:
        return {"packing": overweight_bins[:], "bin_weights": overweight_w[:]}

    # ---------- utilities ----------
    def solution_key(packing: List[List[int]], bin_w: List[int]) -> Tuple[int, int]:
        # primary: bins, secondary: total waste
        waste = 0
        for w in bin_w:
            waste += (C - w)
        return (len(packing), waste)

    def deep_copy_solution(p: List[List[int]], w: List[int]) -> Tuple[List[List[int]], List[int]]:
        return [b[:] for b in p], w[:]

    def block_random_order(items: List[int], block_size: int) -> List[int]:
        arr = sorted(items, key=lambda i: weights[i], reverse=True)
        if block_size <= 1:
            return arr
        out: List[int] = []
        for s in range(0, len(arr), block_size):
            blk = arr[s:s + block_size]
            random.shuffle(blk)
            out.extend(blk)
        return out

    # ---------- baseline: Best-Fit Decreasing with residual index ----------
    def bfd_index(order: List[int]) -> Tuple[List[List[int]], List[int]]:
        # maintain residual -> set of bin indices
        packing: List[List[int]] = []
        bin_w: List[int] = []
        res_bucket: Dict[int, List[int]] = {}  # residual -> list of bins (duplicates allowed, lazily cleaned)

        def add_bin(b: int, res: int) -> None:
            res_bucket.setdefault(res, []).append(b)

        def update_bin(b: int, old_res: int, new_res: int) -> None:
            # lazy: just add new; old will be ignored when popped if mismatched
            add_bin(b, new_res)

        def find_best_fit(w: int) -> int:
            # scan residuals from w upward; C is typically not huge, but to be safe, cap scan using step sampling
            # exact scan is still usually fine for bin packing capacities.
            for r in range(w, C + 1):
                lst = res_bucket.get(r)
                if not lst:
                    continue
                # clean lazily
                while lst:
                    b = lst[-1]
                    if C - bin_w[b] != r:
                        lst.pop()
                        continue
                    return b
            return -1

        for it in order:
            w = weights[it]
            b = find_best_fit(w)
            if b == -1:
                b = len(packing)
                packing.append([it])
                bin_w.append(w)
                add_bin(b, C - w)
            else:
                old_res = C - bin_w[b]
                packing[b].append(it)
                bin_w[b] += w
                new_res = C - bin_w[b]
                update_bin(b, old_res, new_res)
        return packing, bin_w

    # best-known simple start
    base_order = sorted(normal_items, key=lambda i: weights[i], reverse=True)
    best_p, best_w = bfd_index(base_order)
    best_key = solution_key(best_p, best_w)

    # ---------- GRASP constructions ----------
    # We keep few but strong constructors; each provides randomness via RCL or randomized residual choice.

    def choose_from_rcl(cands: List[Tuple[float, int]], alpha: float) -> int:
        # cands: (score, bin_index) lower better
        cands.sort(key=lambda x: x[0])
        mn = cands[0][0]
        mx = cands[-1][0]
        thr = mn + alpha * (mx - mn)
        # if many identical, RCL can be large; sampling is fine
        rcl = [b for (sc, b) in cands if sc <= thr]
        return random.choice(rcl)

    def build_weight_hist(items: List[int]) -> Dict[int, int]:
        hist: Dict[int, int] = {}
        for it in items:
            w = weights[it]
            hist[w] = hist.get(w, 0) + 1
        return hist

    def approx_complement_scarcity(residual: int, hist: Dict[int, int], probe: int = 7) -> int:
        # lower is better
        if residual <= 0:
            return 0
        if residual in hist:
            return 0
        for d in range(1, probe + 1):
            c1 = residual - d
            if c1 > 0 and hist.get(c1, 0) > 0:
                return d
            c2 = residual + d
            if c2 <= C and hist.get(c2, 0) > 0:
                return d
        return probe + 4

    def construct_rbfd(alpha: float, block_size: int) -> Tuple[List[List[int]], List[int]]:
        order = block_random_order(normal_items, block_size)
        remaining_hist = build_weight_hist(order)

        packing: List[List[int]] = []
        bin_w: List[int] = []

        for it in order:
            w = weights[it]
            remaining_hist[w] -= 1
            if remaining_hist[w] == 0:
                del remaining_hist[w]

            cands: List[Tuple[float, int]] = []
            for b, bw in enumerate(bin_w):
                if bw + w <= C:
                    slack_after = C - (bw + w)
                    scarcity = approx_complement_scarcity(slack_after, remaining_hist)
                    # tighter fit + keep residual "easy" to complete
                    score = slack_after + 0.30 * scarcity + 1e-6 * b
                    cands.append((score, b))
            if not cands:
                packing.append([it])
                bin_w.append(w)
            else:
                b = choose_from_rcl(cands, alpha)
                packing[b].append(it)
                bin_w[b] += w
        return packing, bin_w

    def construct_rbf_residual_rcl(alpha: float, block_size: int) -> Tuple[List[List[int]], List[int]]:
        # Like Best-Fit, but instead of enumerating all bins, we search by residual value and randomize among close residuals.
        order = block_random_order(normal_items, block_size)
        remaining_hist = build_weight_hist(order)

        packing: List[List[int]] = []
        bin_w: List[int] = []

        bucket: Dict[int, List[int]] = {}  # residual -> bins (lazy)

        def add_bin(b: int, res: int) -> None:
            bucket.setdefault(res, []).append(b)

        def refresh_pick(res: int) -> int:
            # pop invalid lazily
            lst = bucket.get(res)
            if not lst:
                return -1
            while lst:
                b = lst[-1]
                if C - bin_w[b] != res:
                    lst.pop()
                    continue
                return b
            return -1

        for it in order:
            w = weights[it]
            remaining_hist[w] -= 1
            if remaining_hist[w] == 0:
                del remaining_hist[w]

            if not packing:
                packing.append([it])
                bin_w.append(w)
                add_bin(0, C - w)
                continue

            # build candidate residuals: scan a small window from w upward
            # This strongly biases to best-fit but adds RCL randomness.
            cands: List[Tuple[float, int]] = []
            max_scan = min(C, w + max(25, C // 30))
            for r in range(w, max_scan + 1):
                b = refresh_pick(r)
                if b == -1:
                    continue
                slack_after = r - w
                scarcity = approx_complement_scarcity(slack_after, remaining_hist)
                score = slack_after + 0.25 * scarcity
                cands.append((score, b))
                # collect a few only
                if len(cands) >= 14:
                    break

            if not cands:
                b = len(packing)
                packing.append([it])
                bin_w.append(w)
                add_bin(C - w, C - w)
            else:
                b = choose_from_rcl(cands, alpha)
                old_res = C - bin_w[b]
                packing[b].append(it)
                bin_w[b] += w
                new_res = C - bin_w[b]
                add_bin(new_res, new_res)
                # old_res left lazily
        return packing, bin_w

    def construct_pairing_then_bfd(alpha: float, block_size: int) -> Tuple[List[List[int]], List[int]]:
        # Pair large (>C/2) with complements (classic bin packing GRASP construction idea)
        big = [i for i in normal_items if weights[i] > C // 2]
        small = [i for i in normal_items if weights[i] <= C // 2]

        by_w: Dict[int, List[int]] = {}
        for it in small:
            by_w.setdefault(weights[it], []).append(it)
        for lst in by_w.values():
            random.shuffle(lst)

        big_order = block_random_order(big, max(1, min(block_size, 14)))
        used = set()
        packing: List[List[int]] = []
        bin_w: List[int] = []

        def take(target: int) -> Optional[int]:
            if target <= 0:
                return None
            lst = by_w.get(target)
            if lst:
                return lst.pop()
            # near targets
            for d in range(1, 10):
                t1 = target - d
                if t1 > 0:
                    lst1 = by_w.get(t1)
                    if lst1:
                        return lst1.pop()
                t2 = target + d
                if t2 <= C // 2:
                    lst2 = by_w.get(t2)
                    if lst2:
                        return lst2.pop()
            return None

        for it in big_order:
            used.add(it)
            w = weights[it]
            comp = take(C - w)
            if comp is not None:
                used.add(comp)
                packing.append([it, comp])
                bin_w.append(w + weights[comp])
            else:
                packing.append([it])
                bin_w.append(w)

        rest = [i for i in normal_items if i not in used]
        if rest:
            # use an RBFD packing into existing bins with RCL
            order = block_random_order(rest, block_size)
            remaining_hist = build_weight_hist(order)

            for it in order:
                w = weights[it]
                remaining_hist[w] -= 1
                if remaining_hist[w] == 0:
                    del remaining_hist[w]

                cands: List[Tuple[float, int]] = []
                for b, bw in enumerate(bin_w):
                    if bw + w <= C:
                        slack_after = C - (bw + w)
                        scarcity = approx_complement_scarcity(slack_after, remaining_hist)
                        score = slack_after + 0.30 * scarcity + 1e-6 * b
                        cands.append((score, b))
                if not cands:
                    packing.append([it])
                    bin_w.append(w)
                else:
                    b = choose_from_rcl(cands, alpha)
                    packing[b].append(it)
                    bin_w[b] += w

        return packing, bin_w

    constructors = [
        ("RBFD", construct_rbfd),
        ("RBF-RCL", construct_rbf_residual_rcl),
        ("PAIR", construct_pairing_then_bfd),
    ]

    # ---------- Local search (GRASP essential): bin elimination + repack ----------
    class LSState:
        __slots__ = ("bins", "bin_w", "item_bin", "pos", "res", "bucket")

        def __init__(self, bins: List[List[int]], bin_w: List[int]):
            self.bins = bins
            self.bin_w = bin_w
            self.item_bin = [-1] * n
            self.pos = [-1] * n
            self.res = [C - w for w in bin_w]
            self.bucket: Dict[int, set] = {}
            for b in range(len(bins)):
                self.bucket.setdefault(self.res[b], set()).add(b)
                for p, it in enumerate(bins[b]):
                    self.item_bin[it] = b
                    self.pos[it] = p

        def _bucket_remove(self, b: int, old_r: int) -> None:
            s = self.bucket.get(old_r)
            if s is not None:
                s.discard(b)
                if not s:
                    del self.bucket[old_r]

        def _bucket_add(self, b: int, new_r: int) -> None:
            self.bucket.setdefault(new_r, set()).add(b)

        def _update_res(self, b: int, new_w: int) -> None:
            old_r = self.res[b]
            new_r = C - new_w
            if old_r != new_r:
                self._bucket_remove(b, old_r)
                self.res[b] = new_r
                self._bucket_add(b, new_r)
            self.bin_w[b] = new_w

        def move_item(self, it: int, src: int, dst: int) -> None:
            w = weights[it]
            src_list = self.bins[src]
            i = self.pos[it]
            last = src_list[-1]
            src_list[i] = last
            self.pos[last] = i
            src_list.pop()

            dst_list = self.bins[dst]
            self.pos[it] = len(dst_list)
            dst_list.append(it)
            self.item_bin[it] = dst

            self._update_res(src, self.bin_w[src] - w)
            self._update_res(dst, self.bin_w[dst] + w)

        def swap_items(self, it1: int, b1: int, it2: int, b2: int) -> None:
            if b1 == b2:
                return
            w1, w2 = weights[it1], weights[it2]
            p1, p2 = self.pos[it1], self.pos[it2]
            self.bins[b1][p1] = it2
            self.bins[b2][p2] = it1
            self.pos[it1] = p2
            self.pos[it2] = p1
            self.item_bin[it1] = b2
            self.item_bin[it2] = b1
            self._update_res(b1, self.bin_w[b1] - w1 + w2)
            self._update_res(b2, self.bin_w[b2] - w2 + w1)

        def compact(self) -> None:
            new_bins: List[List[int]] = []
            new_w: List[int] = []
            for b, lst in enumerate(self.bins):
                if lst:
                    new_bins.append(lst)
                    new_w.append(self.bin_w[b])
            self.__init__(new_bins, new_w)

        def find_best_fit_bin(self, w: int, exclude: int = -1) -> int:
            # find bin with residual >= w minimizing residual
            for r in range(w, C + 1):
                s = self.bucket.get(r)
                if not s:
                    continue
                if exclude == -1:
                    return next(iter(s))
                if exclude not in s:
                    return next(iter(s))
                if len(s) >= 2:
                    for b in s:
                        if b != exclude:
                            return b
            return -1

        def candidate_bins_with_residual_at_least(self, w: int, exclude: int, limit: int) -> List[int]:
            # collect a few bins starting from tightest residual
            out: List[int] = []
            for r in range(w, C + 1):
                s = self.bucket.get(r)
                if not s:
                    continue
                for b in s:
                    if b != exclude:
                        out.append(b)
                        if len(out) >= limit:
                            return out
            return out

    def local_search(packing: List[List[int]], bin_w: List[int], ls_seconds: float) -> Tuple[List[List[int]], List[int]]:
        if not packing:
            return packing, bin_w

        st = LSState([b[:] for b in packing], bin_w[:])
        deadline = min(start + time_budget, time.time() + max(0.0, ls_seconds))

        def ls_time_exceeded() -> bool:
            return time.time() >= deadline or time_exceeded()

        # Core LS: repeatedly try to delete a bin.
        # Neighborhoods: direct relocate; 1-1 swap; 1-ejection (move one item out of a target bin).

        def try_delete_bin(src: int) -> bool:
            if not st.bins[src]:
                return False
            # attempt larger items first
            items = st.bins[src][:]
            items.sort(key=lambda i: weights[i], reverse=True)

            trail: List[Tuple] = []

            def rollback() -> None:
                for op in reversed(trail):
                    t = op[0]
                    if t == "m":
                        _, it, a, b = op
                        st.move_item(it, b, a)
                    else:
                        _, it1, b1, it2, b2 = op
                        st.swap_items(it1, b2, it2, b1)

            for it in items:
                if ls_time_exceeded():
                    rollback()
                    return False
                w = weights[it]

                # 1) direct best-fit move
                dst = st.find_best_fit_bin(w, exclude=src)
                if dst != -1:
                    st.move_item(it, src, dst)
                    trail.append(("m", it, src, dst))
                    continue

                # 2) try limited candidate bins and perform 1-1 swap (it with some j)
                cand_bins = st.candidate_bins_with_residual_at_least(1, exclude=src, limit=14)
                random.shuffle(cand_bins)
                improved = False
                for b in cand_bins[:10]:
                    if b == src or not st.bins[b]:
                        continue
                    # need make room: deficit
                    deficit = w - st.res[b]
                    if deficit <= 0:
                        continue
                    # attempt swap with a small-enough item in b such that after swap both bins feasible
                    # feasibility: st.bin_w[b] - wj + w <= C => wj >= deficit
                    # and src after receiving wj: st.bin_w[src] - w + wj <= C always true since src has w removed
                    # but src might be emptying; still fine.
                    # choose minimal wj >= deficit to keep b tight.
                    best_j = None
                    best_wj = None
                    # scan some light items in b
                    b_items = st.bins[b][:]
                    b_items.sort(key=lambda x: weights[x])
                    for j in b_items[: min(12, len(b_items))]:
                        wj = weights[j]
                        if wj >= deficit:
                            best_j = j
                            best_wj = wj
                            break
                    if best_j is None:
                        continue
                    j = best_j
                    wj = best_wj  # type: ignore
                    # check src can take j after removing it
                    if st.bin_w[src] - w + wj > C:
                        continue
                    st.swap_items(it, src, j, b)
                    trail.append(("s", it, src, j, b))
                    improved = True
                    break
                if improved:
                    continue

                # 3) 1-ejection: move one j out of some b to elsewhere, then move it into b
                placed = False
                cand_bins2 = list(range(len(st.bins)))
                random.shuffle(cand_bins2)
                cand_bins2 = cand_bins2[: min(len(cand_bins2), 18)]
                # prefer higher residual
                cand_bins2.sort(key=lambda bb: st.res[bb], reverse=True)

                for b in cand_bins2:
                    if ls_time_exceeded():
                        break
                    if b == src or not st.bins[b]:
                        continue
                    deficit = w - st.res[b]
                    if deficit <= 0:
                        continue
                    # pick smallest j with wj >= deficit
                    b_items = st.bins[b][:]
                    b_items.sort(key=lambda x: weights[x])
                    j = None
                    for cand in b_items[: min(14, len(b_items))]:
                        if weights[cand] >= deficit:
                            j = cand
                            break
                    if j is None:
                        continue
                    wj = weights[j]
                    new_dst = st.find_best_fit_bin(wj, exclude=b)
                    if new_dst == -1:
                        continue
                    st.move_item(j, b, new_dst)
                    trail.append(("m", j, b, new_dst))
                    # now b should fit it
                    if st.res[b] >= w:
                        st.move_item(it, src, b)
                        trail.append(("m", it, src, b))
                        placed = True
                        break
                    # rollback just this attempt
                    rollback()
                    trail.clear()

                if placed:
                    continue

                rollback()
                return False

            # bin emptied
            if not st.bins[src]:
                st.compact()
                return True
            return False

        # Intensification: repeatedly attempt to delete lightest bins
        attempts = 0
        # more attempts with larger instances / more time
        max_attempts = 60 if len(st.bins) < 250 else 40

        while attempts < max_attempts and not ls_time_exceeded() and len(st.bins) > 1:
            attempts += 1
            B = len(st.bins)
            # sample bins; focus on light ones
            idxs = list(range(B))
            random.shuffle(idxs)
            idxs = idxs[: min(B, 14)]
            idxs.sort(key=lambda b: st.bin_w[b])
            src = idxs[0]
            if try_delete_bin(src):
                attempts = max(0, attempts - 3)

        # Small repack of two light bins into one if possible (common LS in bin packing GRASP)
        if not ls_time_exceeded() and len(st.bins) >= 3:
            light = sorted(range(len(st.bins)), key=lambda b: st.bin_w[b])
            a, b = light[0], light[1]
            if a != b:
                items_union = st.bins[a][:] + st.bins[b][:]
                tot = 0
                for it in items_union:
                    tot += weights[it]
                if tot <= C:
                    # rebuild without bins a,b and add merged
                    new_bins: List[List[int]] = []
                    new_w: List[int] = []
                    for bi in range(len(st.bins)):
                        if bi not in (a, b):
                            new_bins.append(st.bins[bi])
                            new_w.append(st.bin_w[bi])
                    items_union.sort(key=lambda i: weights[i], reverse=True)
                    new_bins.append(items_union)
                    new_w.append(tot)
                    st = LSState(new_bins, new_w)

        return st.bins, st.bin_w

    # ---------- elite pool + path relinking (standard GRASP enhancement) ----------
    def residual_signature(bin_w: List[int], bucket_size: int) -> Tuple[Tuple[int, int], ...]:
        hist: Dict[int, int] = {}
        for w in bin_w:
            r = C - w
            k = r // bucket_size
            hist[k] = hist.get(k, 0) + 1
        return tuple(sorted(hist.items()))

    elite_E = 12
    elite: List[Tuple[Tuple[int, int], Tuple[Tuple[int, int], ...], List[List[int]], List[int]]] = []

    def elite_add(p: List[List[int]], w: List[int]) -> None:
        k = solution_key(p, w)
        sig = residual_signature(w, bucket_size=max(5, min(25, C // 35 + 1)))
        for ek, es, _, _ in elite:
            if ek == k and es == sig:
                return
        elite.append((k, sig, [b[:] for b in p], w[:]))
        elite.sort(key=lambda x: x[0])
        if len(elite) > elite_E:
            elite.pop()

    def path_relink(p_from: List[List[int]], w_from: List[int], p_to: List[List[int]], seconds: float) -> Tuple[List[List[int]], List[int]]:
        deadline = min(start + time_budget, time.time() + max(0.0, seconds))

        target_bin_of = [-1] * n
        for b, lst in enumerate(p_to):
            for it in lst:
                target_bin_of[it] = b

        st = LSState([b[:] for b in p_from], w_from[:])

        # Create a mapping from target bins to current bins using load similarity (coarse)
        tgt_loads = []
        for lst in p_to:
            s = 0
            for it in lst:
                s += weights[it]
            tgt_loads.append(s)

        # order items by weight to move heavy mismatches first
        items_all = [it for binlst in st.bins for it in binlst]
        items_all.sort(key=lambda i: weights[i], reverse=True)
        random.shuffle(items_all[: min(20, len(items_all))])

        moves = 0
        for it in items_all:
            if time.time() >= deadline or time_exceeded():
                break
            tb = target_bin_of[it]
            if tb == -1:
                continue
            src = st.item_bin[it]
            if src == -1:
                continue

            # attempt to place into a bin whose load is close to target load
            w = weights[it]
            best_dst = -1
            best_sc = None
            # sample destination bins
            B = len(st.bins)
            cand = list(range(B))
            random.shuffle(cand)
            cand = cand[: min(B, 20)]
            for dst in cand:
                if dst == src:
                    continue
                if st.bin_w[dst] + w <= C:
                    sc = abs((st.bin_w[dst] + w) - tgt_loads[tb])
                    if best_sc is None or sc < best_sc:
                        best_sc = sc
                        best_dst = dst
            if best_dst != -1:
                st.move_item(it, src, best_dst)
                moves += 1
                if moves % 40 == 0 and (time.time() >= deadline or time_exceeded()):
                    break

        # finish with LS slice
        rem = max(0.0, deadline - time.time())
        return local_search(st.bins, st.bin_w, ls_seconds=min(0.08, rem))

    # ---------- Main GRASP loop ----------
    alphas = [0.02, 0.06, 0.10, 0.16, 0.25, 0.40, 0.60, 0.82]

    cons_stats = {name: {"tries": 0, "best": 10**9, "avg": 0.0} for name, _ in constructors}
    alpha_stats = {a: {"tries": 0, "best": 10**9, "avg": 0.0} for a in alphas}

    def choose_constructor(eps: float) -> Tuple[str, callable]:
        if random.random() < eps:
            return random.choice(constructors)
        best_s = None
        pick = constructors[0]
        for name, fn in constructors:
            stt = cons_stats[name]
            if stt["tries"] == 0:
                s = 1e9
            else:
                s = 0.75 * stt["avg"] + 0.25 * stt["best"]
            if best_s is None or s < best_s:
                best_s = s
                pick = (name, fn)
        return pick

    def choose_alpha(eps: float) -> float:
        if random.random() < eps:
            return random.choice(alphas)
        best_s = None
        best_a = alphas[0]
        for a in alphas:
            stt = alpha_stats[a]
            if stt["tries"] == 0:
                s = 1e9
            else:
                s = 0.75 * stt["avg"] + 0.25 * stt["best"]
            if best_s is None or s < best_s:
                best_s = s
                best_a = a
        return best_a

    # fixed iteration budget, but large enough to use full time with periodic checks
    m = len(normal_items)
    max_iter = 12000 + 120 * int(m ** 0.5)

    no_improve = 0
    # seed elite with baseline
    elite_add(best_p, best_w)

    for it in range(max_iter):
        if tick(1023):
            break

        # diversification/intensification schedule by stagnation and remaining time
        elapsed = time.time() - start
        remaining = max(0.0, time_budget - elapsed)

        if no_improve >= 120:
            eps = 0.35
            block_size = 24 if m < 500 else 32
        elif no_improve >= 50:
            eps = 0.22
            block_size = 16 if m < 500 else 24
        else:
            eps = 0.12
            block_size = 10 if m < 400 else 16

        name, cons = choose_constructor(eps)
        alpha = choose_alpha(eps)

        if name == "RBFD":
            p, w = cons(alpha, block_size)
        elif name == "RBF-RCL":
            p, w = cons(alpha, block_size)
        else:  # PAIR
            p, w = cons(alpha, block_size)

        if tick(1023):
            break

        bins0 = len(p)
        cs = cons_stats[name]
        cs["tries"] += 1
        cs["avg"] += (bins0 - cs["avg"]) / cs["tries"]
        cs["best"] = min(cs["best"], bins0)

        as_ = alpha_stats[alpha]
        as_["tries"] += 1
        as_["avg"] += (bins0 - as_["avg"]) / as_["tries"]
        as_["best"] = min(as_["best"], bins0)

        # LS time slice: spend more later / on promising solutions / under stagnation
        # keep bounded but allow heavier use since we can run up to 100s
        if remaining > 0:
            base_ls = 0.008 + 0.020 * (elapsed / max(1e-9, time_budget))
            if no_improve >= 50:
                base_ls *= 1.35
            # if construction already near best bins, intensify
            if bins0 <= best_key[0] + 1:
                base_ls *= 1.35
            ls_seconds = min(0.22, max(0.006, base_ls))
            ls_seconds = min(ls_seconds, remaining * 0.08)
        else:
            ls_seconds = 0.0

        if ls_seconds > 0:
            p, w = local_search(p, w, ls_seconds)

        k = solution_key(p, w)
        if k < best_key:
            best_key = k
            best_p, best_w = deep_copy_solution(p, w)
            elite_add(best_p, best_w)
            no_improve = 0
        else:
            no_improve += 1
            if len(elite) < elite_E or k <= elite[-1][0]:
                elite_add(p, w)

        # path relinking periodically; more when stagnating
        if elite and (it % (35 if m < 800 else 45) == 0) and not time_exceeded():
            if no_improve >= 30 or k[0] <= best_key[0] + 1:
                # choose a good elite target
                _, _, ep, ew = random.choice(elite[: min(len(elite), 6)])
                pr_seconds = min(0.18, max(0.03, 0.012 * time_budget))
                pr_seconds = min(pr_seconds, max(0.0, (time_budget - (time.time() - start)) * 0.06))
                if pr_seconds > 0:
                    pr_p, pr_w = path_relink(p, w, ep, seconds=pr_seconds)
                    pr_k = solution_key(pr_p, pr_w)
                    if pr_k < best_key:
                        best_key = pr_k
                        best_p, best_w = deep_copy_solution(pr_p, pr_w)
                        elite_add(best_p, best_w)
                        no_improve = 0

    # finalize: remove empties and recompute weights
    final_p = [b for b in best_p if b]
    final_w = [sum(weights[i] for i in b) for b in final_p]

    # append overweight bins
    final_p.extend(overweight_bins)
    final_w.extend(overweight_w)

    return {"packing": final_p, "bin_weights": final_w}
