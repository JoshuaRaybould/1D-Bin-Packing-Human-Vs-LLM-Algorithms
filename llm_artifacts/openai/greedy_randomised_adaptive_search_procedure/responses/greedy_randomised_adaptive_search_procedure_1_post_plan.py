import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    C = bin_capacity

    # --- time checking with low overhead ---
    _tick = 0
    def time_exceeded() -> bool:
        return (time.time() - start) >= time_limit

    def tick(check_every: int = 2048) -> bool:
        nonlocal _tick
        _tick += 1
        if (_tick & (check_every - 1)) == 0:
            return time_exceeded()
        return False

    # --- item partition: keep overweight items isolated and out of LS ---
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

    # work arrays for normal items only, but indices are original indices

    # --- objective ---
    def solution_key(packing: List[List[int]], bin_w: List[int]) -> Tuple[int, int]:
        waste = sum(C - w for w in bin_w)
        return (len(packing), waste)

    def deep_copy_solution(p: List[List[int]], w: List[int]) -> Tuple[List[List[int]], List[int]]:
        return [b[:] for b in p], w[:]

    # --- block-wise randomized decreasing order ---
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

    # --- residual bucketization helper for construction ---
    def residual_bucket(r: int, bucket_size: int) -> int:
        return r // bucket_size

    # --- small histogram for scarcity penalty in construction ---
    def build_weight_hist(items: List[int]) -> Dict[int, int]:
        hist: Dict[int, int] = {}
        for it in items:
            w = weights[it]
            hist[w] = hist.get(w, 0) + 1
        return hist

    def approx_complement_scarcity(residual: int, hist: Dict[int, int], probe: int = 6) -> int:
        """Lower is better (more fillable residual). Probe a few nearby weights."""
        if residual <= 0:
            return 0
        # exact hit best
        if residual in hist:
            return 0
        # probe nearby values (integer)
        best = None
        for d in range(1, probe + 1):
            c1 = residual - d
            c2 = residual + d
            v = 0
            if c1 > 0:
                v += hist.get(c1, 0)
            if c2 <= C:
                v += hist.get(c2, 0)
            if v > 0:
                best = d
                break
        # If we found something close, small penalty; else larger
        return best if best is not None else (probe + 3)

    # ----------------------- Constructions (GRASP) -----------------------
    # Common RCL builder based on scalar score
    def choose_from_rcl(cands: List[Tuple[float, int]], alpha: float) -> int:
        # cands: (score, bin_index)
        cands.sort(key=lambda x: x[0])
        mn = cands[0][0]
        mx = cands[-1][0]
        thr = mn + alpha * (mx - mn)
        rcl = [b for (sc, b) in cands if sc <= thr]
        return random.choice(rcl)

    def construct_rbfd(alpha: float, block_size: int) -> Tuple[List[List[int]], List[int]]:
        order = block_random_order(normal_items, block_size)
        remaining_hist = build_weight_hist(order)

        packing: List[List[int]] = []
        bin_w: List[int] = []

        for it in order:
            w = weights[it]
            # update remaining histogram (item is being placed now)
            remaining_hist[w] -= 1
            if remaining_hist[w] == 0:
                del remaining_hist[w]

            # candidate bins by best-fit + scarcity penalty on residual after placement
            cands: List[Tuple[float, int]] = []
            for b, bw in enumerate(bin_w):
                if bw + w <= C:
                    slack_after = C - (bw + w)
                    scarcity = approx_complement_scarcity(slack_after, remaining_hist)
                    # score: slack primary, scarcity secondary, tiny tie-break on bin index
                    score = slack_after + 0.35 * scarcity + 1e-6 * b
                    cands.append((score, b))
            if not cands:
                packing.append([it])
                bin_w.append(w)
            else:
                b = choose_from_rcl(cands, alpha)
                packing[b].append(it)
                bin_w[b] += w
        return packing, bin_w

    def construct_rffd(alpha: float, block_size: int) -> Tuple[List[List[int]], List[int]]:
        order = block_random_order(normal_items, block_size)
        packing: List[List[int]] = []
        bin_w: List[int] = []

        # RCL over feasible bins by (earliest index) but with a score that still favors tighter fit a bit.
        for it in order:
            w = weights[it]
            feasible: List[Tuple[float, int]] = []
            for b, bw in enumerate(bin_w):
                if bw + w <= C:
                    slack_after = C - (bw + w)
                    # prefer earlier bins, but allow a slack-based component
                    score = 0.7 * b + 0.3 * (slack_after / max(1, C))
                    feasible.append((score, b))
            if not feasible:
                packing.append([it])
                bin_w.append(w)
            else:
                b = choose_from_rcl(feasible, alpha)
                packing[b].append(it)
                bin_w[b] += w
        return packing, bin_w

    def construct_residual_class_bfd(alpha: float, block_size: int, bucket_size: int) -> Tuple[List[List[int]], List[int]]:
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
                    buck = residual_bucket(slack_after, bucket_size)
                    scarcity = approx_complement_scarcity(slack_after, remaining_hist)
                    # prioritize good residual classes explicitly
                    score = slack_after + 0.25 * scarcity + 0.10 * buck
                    cands.append((score, b))
            if not cands:
                packing.append([it])
                bin_w.append(w)
            else:
                b = choose_from_rcl(cands, alpha)
                packing[b].append(it)
                bin_w[b] += w
        return packing, bin_w

    def construct_pair_then_pack(alpha: float, block_size: int) -> Tuple[List[List[int]], List[int]]:
        # Pair big items (> C/2) with best complement small items when possible
        big = [i for i in normal_items if weights[i] > C // 2]
        small = [i for i in normal_items if weights[i] <= C // 2]

        # multimap weight -> list of items with that weight
        by_w: Dict[int, List[int]] = {}
        for it in small:
            by_w.setdefault(weights[it], []).append(it)
        for lst in by_w.values():
            random.shuffle(lst)

        big_order = block_random_order(big, max(1, min(block_size, 12)))
        used = set()

        packing: List[List[int]] = []
        bin_w: List[int] = []

        def take_complement(target: int) -> Optional[int]:
            # try exact complement first, else near
            if target <= 0:
                return None
            if target in by_w and by_w[target]:
                return by_w[target].pop()
            # near complements
            for d in range(1, 8):
                t1 = target - d
                if t1 > 0 and t1 in by_w and by_w[t1]:
                    return by_w[t1].pop()
                t2 = target + d
                if t2 <= C // 2 and t2 in by_w and by_w[t2]:
                    return by_w[t2].pop()
            return None

        for it in big_order:
            used.add(it)
            w = weights[it]
            comp = take_complement(C - w)
            if comp is not None:
                used.add(comp)
                packing.append([it, comp])
                bin_w.append(w + weights[comp])
            else:
                packing.append([it])
                bin_w.append(w)

        # pack remaining items (unpaired small + any non-big)
        rest = [i for i in normal_items if i not in used]
        if rest:
            # Use RBFD style packing into existing bins, with RCL.
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
                        score = slack_after + 0.35 * scarcity + 1e-6 * b
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
        ("RFFD", construct_rffd),
        ("RCBFD", construct_residual_class_bfd),
        ("PAIR", construct_pair_then_pack),
    ]

    # ----------------------- Baselines -----------------------
    def ffd() -> Tuple[List[List[int]], List[int]]:
        order = sorted(normal_items, key=lambda i: weights[i], reverse=True)
        packing: List[List[int]] = []
        bin_w: List[int] = []
        for it in order:
            w = weights[it]
            for b in range(len(packing)):
                if bin_w[b] + w <= C:
                    packing[b].append(it)
                    bin_w[b] += w
                    break
            else:
                packing.append([it])
                bin_w.append(w)
        return packing, bin_w

    def bfd() -> Tuple[List[List[int]], List[int]]:
        order = sorted(normal_items, key=lambda i: weights[i], reverse=True)
        packing: List[List[int]] = []
        bin_w: List[int] = []
        for it in order:
            w = weights[it]
            best_b = -1
            best_slack = None
            for b, bw in enumerate(bin_w):
                if bw + w <= C:
                    slack = C - (bw + w)
                    if best_slack is None or slack < best_slack:
                        best_slack = slack
                        best_b = b
            if best_b == -1:
                packing.append([it])
                bin_w.append(w)
            else:
                packing[best_b].append(it)
                bin_w[best_b] += w
        return packing, bin_w

    def ffd_bucket_tiebreak(bucket_size: int = 10) -> Tuple[List[List[int]], List[int]]:
        order = sorted(normal_items, key=lambda i: weights[i], reverse=True)
        packing: List[List[int]] = []
        bin_w: List[int] = []
        for it in order:
            w = weights[it]
            best_b = -1
            best_key = None
            for b, bw in enumerate(bin_w):
                if bw + w <= C:
                    slack = C - (bw + w)
                    key = (residual_bucket(slack, bucket_size), slack)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_b = b
            if best_b == -1:
                packing.append([it])
                bin_w.append(w)
            else:
                packing[best_b].append(it)
                bin_w[best_b] += w
        return packing, bin_w

    best_p, best_w = ffd()
    best_key = solution_key(best_p, best_w)
    for base in (bfd, ffd_bucket_tiebreak):
        p, w = base()
        k = solution_key(p, w)
        if k < best_key:
            best_key = k
            best_p, best_w = deep_copy_solution(p, w)

    # ----------------------- Local Search Engine -----------------------
    class LSState:
        __slots__ = ("bins", "bin_w", "item_bin", "pos_in_bin", "bucket", "residuals")

        def __init__(self, bins: List[List[int]], bin_w: List[int]):
            self.bins = bins
            self.bin_w = bin_w
            self.item_bin = [-1] * n
            self.pos_in_bin = [-1] * n
            self.residuals = [C - w for w in bin_w]
            self.bucket: Dict[int, set] = {}
            for b in range(len(bins)):
                self.bucket.setdefault(self.residuals[b], set()).add(b)
                for pos, it in enumerate(bins[b]):
                    self.item_bin[it] = b
                    self.pos_in_bin[it] = pos

        def _bucket_remove(self, b: int, old_res: int) -> None:
            s = self.bucket.get(old_res)
            if s is not None:
                s.discard(b)
                if not s:
                    del self.bucket[old_res]

        def _bucket_add(self, b: int, new_res: int) -> None:
            self.bucket.setdefault(new_res, set()).add(b)

        def _update_bin_residual(self, b: int, new_w: int) -> None:
            old_res = self.residuals[b]
            new_res = C - new_w
            if old_res == new_res:
                self.bin_w[b] = new_w
                return
            self._bucket_remove(b, old_res)
            self.residuals[b] = new_res
            self.bin_w[b] = new_w
            self._bucket_add(b, new_res)

        def move_item(self, it: int, src: int, dst: int) -> None:
            w = weights[it]
            # remove from src in O(1)
            src_list = self.bins[src]
            idx = self.pos_in_bin[it]
            last = src_list[-1]
            src_list[idx] = last
            self.pos_in_bin[last] = idx
            src_list.pop()

            # add to dst
            dst_list = self.bins[dst]
            self.pos_in_bin[it] = len(dst_list)
            dst_list.append(it)
            self.item_bin[it] = dst

            # update weights/residuals
            self._update_bin_residual(src, self.bin_w[src] - w)
            self._update_bin_residual(dst, self.bin_w[dst] + w)

        def swap_items(self, it1: int, b1: int, it2: int, b2: int) -> None:
            if b1 == b2:
                return
            w1, w2 = weights[it1], weights[it2]
            # positions
            p1 = self.pos_in_bin[it1]
            p2 = self.pos_in_bin[it2]
            self.bins[b1][p1] = it2
            self.bins[b2][p2] = it1
            self.pos_in_bin[it1] = p2
            self.pos_in_bin[it2] = p1
            self.item_bin[it1] = b2
            self.item_bin[it2] = b1
            self._update_bin_residual(b1, self.bin_w[b1] - w1 + w2)
            self._update_bin_residual(b2, self.bin_w[b2] - w2 + w1)

        def compact(self) -> None:
            # rebuild bins removing empties
            new_bins: List[List[int]] = []
            new_w: List[int] = []
            remap: Dict[int, int] = {}
            for old_b, b in enumerate(self.bins):
                if b:
                    remap[old_b] = len(new_bins)
                    new_bins.append(b)
                    new_w.append(self.bin_w[old_b])

            self.bins = new_bins
            self.bin_w = new_w
            self.residuals = [C - w for w in new_w]
            self.bucket = {}
            for b in range(len(new_bins)):
                self.bucket.setdefault(self.residuals[b], set()).add(b)

            # rebuild item mappings
            self.item_bin = [-1] * n
            self.pos_in_bin = [-1] * n
            for b, lst in enumerate(self.bins):
                for pos, it in enumerate(lst):
                    self.item_bin[it] = b
                    self.pos_in_bin[it] = pos

        def find_best_fit_bin(self, w: int, exclude: int = -1) -> int:
            # find bin with residual >= w minimizing residual after (i.e., minimal residual)
            # residuals are integers up to C; we scan from w upward but stop quickly in practice.
            for r in range(w, C + 1):
                s = self.bucket.get(r)
                if s:
                    if exclude == -1:
                        return next(iter(s))
                    if exclude not in s:
                        return next(iter(s))
                    if len(s) >= 2:
                        for b in s:
                            if b != exclude:
                                return b
            return -1

    def local_search(packing: List[List[int]], bin_w: List[int], ls_seconds: float) -> Tuple[List[List[int]], List[int]]:
        if not packing:
            return packing, bin_w

        st = LSState([b[:] for b in packing], bin_w[:])
        deadline = min(start + time_limit, time.time() + max(0.0, ls_seconds))

        def ls_time_exceeded() -> bool:
            return time.time() >= deadline or time_exceeded()

        # Phase A: bin deletion via direct moves + bounded ejection chain
        # Try limited number of source bins each call.
        attempts = 0
        max_attempts = 24 if len(st.bins) < 200 else 14

        def try_delete_bin(src: int, chain_limit: int = 3) -> bool:
            if not st.bins[src]:
                return False
            # work on snapshot of src items (descending)
            src_items = st.bins[src][:]
            src_items.sort(key=lambda i: weights[i], reverse=True)

            # record moves for rollback: (type, args...)
            trail: List[Tuple] = []

            def rollback() -> None:
                for op in reversed(trail):
                    if op[0] == "move":
                        _, it, a, b = op
                        st.move_item(it, b, a)
                    else:  # swap
                        _, it1, b1, it2, b2 = op
                        st.swap_items(it1, b2, it2, b1)

            for it in src_items:
                if ls_time_exceeded():
                    rollback()
                    return False

                w = weights[it]
                dst = st.find_best_fit_bin(w, exclude=src)
                if dst != -1:
                    st.move_item(it, src, dst)
                    trail.append(("move", it, src, dst))
                    continue

                # ejection chain: try to fit it by ejecting one small item from some bin
                placed = False
                if chain_limit > 0:
                    # probe a few candidate target bins (with small residual deficit)
                    # choose bins with largest residual first (most likely with one ejection)
                    # (sample to control runtime)
                    cand_bins = list(range(len(st.bins)))
                    random.shuffle(cand_bins)
                    cand_bins = cand_bins[: min(len(cand_bins), 18)]
                    cand_bins.sort(key=lambda b: st.residuals[b], reverse=True)

                    for b in cand_bins:
                        if b == src or not st.bins[b]:
                            continue
                        # need additional space: deficit = w - residual
                        deficit = w - st.residuals[b]
                        if deficit <= 0:
                            continue
                        # try eject a small item j with weight >= deficit (best is minimal such)
                        best_j = None
                        best_wj = None
                        # scan a few smallest items in b
                        items_b = st.bins[b][:]
                        items_b.sort(key=lambda x: weights[x])
                        for j in items_b[: min(10, len(items_b))]:
                            wj = weights[j]
                            if wj >= deficit:
                                if best_wj is None or wj < best_wj:
                                    best_wj = wj
                                    best_j = j
                        if best_j is None:
                            continue

                        # try to relocate j elsewhere
                        j = best_j
                        wj = weights[j]
                        new_dst = st.find_best_fit_bin(wj, exclude=b)
                        if new_dst == -1:
                            continue

                        # perform ejection: move j out, then move it into b
                        st.move_item(j, b, new_dst)
                        trail.append(("move", j, b, new_dst))

                        # now b has more residual, should fit it
                        if st.residuals[b] >= w:
                            st.move_item(it, src, b)
                            trail.append(("move", it, src, b))
                            placed = True
                            break
                        else:
                            # rollback ejection and continue
                            rollback()
                            trail.clear()

                    if placed:
                        continue

                # cannot place it
                rollback()
                return False

            # if we got here, all moved
            if not st.bins[src]:
                st.compact()
                return True
            return False

        while attempts < max_attempts and not ls_time_exceeded():
            attempts += 1
            # choose source bin biased toward low load / high residual
            B = len(st.bins)
            if B <= 1:
                break
            # sample a few bins; pick best candidate to delete
            idxs = list(range(B))
            random.shuffle(idxs)
            idxs = idxs[: min(B, 12)]
            # prefer bins with smaller load
            idxs.sort(key=lambda b: st.bin_w[b])
            src = idxs[0]

            if try_delete_bin(src, chain_limit=3):
                # reset attempts a bit after success to intensify
                attempts = max(0, attempts - 2)

        # 2-bin merge attempt occasionally (still Phase A style)
        if not ls_time_exceeded() and len(st.bins) >= 3:
            # pick two light bins and try to repack their union into one
            light = sorted(range(len(st.bins)), key=lambda b: st.bin_w[b])
            a, b = light[0], light[1]
            if a != b and st.bin_w[a] + st.bin_w[b] <= C:
                # try simply move all items of b into a using best-fit within the same bin (trivial)
                # but to allow rearrangement, do a small greedy repack into a single bin
                items_union = st.bins[a][:] + st.bins[b][:]
                items_union.sort(key=lambda i: weights[i], reverse=True)
                if sum(weights[i] for i in items_union) <= C:
                    # empty both then fill a
                    # move items out of b then a by compacting via moves to temporary new bins is complex;
                    # instead do direct rewrite and rebuild state quickly.
                    new_bins = []
                    new_w = []
                    for bi in range(len(st.bins)):
                        if bi not in (a, b):
                            new_bins.append(st.bins[bi])
                            new_w.append(st.bin_w[bi])
                    new_bins.append(items_union)
                    new_w.append(sum(weights[i] for i in items_union))
                    st = LSState(new_bins, new_w)

        # Phase B: small bounded diversification relocations (slack-improving)
        if not ls_time_exceeded() and len(st.bins) >= 2:
            B = len(st.bins)
            sample_bins = list(range(B))
            random.shuffle(sample_bins)
            sample_bins = sample_bins[: min(B, 10)]

            best_delta = 0
            best_move = None  # (it, src, dst)
            for src in sample_bins:
                if not st.bins[src]:
                    continue
                # take a few items
                items_src = st.bins[src][:]
                random.shuffle(items_src)
                items_src = items_src[: min(10, len(items_src))]
                for it in items_src:
                    if ls_time_exceeded():
                        break
                    w = weights[it]
                    # best-fit destination using buckets
                    dst = st.find_best_fit_bin(w, exclude=src)
                    if dst == -1:
                        continue
                    # measure: reduce max residual among the two bins
                    old = max(st.residuals[src], st.residuals[dst])
                    new_src_res = st.residuals[src] + w
                    new_dst_res = st.residuals[dst] - w
                    new = max(new_src_res, new_dst_res)
                    delta = old - new
                    if delta > best_delta:
                        best_delta = delta
                        best_move = (it, src, dst)
                if ls_time_exceeded():
                    break

            if best_move is not None and best_delta > 0 and not ls_time_exceeded():
                it, src, dst = best_move
                st.move_item(it, src, dst)
                if not st.bins[src]:
                    st.compact()

        return st.bins, st.bin_w

    # ----------------------- Elite pool + path relinking -----------------------
    def residual_signature(bin_w: List[int], bucket_size: int = 10) -> Tuple[int, ...]:
        # histogram of residual buckets for diversity
        hist: Dict[int, int] = {}
        for w in bin_w:
            r = C - w
            hist[residual_bucket(r, bucket_size)] = hist.get(residual_bucket(r, bucket_size), 0) + 1
        return tuple(sorted(hist.items()))

    elite_E = 8
    elite: List[Tuple[Tuple[int, int], Tuple[int, ...], List[List[int]], List[int]]] = []

    def elite_add(p: List[List[int]], w: List[int]) -> None:
        k = solution_key(p, w)
        sig = residual_signature(w)
        # simple diversity: avoid identical (k, sig)
        for ek, esig, _, _ in elite:
            if ek == k and esig == sig:
                return
        elite.append((k, sig, [b[:] for b in p], w[:]))
        elite.sort(key=lambda x: x[0])
        if len(elite) > elite_E:
            elite.pop()

    def path_relink(p_from: List[List[int]], w_from: List[int], p_to: List[List[int]], w_to: List[int], seconds: float) -> Tuple[List[List[int]], List[int]]:
        # lightweight PR: move items toward target bin assignments when feasible
        deadline = min(start + time_limit, time.time() + max(0.0, seconds))

        # build target assignment item-> target bin id (by position in p_to)
        target_bin_of = [-1] * n
        for b, lst in enumerate(p_to):
            for it in lst:
                target_bin_of[it] = b

        st = LSState([b[:] for b in p_from], w_from[:])

        # map target bins to current bins by trying to match residual buckets
        # (coarse mapping for feasibility)
        tgt_loads = [sum(weights[it] for it in lst) for lst in p_to]
        tgt_res = [C - lw for lw in tgt_loads]

        # attempt to move a subset of mismatched items
        items_all = [it for binlst in st.bins for it in binlst]
        random.shuffle(items_all)

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
            # if already in a bin that resembles target residual, skip sometimes
            if random.random() < 0.3:
                continue

            w = weights[it]
            # choose a destination bin in current solution that best matches target residual after placing
            best_dst = -1
            best_sc = None
            for dst in range(len(st.bins)):
                if dst == src:
                    continue
                if st.bin_w[dst] + w <= C:
                    slack_after = C - (st.bin_w[dst] + w)
                    sc = abs(slack_after - tgt_res[tb])
                    if best_sc is None or sc < best_sc:
                        best_sc = sc
                        best_dst = dst
            if best_dst != -1:
                st.move_item(it, src, best_dst)
                moves += 1
                if moves % 64 == 0 and (time.time() >= deadline or time_exceeded()):
                    break

        # finish with a tiny LS slice
        return local_search(st.bins, st.bin_w, ls_seconds=min(0.02, max(0.0, deadline - time.time())))

    # ----------------------- Main GRASP loop with adaptation -----------------------
    alphas = [0.0, 0.08, 0.15, 0.25, 0.4, 0.6, 0.85]

    # constructor stats for epsilon-greedy
    cons_stats = {name: {"tries": 0, "best_bins": 10**9, "avg_bins": 0.0} for name, _ in constructors}
    alpha_stats = {a: {"tries": 0, "best_bins": 10**9, "avg_bins": 0.0} for a in alphas}

    def choose_constructor(eps: float) -> Tuple[str, callable]:
        if random.random() < eps:
            return random.choice(constructors)
        # best recent: minimize avg_bins with some smoothing; fall back to best_bins
        best = None
        best_name = None
        best_fn = None
        for name, fn in constructors:
            stt = cons_stats[name]
            if stt["tries"] == 0:
                score = 1e9
            else:
                score = stt["avg_bins"] * 0.8 + stt["best_bins"] * 0.2
            if best is None or score < best:
                best = score
                best_name, best_fn = name, fn
        return best_name, best_fn

    def choose_alpha(eps: float) -> float:
        if random.random() < eps:
            return random.choice(alphas)
        best = None
        best_a = alphas[0]
        for a in alphas:
            stt = alpha_stats[a]
            if stt["tries"] == 0:
                score = 1e9
            else:
                score = stt["avg_bins"] * 0.8 + stt["best_bins"] * 0.2
            if best is None or score < best:
                best = score
                best_a = a
        return best_a

    # iteration count (fixed), exits early on time
    max_iter = 2000 + 50 * int(len(normal_items) ** 0.5)

    no_improve = 0
    for it in range(max_iter):
        if tick(1024):
            break

        # stagnation handling -> more diversification
        if no_improve >= 80:
            eps = 0.30
            block_size = 18 if len(normal_items) < 400 else 26
            ls_slice = 0.010
        else:
            eps = 0.15
            block_size = 10 if len(normal_items) < 300 else 18
            ls_slice = 0.015

        name, cons = choose_constructor(eps)
        alpha = choose_alpha(eps)

        # constructor-specific params
        if name == "RCBFD":
            p, w = cons(alpha, block_size, bucket_size=max(5, min(25, C // 40 + 1)))
        elif name == "RBFD":
            p, w = cons(alpha, block_size)
        elif name == "RFFD":
            p, w = cons(alpha, block_size)
        else:  # PAIR
            p, w = cons(alpha, block_size)

        if tick(1024):
            break

        # update constructor/alpha stats based on construction bins
        bins0 = len(p)
        cs = cons_stats[name]
        cs["tries"] += 1
        cs["avg_bins"] += (bins0 - cs["avg_bins"]) / cs["tries"]
        cs["best_bins"] = min(cs["best_bins"], bins0)

        as_ = alpha_stats[alpha]
        as_["tries"] += 1
        as_["avg_bins"] += (bins0 - as_["avg_bins"]) / as_["tries"]
        as_["best_bins"] = min(as_["best_bins"], bins0)

        # local search with per-call time slice proportional to time_limit and remaining time
        if time_limit > 0:
            remaining = max(0.0, time_limit - (time.time() - start))
        else:
            remaining = 0.0
        # keep predictable and small, but allow a bit more when plenty of time remains
        ls_seconds = min(max(ls_slice * time_limit, 0.004), 0.06)
        ls_seconds = min(ls_seconds, max(0.0, remaining * 0.05))

        p, w = local_search(p, w, ls_seconds=ls_seconds)

        k = solution_key(p, w)
        if k < best_key:
            best_key = k
            best_p, best_w = deep_copy_solution(p, w)
            elite_add(best_p, best_w)
            no_improve = 0
        else:
            no_improve += 1
            # still add good solutions to elite
            if len(elite) < elite_E or k <= elite[-1][0]:
                elite_add(p, w)

        # Path relinking occasionally (GRASP intensification)
        if elite and (it % 45 == 0) and not time_exceeded():
            # when near-best or stagnating
            if no_improve >= 25 or k[0] <= best_key[0] + 1:
                _, _, ep, ew = random.choice(elite)
                pr_seconds = min(0.01 * time_limit, 0.05)
                pr_p, pr_w = path_relink(p, w, ep, ew, seconds=pr_seconds)
                pr_k = solution_key(pr_p, pr_w)
                if pr_k < best_key:
                    best_key = pr_k
                    best_p, best_w = deep_copy_solution(pr_p, pr_w)
                    elite_add(best_p, best_w)
                    no_improve = 0

    # finalize: remove empty bins, compute weights, and append overweight bins
    final_p = [b for b in best_p if b]
    final_w = [sum(weights[i] for i in b) for b in final_p]

    # add overweight bins at end
    final_p.extend(overweight_bins)
    final_w.extend(overweight_w)

    return {"packing": final_p, "bin_weights": final_w}
