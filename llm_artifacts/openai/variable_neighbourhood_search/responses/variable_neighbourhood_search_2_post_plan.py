import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = int(bin_capacity)
    w = weights
    n = len(w)

    start = time.time()
    # Cap at 100s as in plan; still respects caller time_limit.
    deadline = start + min(max(float(time_limit), 0.0), 100.0)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ---------------------------
    # Secondary objective (incremental): tuple (slack_sum, slack2_sum, max_slack)
    # Lower is better lexicographically among same #bins.
    # ---------------------------
    def sec_contrib(load: int) -> Tuple[int, int, int]:
        s = C - load
        return (s, s * s, s)

    def sec_add(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (a[0] + b[0], a[1] + b[1], max(a[2], b[2]))

    def sec_sub(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
        # max_slack cannot be maintained by subtraction; we will keep max_slack approximate incrementally
        # and recompute it during periodic rebuilds.
        return (a[0] - b[0], a[1] - b[1], a[2])

    # ---------------------------
    # Stateful solution with incremental positions, slack buckets, undo moves, periodic rebuild.
    # ---------------------------
    class SolutionState:
        __slots__ = (
            "bins", "loads", "pos_bin", "pos_idx", "alive",
            "B", "sec", "bins_by_slack", "empties",
            "max_slack_est", "moves_since_rebuild"
        )

        def __init__(self, bins: List[List[int]], loads: List[int]):
            self.bins = [lst[:] for lst in bins]
            self.loads = loads[:]
            self.pos_bin = [-1] * n
            self.pos_idx = [-1] * n
            self.alive = [True] * len(self.bins)
            self.B = len(self.bins)
            self.sec = (0, 0, 0)
            self.bins_by_slack = [[] for _ in range(C + 1)]  # slack -> list of bin indices (with duplicates possible; filtered by alive)
            self.empties = 0
            self.max_slack_est = 0
            self.moves_since_rebuild = 0
            self._rebuild_full()

        def clone_bins_loads_compact(self) -> Tuple[List[List[int]], List[int]]:
            self.compact()
            return ([lst[:] for lst in self.bins], self.loads[:])

        def _rebuild_full(self) -> None:
            # recompute positions, alive, buckets, sec, B
            B = len(self.bins)
            self.alive = [True] * B
            self.B = B
            self.pos_bin = [-1] * n
            self.pos_idx = [-1] * n
            self.bins_by_slack = [[] for _ in range(C + 1)]
            slack_sum = 0
            slack2_sum = 0
            max_slack = 0
            for b in range(B):
                L = self.loads[b]
                s = C - L
                if s < 0:
                    # Should not happen, but keep safe.
                    s = 0
                self.bins_by_slack[s].append(b)
                slack_sum += s
                slack2_sum += s * s
                if s > max_slack:
                    max_slack = s
                for j, it in enumerate(self.bins[b]):
                    self.pos_bin[it] = b
                    self.pos_idx[it] = j
            self.sec = (slack_sum, slack2_sum, max_slack)
            self.max_slack_est = max_slack
            self.empties = 0
            self.moves_since_rebuild = 0

        def compact(self) -> None:
            # Remove empty/dead bins and reindex; O(total items).
            new_bins: List[List[int]] = []
            new_loads: List[int] = []
            for b, alive in enumerate(self.alive):
                if not alive:
                    continue
                if not self.bins[b]:
                    continue
                new_bins.append(self.bins[b])
                new_loads.append(self.loads[b])
            self.bins = new_bins
            self.loads = new_loads
            self._rebuild_full()

        def _bucket_remove(self, slack: int, b: int) -> None:
            # Lazy buckets: we do not remove; we only add and later filter by alive and slack match.
            # So no-op.
            return

        def _bucket_add(self, slack: int, b: int) -> None:
            if 0 <= slack <= C:
                self.bins_by_slack[slack].append(b)
                if slack > self.max_slack_est:
                    self.max_slack_est = slack

        def _update_sec_for_bin(self, b: int, old_load: int, new_load: int) -> None:
            # Update sum components; max_slack is approximate and fixed on rebuild.
            old_s = C - old_load
            new_s = C - new_load
            self.sec = (self.sec[0] - old_s + new_s, self.sec[1] - old_s * old_s + new_s * new_s, self.sec[2])

        def ensure_bin(self) -> int:
            # Create new empty bin.
            b = len(self.bins)
            self.bins.append([])
            self.loads.append(0)
            self.alive.append(True)
            self.B += 1
            self._bucket_add(C, b)
            return b

        # ---- Move primitives with undo ----
        # Undo records are tuples with a leading tag.

        def relocate(self, it: int, to_b: int) -> Optional[Tuple]:
            b_from = self.pos_bin[it]
            if b_from < 0:
                return None
            if to_b == b_from:
                return None
            wi = w[it]

            created = False
            if to_b == len(self.bins):
                to_b = self.ensure_bin()
                created = True

            if not self.alive[to_b]:
                # Should not target dead bin.
                return None
            if self.loads[to_b] + wi > C:
                if created:
                    # revert bin creation quickly by marking dead (will be compacted later)
                    self.alive[to_b] = False
                    self.B -= 1
                return None

            # update buckets + sec for from/to bins
            old_load_from = self.loads[b_from]
            old_load_to = self.loads[to_b]

            # remove item from b_from (swap-pop)
            idx = self.pos_idx[it]
            last_it = self.bins[b_from][-1]
            self.bins[b_from][idx] = last_it
            self.pos_idx[last_it] = idx
            self.bins[b_from].pop()

            self.loads[b_from] -= wi
            self._update_sec_for_bin(b_from, old_load_from, self.loads[b_from])
            self._bucket_add(C - self.loads[b_from], b_from)

            # add to to_b
            self.pos_bin[it] = to_b
            self.pos_idx[it] = len(self.bins[to_b])
            self.bins[to_b].append(it)
            self.loads[to_b] += wi
            self._update_sec_for_bin(to_b, old_load_to, self.loads[to_b])
            self._bucket_add(C - self.loads[to_b], to_b)

            # mark empty bin as dead lazily (keep structure)
            from_became_empty = (len(self.bins[b_from]) == 0)
            if from_became_empty:
                self.alive[b_from] = False
                self.B -= 1
                self.empties += 1

            self.moves_since_rebuild += 1
            return ("rel", it, b_from, to_b, idx, last_it, created, from_became_empty)

        def undo(self, rec: Tuple) -> None:
            tag = rec[0]
            if tag == "rel":
                _, it, b_from, b_to, idx, last_it, created, from_became_empty = rec
                wi = w[it]

                # If source bin was marked dead due to emptiness, revive it.
                if from_became_empty:
                    self.alive[b_from] = True
                    self.B += 1
                    self.empties -= 1

                # update sec for bins b_from and b_to (reverse)
                old_load_from = self.loads[b_from]
                old_load_to = self.loads[b_to]

                # remove it from b_to (swap-pop at its position)
                it_pos = self.pos_idx[it]
                last2 = self.bins[b_to][-1]
                self.bins[b_to][it_pos] = last2
                self.pos_idx[last2] = it_pos
                self.bins[b_to].pop()
                self.loads[b_to] -= wi
                self._update_sec_for_bin(b_to, old_load_to, self.loads[b_to])
                self._bucket_add(C - self.loads[b_to], b_to)

                # insert back into b_from at idx by appending then swapping into place
                self.pos_bin[it] = b_from
                self.pos_idx[it] = len(self.bins[b_from])
                self.bins[b_from].append(it)
                self.loads[b_from] += wi
                # swap into original idx
                if idx != self.pos_idx[it]:
                    other = self.bins[b_from][idx]
                    self.bins[b_from][idx] = it
                    self.bins[b_from][-1] = other
                    self.pos_idx[it] = idx
                    self.pos_idx[other] = len(self.bins[b_from]) - 1
                # restore last_it mapping (it was used only for swap-pop consistency; no extra action needed)
                self._update_sec_for_bin(b_from, old_load_from, self.loads[b_from])
                self._bucket_add(C - self.loads[b_from], b_from)

                if created:
                    # Remove the created bin by marking dead (must be empty now)
                    if len(self.bins[b_to]) == 0:
                        self.alive[b_to] = False
                        self.B -= 1
                        self.empties += 1

            elif tag == "swp":
                _, a, b, ba, bb, ia, ib, old_la, old_lb = rec
                wa, wb = w[a], w[b]

                # undo swap in lists
                self.bins[ba][ia], self.bins[bb][ib] = self.bins[bb][ib], self.bins[ba][ia]
                self.pos_bin[a], self.pos_bin[b] = ba, bb
                self.pos_idx[a], self.pos_idx[b] = ia, ib

                # undo loads + sec
                cur_la = self.loads[ba]
                cur_lb = self.loads[bb]
                self.loads[ba] = old_la
                self.loads[bb] = old_lb
                self._update_sec_for_bin(ba, cur_la, old_la)
                self._update_sec_for_bin(bb, cur_lb, old_lb)
                self._bucket_add(C - old_la, ba)
                self._bucket_add(C - old_lb, bb)

            else:
                # Unknown undo tag; ignore.
                return

        def swap(self, a: int, b_it: int) -> Optional[Tuple]:
            ba = self.pos_bin[a]
            bb = self.pos_bin[b_it]
            if ba < 0 or bb < 0 or ba == bb:
                return None
            if not self.alive[ba] or not self.alive[bb]:
                return None
            wa = w[a]
            wb = w[b_it]
            if self.loads[ba] - wa + wb > C:
                return None
            if self.loads[bb] - wb + wa > C:
                return None

            ia = self.pos_idx[a]
            ib = self.pos_idx[b_it]

            old_la = self.loads[ba]
            old_lb = self.loads[bb]

            # apply swap
            self.bins[ba][ia], self.bins[bb][ib] = self.bins[bb][ib], self.bins[ba][ia]
            self.pos_bin[a], self.pos_bin[b_it] = bb, ba
            self.pos_idx[a], self.pos_idx[b_it] = ib, ia

            self.loads[ba] = old_la - wa + wb
            self.loads[bb] = old_lb - wb + wa
            self._update_sec_for_bin(ba, old_la, self.loads[ba])
            self._update_sec_for_bin(bb, old_lb, self.loads[bb])
            self._bucket_add(C - self.loads[ba], ba)
            self._bucket_add(C - self.loads[bb], bb)

            self.moves_since_rebuild += 1
            return ("swp", a, b_it, ba, bb, ia, ib, old_la, old_lb)

        # ---- Candidate bins using slack buckets ----
        def candidate_bins_for_weight(self, wi: int, exclude: int = -1, window: int = 12, limit: int = 30) -> List[int]:
            # Consider bins with slack in [wi .. wi+window], i.e., tight fits first.
            res: List[int] = []
            seen = set()
            hi = min(C, wi + window)
            for s in range(wi, hi + 1):
                bucket = self.bins_by_slack[s]
                # iterate from end (more recent)
                for b in reversed(bucket[-60:]):
                    if b == exclude:
                        continue
                    if b in seen:
                        continue
                    if b >= len(self.bins) or not self.alive[b]:
                        continue
                    if C - self.loads[b] != s:
                        continue
                    seen.add(b)
                    res.append(b)
                    if len(res) >= limit:
                        return res
            # Fallback: a few random alive bins if not enough
            if len(res) < min(6, limit) and self.B > 0:
                trials = 0
                while len(res) < min(10, limit) and trials < 60:
                    trials += 1
                    b = random.randrange(len(self.bins))
                    if b == exclude or b in seen:
                        continue
                    if b >= len(self.bins) or not self.alive[b]:
                        continue
                    if self.loads[b] + wi <= C:
                        seen.add(b)
                        res.append(b)
            return res

        def alive_bins_sorted_by_load(self, limit: Optional[int] = None) -> List[int]:
            bins = [b for b in range(len(self.bins)) if self.alive[b] and self.bins[b]]
            bins.sort(key=lambda x: self.loads[x])
            if limit is not None and len(bins) > limit:
                return bins[:limit]
            return bins

    # ---------------------------
    # Construction: multi-start FFD/BFD + randomized within equal-weight blocks + lookahead tie-break.
    # ---------------------------
    items_of_weight: Dict[int, List[int]] = {}
    for i, wi in enumerate(w):
        items_of_weight.setdefault(wi, []).append(i)

    weights_sorted_unique = sorted(items_of_weight.keys(), reverse=True)

    def construct(order: List[int], mode: str, freq: Dict[int, int]) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        loads: List[int] = []

        for it in order:
            wi = w[it]
            best_j = -1
            best_key = None
            if mode == "FFD":
                # first-fit decreasing
                for j, L in enumerate(loads):
                    if L + wi <= C:
                        best_j = j
                        break
            else:
                # best-fit with lookahead tie-break
                for j, L in enumerate(loads):
                    rem = C - (L + wi)
                    if rem < 0:
                        continue
                    # tie-break: prefer rem that matches some heavy items frequency
                    # Use freq[rem] (items of weight rem) as a proxy, and smaller rem.
                    key = (rem, -freq.get(rem, 0))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_j = j
            if best_j == -1:
                bins.append([it])
                loads.append(wi)
            else:
                bins[best_j].append(it)
                loads[best_j] += wi
        return bins, loads

    def order_desc_deterministic() -> List[int]:
        return sorted(range(n), key=lambda i: (-w[i], i))

    def order_desc_randomized() -> List[int]:
        # Shuffle within equal-weight blocks: stable by weight groups.
        order: List[int] = []
        for wt in weights_sorted_unique:
            block = items_of_weight[wt][:]
            random.shuffle(block)
            order.extend(block)
        return order

    def order_desc_slight_noise() -> List[int]:
        # Noise only within same weight via random tie-break
        return sorted(range(n), key=lambda i: (-w[i], random.random()))

    # Frequency table for lookahead tie-break (for rem weights)
    freq_weight = {wt: len(lst) for wt, lst in items_of_weight.items()}

    # Initial multi-start slice
    init_best_bins: List[List[int]] = []
    init_best_loads: List[int] = []
    init_best_key = (10**9, (10**18, 10**18, 10**9))

    def eval_key(bins: List[List[int]], loads: List[int]) -> Tuple[int, Tuple[int, int, int]]:
        B = len(bins)
        slack_sum = 0
        slack2_sum = 0
        max_slack = 0
        for L in loads:
            s = C - L
            slack_sum += s
            slack2_sum += s * s
            if s > max_slack:
                max_slack = s
        return (B, (slack_sum, slack2_sum, max_slack))

    init_deadline = start + 0.04 * (deadline - start)  # ~4% of budget
    starts = 0
    while starts < 12 and time.time() < init_deadline:
        starts += 1
        if starts == 1:
            order = order_desc_deterministic()
            mode = "BFD"
        elif starts == 2:
            order = order_desc_deterministic()
            mode = "FFD"
        elif starts % 3 == 0:
            order = order_desc_randomized()
            mode = "BFD"
        elif starts % 3 == 1:
            order = order_desc_slight_noise()
            mode = "BFD"
        else:
            order = order_desc_randomized()
            mode = "FFD"

        bins, loads = construct(order, mode, freq_weight)
        key = eval_key(bins, loads)
        if key < init_best_key:
            init_best_key = key
            init_best_bins, init_best_loads = bins, loads

    state = SolutionState(init_best_bins, init_best_loads)

    # ---------------------------
    # VND neighborhoods N1..N4
    # ---------------------------
    def maybe_rebuild(st: SolutionState) -> None:
        # Periodic rebuild/compact
        if st.empties > 0.15 * max(1, len(st.bins)) or st.moves_since_rebuild > 800:
            st.compact()
            # recompute exact max slack
            st.sec = (st.sec[0], st.sec[1], max(C - L for L in st.loads) if st.loads else 0)
            st.max_slack_est = st.sec[2]

    def n1_empty_one_bin(st: SolutionState, bin_limit: int = 30, cand_per_item: int = 18) -> bool:
        # Try to remove one of the lightest bins with limited backtracking depth 2.
        src_bins = st.alive_bins_sorted_by_load(limit=bin_limit)
        for b in src_bins:
            if time.time() >= deadline:
                return False
            if not st.alive[b] or not st.bins[b]:
                continue

            items = st.bins[b][:]
            items.sort(key=lambda it: -w[it])
            move_stack: List[Tuple] = []

            def attempt(idx_item: int, backtrack_budget: int) -> bool:
                if idx_item >= len(items):
                    return True
                it = items[idx_item]
                if st.pos_bin[it] != b:
                    # already moved
                    return attempt(idx_item + 1, backtrack_budget)
                wi = w[it]
                candidates = st.candidate_bins_for_weight(wi, exclude=b, window=10, limit=cand_per_item)
                # Sort by resulting slack (best-fit) and prefer bins that become near-full
                candidates.sort(key=lambda tb: (C - (st.loads[tb] + wi),))
                for tb in candidates:
                    rec = st.relocate(it, tb)
                    if rec is None:
                        continue
                    move_stack.append(rec)
                    if attempt(idx_item + 1, backtrack_budget):
                        return True
                    # undo
                    st.undo(move_stack.pop())

                # limited backtracking: undo one previous move and retry with alternative paths
                if backtrack_budget > 0 and move_stack:
                    last = move_stack.pop()
                    st.undo(last)
                    return attempt(idx_item, backtrack_budget - 1)
                return False

            ok = attempt(0, backtrack_budget=2)
            if ok:
                # If bin b is now empty, it is already marked dead by relocate operations.
                maybe_rebuild(st)
                return True
            else:
                # rollback all
                while move_stack:
                    st.undo(move_stack.pop())

        return False

    def n2_ejection_2step(st: SolutionState, trials_bins: int = 18, K_targets: int = 14) -> bool:
        # 2-1 exchange / ejection: for an item x in light bin b, force into target by ejecting y.
        src_bins = st.alive_bins_sorted_by_load(limit=trials_bins)
        for b in src_bins:
            if time.time() >= deadline:
                return False
            if not st.alive[b] or not st.bins[b]:
                continue
            # try harder items first
            items = st.bins[b][:]
            items.sort(key=lambda it: -w[it])
            for x in items:
                if time.time() >= deadline:
                    return False
                if st.pos_bin[x] != b:
                    continue
                wx = w[x]
                # if x already fits somewhere, N1 would handle; focus on those that do not fit tight buckets
                targets = st.candidate_bins_for_weight(wx, exclude=b, window=18, limit=K_targets)
                random.shuffle(targets)
                # consider targets where x doesn't fit but could with ejection
                for t in targets:
                    if not st.alive[t] or t == b:
                        continue
                    slack_t = C - st.loads[t]
                    if slack_t >= wx:
                        continue
                    need = wx - slack_t
                    # eject a small item y from t with weight >= need (prefer smallest such)
                    cand_y = None
                    cand_y_w = 10**9
                    for y in st.bins[t]:
                        wy = w[y]
                        if wy >= need and wy < cand_y_w:
                            cand_y = y
                            cand_y_w = wy
                    if cand_y is None:
                        continue

                    # Perform swap x <-> y if feasible
                    # after swap, x in t, y in b (temporarily) then try move y elsewhere
                    rec_swp = st.swap(x, cand_y)
                    if rec_swp is None:
                        continue

                    # now y is in bin b; try to relocate y to some bin != b
                    y = cand_y
                    wy = w[y]
                    cands2 = st.candidate_bins_for_weight(wy, exclude=b, window=14, limit=18)
                    # allow creating a new bin in last resort sometimes
                    if random.random() < 0.25:
                        cands2.append(len(st.bins))
                    placed = False
                    for tb2 in cands2:
                        if tb2 == b:
                            continue
                        rec_rel = st.relocate(y, tb2)
                        if rec_rel is None:
                            continue
                        # success: configuration changed
                        placed = True
                        maybe_rebuild(st)
                        return True
                    # rollback
                    st.undo(rec_swp)

        return False

    def n3_guided_swaps(st: SolutionState, attempts: int = 220) -> bool:
        # Guided 1-1 swaps: focus on bins with large slack and bins with awkward slack.
        alive_bins = [b for b in range(len(st.bins)) if st.alive[b] and st.bins[b]]
        if len(alive_bins) < 2:
            return False

        # Build a small list of top slack bins
        alive_bins.sort(key=lambda b: C - st.loads[b], reverse=True)
        slack_bins = alive_bins[: min(12, len(alive_bins))]

        for _ in range(attempts):
            if time.time() >= deadline:
                return False
            b1 = random.choice(slack_bins)
            # pick another bin biased to light bins (potentially removable)
            b2 = random.choice(st.alive_bins_sorted_by_load(limit=30))
            if b1 == b2 or not st.alive[b2] or not st.bins[b2] or not st.bins[b1]:
                continue

            # pick item a from b2 (heavier) and item b from b1 (lighter) to improve fit
            a = max(st.bins[b2], key=lambda it: w[it])
            b_it = min(st.bins[b1], key=lambda it: w[it])

            # accept if swap improves secondary (slack_sum, slack2) or makes b2 lighter
            ba = st.pos_bin[a]
            bb = st.pos_bin[b_it]
            if ba < 0 or bb < 0 or ba == bb:
                continue

            # compute delta secondary quickly
            old_a = st.loads[ba]
            old_b = st.loads[bb]
            wa, wb = w[a], w[b_it]
            new_a = old_a - wa + wb
            new_b = old_b - wb + wa
            if new_a > C or new_b > C:
                continue
            old_sl2 = (C - old_a) * (C - old_a) + (C - old_b) * (C - old_b)
            new_sl2 = (C - new_a) * (C - new_a) + (C - new_b) * (C - new_b)
            if new_sl2 > old_sl2 and random.random() > 0.15:
                continue

            rec = st.swap(a, b_it)
            if rec is not None:
                maybe_rebuild(st)
                return True

        return False

    def n4_fill_holes_relocate(st: SolutionState, attempts: int = 260) -> bool:
        # Move items into high-slack bins to make some other bin emptyable.
        alive_bins = [b for b in range(len(st.bins)) if st.alive[b] and st.bins[b]]
        if not alive_bins:
            return False
        alive_bins.sort(key=lambda b: C - st.loads[b], reverse=True)
        hole_bins = alive_bins[: min(16, len(alive_bins))]

        for _ in range(attempts):
            if time.time() >= deadline:
                return False
            hb = random.choice(hole_bins)
            slack = C - st.loads[hb]
            if slack <= 0:
                continue

            # choose a candidate weight close to slack
            target_w = slack
            # try exact matches first, then a small downward range
            cand_items: List[int] = []
            for wt in range(target_w, max(1, target_w - 6), -1):
                lst = items_of_weight.get(wt)
                if lst:
                    cand_items.extend(random.sample(lst, k=min(3, len(lst))))
            if not cand_items:
                continue

            random.shuffle(cand_items)
            for it in cand_items:
                if st.pos_bin[it] < 0:
                    continue
                bf = st.pos_bin[it]
                if bf == hb:
                    continue
                wi = w[it]
                if wi > slack:
                    continue
                # bias: don’t move out of very tight bins unless it helps
                if (C - st.loads[bf]) <= 1 and random.random() < 0.6:
                    continue
                rec = st.relocate(it, hb)
                if rec is not None:
                    maybe_rebuild(st)
                    return True

        return False

    def vnd(st: SolutionState) -> None:
        # Deterministic schedule N1->N2->N3->N4, restart on improvements.
        # Stop when no neighborhood changes solution.
        changed = True
        while changed and time.time() < deadline:
            changed = False
            prevB = st.B
            prevSec = st.sec

            if n1_empty_one_bin(st):
                changed = True
                continue
            if n2_ejection_2step(st):
                changed = True
                # if bins decreased or sec improved, restart at N1
                if st.B < prevB or st.sec < prevSec:
                    continue
            if n3_guided_swaps(st):
                changed = True
                if st.B < prevB or st.sec < prevSec:
                    continue
            if n4_fill_holes_relocate(st):
                changed = True

    # Initial improvement
    vnd(state)
    state.compact()

    best_bins, best_loads = state.clone_bins_loads_compact()
    best_key = (len(best_bins), eval_key(best_bins, best_loads)[1])

    # ---------------------------
    # Guided shaking + regret repair
    # ---------------------------
    def remove_items(st: SolutionState, items: List[int]) -> List[Tuple]:
        # remove by relocating to a dead marker (pos_bin=-1) via manual extraction with undo recs
        # We implement as: relocate to a new bin then mark bin dead? Too costly.
        # Instead: custom extraction without undo; we will repair from scratch on selected bins.
        # Here, we only used in LNS where we rebuild/compact after.
        removed = []
        for it in items:
            b = st.pos_bin[it]
            if b < 0 or not st.alive[b]:
                continue
            wi = w[it]
            old_load_b = st.loads[b]
            idx = st.pos_idx[it]
            last = st.bins[b][-1]
            st.bins[b][idx] = last
            st.pos_idx[last] = idx
            st.bins[b].pop()
            st.loads[b] -= wi
            st._update_sec_for_bin(b, old_load_b, st.loads[b])
            st._bucket_add(C - st.loads[b], b)
            st.pos_bin[it] = -1
            st.pos_idx[it] = -1
            removed.append(it)
            if len(st.bins[b]) == 0 and st.alive[b]:
                st.alive[b] = False
                st.B -= 1
                st.empties += 1
        st.moves_since_rebuild += len(removed)
        return removed

    def regret_repair(st: SolutionState, removed: List[int], k_regret: int = 2) -> None:
        # Regret-2/3 insertion: choose item with largest regret (2nd-best - best).
        # Bounded candidate bins.
        unplaced = removed[:]
        unplaced.sort(key=lambda it: -w[it])

        while unplaced and time.time() < deadline:
            best_choice = None  # (regret, -weight, item, best_bin)
            best_bin_for_item = None

            for it in unplaced[: min(len(unplaced), 80)]:
                wi = w[it]
                cands = st.candidate_bins_for_weight(wi, exclude=-1, window=18, limit=28)
                fits = []
                for b in cands:
                    if st.loads[b] + wi <= C:
                        rem = C - (st.loads[b] + wi)
                        fits.append((rem, b))
                fits.sort()
                # allow new bin option
                fits.append((C - wi, len(st.bins)))

                if not fits:
                    continue

                best_rem, best_b = fits[0]
                if k_regret >= 3 and len(fits) >= 3:
                    regret = fits[2][0] - best_rem
                elif len(fits) >= 2:
                    regret = fits[1][0] - best_rem
                else:
                    regret = 10**9

                choice = (regret, -wi, it, best_b)
                if best_choice is None or choice > best_choice:
                    best_choice = choice
                    best_bin_for_item = best_b

            if best_choice is None:
                # place the heaviest remaining in a new bin
                it = unplaced.pop(0)
                bnew = st.ensure_bin()
                # direct append
                st.pos_bin[it] = bnew
                st.pos_idx[it] = len(st.bins[bnew])
                st.bins[bnew].append(it)
                oldL = st.loads[bnew]
                st.loads[bnew] += w[it]
                st._update_sec_for_bin(bnew, oldL, st.loads[bnew])
                st._bucket_add(C - st.loads[bnew], bnew)
                continue

            it = best_choice[2]
            b = best_bin_for_item
            unplaced.remove(it)
            if b == len(st.bins):
                b = st.ensure_bin()

            # insert
            st.pos_bin[it] = b
            st.pos_idx[it] = len(st.bins[b])
            st.bins[b].append(it)
            oldL = st.loads[b]
            st.loads[b] += w[it]
            st._update_sec_for_bin(b, oldL, st.loads[b])
            st._bucket_add(C - st.loads[b], b)

        maybe_rebuild(st)

    def shake(st: SolutionState, k: int) -> None:
        # Guided shaking matched to neighborhoods.
        maybe_rebuild(st)
        st.compact() if st.empties > 0 else None

        alive_bins = [b for b in range(len(st.bins)) if st.alive[b] and st.bins[b]]
        if not alive_bins:
            return

        if k == 1:
            # targeted single relocation from a near-full bin to a looser bin
            alive_bins.sort(key=lambda b: st.loads[b], reverse=True)
            b_from = alive_bins[0]
            it = random.choice(st.bins[b_from])
            wi = w[it]
            # choose a target with big slack (loose)
            alive_bins.sort(key=lambda b: C - st.loads[b], reverse=True)
            for tb in alive_bins[: min(12, len(alive_bins))]:
                if tb == b_from:
                    continue
                if st.loads[tb] + wi <= C:
                    st.relocate(it, tb)
                    break
            maybe_rebuild(st)
            return

        if k == 2:
            # ejection-style shake (force insert into best target by swapping out)
            alive_bins.sort(key=lambda b: st.loads[b])
            b_light = alive_bins[0]
            x = random.choice(st.bins[b_light])
            wx = w[x]
            targets = st.candidate_bins_for_weight(max(1, wx // 2), exclude=b_light, window=25, limit=18)
            random.shuffle(targets)
            for t in targets:
                slack_t = C - st.loads[t]
                if slack_t >= wx:
                    # direct move (still can diversify)
                    st.relocate(x, t)
                    return
                need = wx - slack_t
                # eject y with wy>=need
                y = None
                best_wy = 10**9
                for cand in st.bins[t]:
                    wy = w[cand]
                    if wy >= need and wy < best_wy:
                        y = cand
                        best_wy = wy
                if y is None:
                    continue
                rec = st.swap(x, y)
                if rec is None:
                    continue
                # reinsert y elsewhere (or new bin)
                c2 = st.candidate_bins_for_weight(w[y], exclude=st.pos_bin[y], window=20, limit=18)
                if random.random() < 0.35:
                    c2.append(len(st.bins))
                for tb2 in c2:
                    if tb2 == st.pos_bin[y]:
                        continue
                    if st.relocate(y, tb2) is not None:
                        maybe_rebuild(st)
                        return
                st.undo(rec)
            maybe_rebuild(st)
            return

        if k == 3:
            # bin split and repack: choose 2-3 bins (lightest + medium)
            alive_bins.sort(key=lambda b: st.loads[b])
            chosen_bins = alive_bins[:1]
            if len(alive_bins) > 1:
                chosen_bins.append(alive_bins[len(alive_bins) // 2])
            if len(alive_bins) > 2 and random.random() < 0.6:
                chosen_bins.append(random.choice(alive_bins[1:]))
            chosen_bins = list(dict.fromkeys(chosen_bins))

            removed_items: List[int] = []
            for b in chosen_bins:
                removed_items.extend(st.bins[b][:])
            remove_items(st, removed_items)
            maybe_rebuild(st)
            regret_repair(st, removed_items, k_regret=2)
            maybe_rebuild(st)
            return

        # k>=4: larger LNS destroy/repair
        # remove all items from m bins (increases with k) + some random items
        m = min(2 + (k - 3), max(2, len(alive_bins) // 6 + 1))
        alive_bins.sort(key=lambda b: st.loads[b])
        bins_to_remove = alive_bins[: min(m, len(alive_bins))]
        if len(alive_bins) > m and random.random() < 0.5:
            bins_to_remove.append(random.choice(alive_bins[m:]))
        bins_to_remove = list(dict.fromkeys(bins_to_remove))

        removed_items: List[int] = []
        for b in bins_to_remove:
            removed_items.extend(st.bins[b][:])

        # additional random removals
        r_extra = min(n, 3 + 2 * k)
        for _ in range(r_extra):
            it = random.randrange(n)
            if st.pos_bin[it] >= 0:
                removed_items.append(it)

        removed_items = list(dict.fromkeys([it for it in removed_items if st.pos_bin[it] >= 0]))
        remove_items(st, removed_items)
        maybe_rebuild(st)
        regret_repair(st, removed_items, k_regret=3 if k >= 6 else 2)
        maybe_rebuild(st)

    # ---------------------------
    # Main VNS loop with new acceptance
    # ---------------------------
    # Fixed iteration budget (scaled, capped). Time checked periodically.
    iter_budget = min(60000, 2500 + 60 * n)

    # Sideways acceptance among equal-B solutions
    p_side = 0.06

    def accept(curr_key, cand_key) -> bool:
        (cB, cSec) = curr_key
        (nB, nSec) = cand_key
        if nB < cB:
            return True
        if nB > cB:
            return False
        # equal B
        if nSec < cSec:
            return True
        # occasional sideways acceptance with bounded worsening
        # delta scales with bins and capacity (slack_sum can vary up to B*C)
        delta = max(5, (C * max(1, cB)) // 60)
        if random.random() < p_side:
            if nSec[0] <= cSec[0] + delta and nSec[1] <= cSec[1] + delta * delta * 2:
                return True
        return False

    curr = state
    curr.compact()
    curr_key = (curr.B, curr.sec)

    bestB = best_key[0]
    stagnation = 0

    kmax_base = 10
    check_every = 220
    ops = 0

    for it_outer in range(iter_budget):
        ops += 1
        if ops % check_every == 0 and time.time() >= deadline:
            break

        # adaptive diversification via stagnation
        kmax = kmax_base
        if stagnation > 40:
            kmax = min(14, kmax_base + 3)
        if stagnation > 120:
            kmax = min(18, kmax_base + 6)

        k = 1
        while k <= kmax:
            ops += 1
            if ops % check_every == 0 and time.time() >= deadline:
                break

            # Create candidate by shaking a compact copy (avoid heavy clone too often: clone bins/loads only)
            cbins, cloads = curr.clone_bins_loads_compact()
            cand = SolutionState(cbins, cloads)

            shake(cand, k)
            vnd(cand)
            cand.compact()

            cand_key = (cand.B, cand.sec)

            if accept(curr_key, cand_key):
                curr = cand
                curr_key = cand_key
                k = 1
            else:
                k += 1

            # Update best
            if cand_key < best_key:
                best_key = cand_key
                best_bins, best_loads = cand.clone_bins_loads_compact()
                if best_key[0] < bestB:
                    bestB = best_key[0]
                    stagnation = 0
                else:
                    stagnation = 0
            else:
                stagnation += 1

        # periodic restart tied to stagnation
        ops += 1
        if ops % check_every == 0 and time.time() >= deadline:
            break
        if it_outer % 45 == 0 or (stagnation > 80 and it_outer % 15 == 0):
            # restart from best, then medium shake
            curr = SolutionState([lst[:] for lst in best_bins], best_loads[:])
            shake(curr, random.choice([3, 4, 5]))
            vnd(curr)
            curr.compact()
            curr_key = (curr.B, curr.sec)

    # Final best cleanup (already compact)
    # Safety: ensure indices unique and loads computed; avoid heavy checks.
    return {"packing": best_bins, "bin_weights": best_loads}
