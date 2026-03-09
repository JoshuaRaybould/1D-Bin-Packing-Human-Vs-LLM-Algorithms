import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = int(bin_capacity)
    w = weights
    n = len(w)

    start = time.time()
    deadline = start + min(max(float(time_limit), 0.0), 100.0)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ---------------------------
    # Helpers
    # ---------------------------
    def now() -> float:
        return time.time()

    # Secondary evaluation: prefer tight packings among same #bins
    # Use (slack2_sum, slack_sum, max_slack) (lexicographic)
    def eval_key(bins: List[List[int]], loads: List[int]) -> Tuple[int, Tuple[int, int, int]]:
        slack_sum = 0
        slack2_sum = 0
        max_slack = 0
        for L in loads:
            s = C - L
            slack_sum += s
            slack2_sum += s * s
            if s > max_slack:
                max_slack = s
        return (len(bins), (slack2_sum, slack_sum, max_slack))

    # weight -> items indices
    items_of_weight: Dict[int, List[int]] = {}
    for i, wi in enumerate(w):
        items_of_weight.setdefault(wi, []).append(i)
    unique_wts_desc = sorted(items_of_weight.keys(), reverse=True)
    freq_weight = {wt: len(lst) for wt, lst in items_of_weight.items()}

    # ---------------------------
    # Construction heuristics (multi-start)
    # ---------------------------
    def order_desc_deterministic() -> List[int]:
        return sorted(range(n), key=lambda i: (-w[i], i))

    def order_desc_shuffle_blocks() -> List[int]:
        order: List[int] = []
        for wt in unique_wts_desc:
            block = items_of_weight[wt][:]
            random.shuffle(block)
            order.extend(block)
        return order

    def order_desc_noise() -> List[int]:
        return sorted(range(n), key=lambda i: (-w[i], random.random()))

    def order_wave() -> List[int]:
        # Interleave largest and medium to diversify (still non-increasing on average)
        # Make two lists: heavy half and light half by weight rank.
        all_items = sorted(range(n), key=lambda i: (-w[i], i))
        mid = len(all_items) // 2
        A = all_items[:mid]
        B = all_items[mid:]
        # shuffle within small windows in B
        for k in range(0, len(B), 8):
            chunk = B[k:k+8]
            random.shuffle(chunk)
            B[k:k+8] = chunk
        out = []
        i = j = 0
        while i < len(A) or j < len(B):
            if i < len(A):
                out.append(A[i]); i += 1
            if j < len(B):
                out.append(B[j]); j += 1
        return out

    def construct(order: List[int], mode: str) -> Tuple[List[List[int]], List[int]]:
        bins: List[List[int]] = []
        loads: List[int] = []

        # For BFD variants, keep bins sorted by load is expensive; instead scan with best fit key.
        for it in order:
            wi = w[it]
            best_j = -1
            best_key = None

            if mode == "FFD":
                for j, L in enumerate(loads):
                    if L + wi <= C:
                        best_j = j
                        break
            else:
                # Best-fit decreasing with stronger tie-breaks:
                # primary: min remainder
                # tie: prefer remainder that is a common weight (complement opportunity)
                # tie: prefer fuller bins (same as smaller remainder)
                for j, L in enumerate(loads):
                    rem = C - (L + wi)
                    if rem < 0:
                        continue
                    key = (rem, -freq_weight.get(rem, 0), -L)
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

    # ---------------------------
    # Stateful solution for fast local moves
    # ---------------------------
    class SolutionState:
        __slots__ = (
            "bins", "loads", "pos_bin", "pos_idx", "alive", "B",
            "slack2_sum", "slack_sum", "max_slack",
            "bins_by_slack", "empties", "moves_since_rebuild"
        )

        def __init__(self, bins: List[List[int]], loads: List[int]):
            self.bins = [b[:] for b in bins]
            self.loads = loads[:]
            self.pos_bin = [-1] * n
            self.pos_idx = [-1] * n
            self.alive = [True] * len(self.bins)
            self.B = len(self.bins)
            self.slack2_sum = 0
            self.slack_sum = 0
            self.max_slack = 0
            self.bins_by_slack = [[] for _ in range(C + 1)]
            self.empties = 0
            self.moves_since_rebuild = 0
            self.rebuild()

        def key(self) -> Tuple[int, Tuple[int, int, int]]:
            return (self.B, (self.slack2_sum, self.slack_sum, self.max_slack))

        def rebuild(self) -> None:
            B = len(self.bins)
            self.alive = [True] * B
            self.B = B
            self.pos_bin = [-1] * n
            self.pos_idx = [-1] * n
            self.bins_by_slack = [[] for _ in range(C + 1)]
            ssum = 0
            s2sum = 0
            msl = 0
            for b in range(B):
                L = self.loads[b]
                s = C - L
                if s < 0:
                    s = 0
                self.bins_by_slack[s].append(b)
                ssum += s
                s2sum += s * s
                if s > msl:
                    msl = s
                for j, it in enumerate(self.bins[b]):
                    self.pos_bin[it] = b
                    self.pos_idx[it] = j
            self.slack_sum = ssum
            self.slack2_sum = s2sum
            self.max_slack = msl
            self.empties = 0
            self.moves_since_rebuild = 0

        def compact(self) -> None:
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
            self.rebuild()

        def maybe_rebuild(self) -> None:
            if self.empties > 0.18 * max(1, len(self.bins)) or self.moves_since_rebuild > 900:
                self.compact()

        def ensure_bin(self) -> int:
            b = len(self.bins)
            self.bins.append([])
            self.loads.append(0)
            self.alive.append(True)
            self.B += 1
            self.bins_by_slack[C].append(b)
            self.slack_sum += C
            self.slack2_sum += C * C
            if C > self.max_slack:
                self.max_slack = C
            return b

        def _update_bin_load(self, b: int, oldL: int, newL: int) -> None:
            oldS = C - oldL
            newS = C - newL
            self.slack_sum += (newS - oldS)
            self.slack2_sum += (newS * newS - oldS * oldS)
            # max_slack maintained lazily; recompute on rebuild
            if newS > self.max_slack:
                self.max_slack = newS
            if 0 <= newS <= C:
                self.bins_by_slack[newS].append(b)

        def candidate_bins(self, wi: int, exclude: int = -1, window: int = 14, limit: int = 36) -> List[int]:
            res: List[int] = []
            seen = set()
            hi = min(C, wi + window)
            for s in range(wi, hi + 1):
                bucket = self.bins_by_slack[s]
                for b in reversed(bucket[-80:]):
                    if b == exclude or b in seen:
                        continue
                    if b >= len(self.bins) or not self.alive[b]:
                        continue
                    if C - self.loads[b] != s:
                        continue
                    seen.add(b)
                    res.append(b)
                    if len(res) >= limit:
                        return res
            # fallback random
            if self.B > 0 and len(res) < min(limit, 10):
                trials = 0
                while trials < 80 and len(res) < min(limit, 14):
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

        def relocate(self, it: int, to_b: int) -> Optional[Tuple]:
            b_from = self.pos_bin[it]
            if b_from < 0 or to_b == b_from:
                return None
            wi = w[it]

            created = False
            if to_b == len(self.bins):
                to_b = self.ensure_bin()
                created = True

            if not self.alive[to_b] or self.loads[to_b] + wi > C:
                if created:
                    # kill the empty created bin
                    self.alive[to_b] = False
                    self.B -= 1
                    self.empties += 1
                return None

            oldL_from = self.loads[b_from]
            oldL_to = self.loads[to_b]

            idx = self.pos_idx[it]
            last = self.bins[b_from][-1]
            self.bins[b_from][idx] = last
            self.pos_idx[last] = idx
            self.bins[b_from].pop()

            self.loads[b_from] = oldL_from - wi
            self._update_bin_load(b_from, oldL_from, self.loads[b_from])

            self.pos_bin[it] = to_b
            self.pos_idx[it] = len(self.bins[to_b])
            self.bins[to_b].append(it)
            self.loads[to_b] = oldL_to + wi
            self._update_bin_load(to_b, oldL_to, self.loads[to_b])

            from_empty = (len(self.bins[b_from]) == 0)
            if from_empty and self.alive[b_from]:
                self.alive[b_from] = False
                self.B -= 1
                self.empties += 1

            self.moves_since_rebuild += 1
            return ("rel", it, b_from, to_b, idx, last, created, from_empty, oldL_from, oldL_to)

        def swap(self, a: int, b_it: int) -> Optional[Tuple]:
            ba = self.pos_bin[a]
            bb = self.pos_bin[b_it]
            if ba < 0 or bb < 0 or ba == bb:
                return None
            if not self.alive[ba] or not self.alive[bb]:
                return None
            wa = w[a]
            wb = w[b_it]
            oldLa = self.loads[ba]
            oldLb = self.loads[bb]
            newLa = oldLa - wa + wb
            newLb = oldLb - wb + wa
            if newLa > C or newLb > C:
                return None
            ia = self.pos_idx[a]
            ib = self.pos_idx[b_it]

            self.bins[ba][ia], self.bins[bb][ib] = self.bins[bb][ib], self.bins[ba][ia]
            self.pos_bin[a], self.pos_bin[b_it] = bb, ba
            self.pos_idx[a], self.pos_idx[b_it] = ib, ia

            self.loads[ba] = newLa
            self.loads[bb] = newLb
            self._update_bin_load(ba, oldLa, newLa)
            self._update_bin_load(bb, oldLb, newLb)

            self.moves_since_rebuild += 1
            return ("swp", a, b_it, ba, bb, ia, ib, oldLa, oldLb)

        def undo(self, rec: Tuple) -> None:
            tag = rec[0]
            if tag == "rel":
                _, it, b_from, b_to, idx, _last, created, from_empty, oldL_from, oldL_to = rec
                wi = w[it]

                if from_empty:
                    self.alive[b_from] = True
                    self.B += 1
                    self.empties -= 1

                # remove it from b_to
                it_pos = self.pos_idx[it]
                last2 = self.bins[b_to][-1]
                self.bins[b_to][it_pos] = last2
                self.pos_idx[last2] = it_pos
                self.bins[b_to].pop()

                cur_to = self.loads[b_to]
                self.loads[b_to] = oldL_to
                self._update_bin_load(b_to, cur_to, oldL_to)

                # add it back to b_from at idx
                self.pos_bin[it] = b_from
                self.pos_idx[it] = len(self.bins[b_from])
                self.bins[b_from].append(it)
                cur_from = self.loads[b_from]
                self.loads[b_from] = oldL_from
                self._update_bin_load(b_from, cur_from, oldL_from)

                # swap into position idx
                if idx != self.pos_idx[it]:
                    other = self.bins[b_from][idx]
                    self.bins[b_from][idx] = it
                    self.bins[b_from][-1] = other
                    self.pos_idx[it] = idx
                    self.pos_idx[other] = len(self.bins[b_from]) - 1

                if created:
                    # created bin should now be empty; mark dead
                    if len(self.bins[b_to]) == 0 and self.alive[b_to]:
                        self.alive[b_to] = False
                        self.B -= 1
                        self.empties += 1

            elif tag == "swp":
                _, a, b_it, ba, bb, ia, ib, oldLa, oldLb = rec
                # swap back
                self.bins[ba][ia], self.bins[bb][ib] = self.bins[bb][ib], self.bins[ba][ia]
                self.pos_bin[a], self.pos_bin[b_it] = ba, bb
                self.pos_idx[a], self.pos_idx[b_it] = ia, ib

                curLa = self.loads[ba]
                curLb = self.loads[bb]
                self.loads[ba] = oldLa
                self.loads[bb] = oldLb
                self._update_bin_load(ba, curLa, oldLa)
                self._update_bin_load(bb, curLb, oldLb)

        def alive_bins_sorted_by_load(self, limit: Optional[int] = None) -> List[int]:
            bins = [b for b in range(len(self.bins)) if self.alive[b] and self.bins[b]]
            bins.sort(key=lambda b: self.loads[b])
            if limit is not None and len(bins) > limit:
                return bins[:limit]
            return bins

        def recompute_max_slack(self) -> None:
            self.max_slack = 0
            for b in range(len(self.bins)):
                if self.alive[b] and self.bins[b]:
                    s = C - self.loads[b]
                    if s > self.max_slack:
                        self.max_slack = s

    # ---------------------------
    # VND neighborhoods
    # ---------------------------
    def n1_empty_light_bin(st: SolutionState, bin_limit: int = 38, cand_per_item: int = 28) -> bool:
        # Try to empty one of the lightest bins via greedy best-fit, with shallow backtracking.
        src_bins = st.alive_bins_sorted_by_load(limit=bin_limit)
        for b in src_bins:
            if now() >= deadline:
                return False
            if not st.alive[b] or not st.bins[b]:
                continue
            items = st.bins[b][:]
            items.sort(key=lambda it: -w[it])
            move_stack: List[Tuple] = []

            def attempt(i: int, back: int) -> bool:
                if i >= len(items):
                    return True
                it = items[i]
                if st.pos_bin[it] != b:
                    return attempt(i + 1, back)
                wi = w[it]
                cands = st.candidate_bins(wi, exclude=b, window=12, limit=cand_per_item)
                # best-fit
                cands.sort(key=lambda tb: (C - (st.loads[tb] + wi), st.loads[tb]))
                for tb in cands:
                    rec = st.relocate(it, tb)
                    if rec is None:
                        continue
                    move_stack.append(rec)
                    if attempt(i + 1, back):
                        return True
                    st.undo(move_stack.pop())

                if back > 0 and move_stack:
                    last = move_stack.pop()
                    st.undo(last)
                    return attempt(i, back - 1)
                return False

            if attempt(0, back=2):
                st.maybe_rebuild()
                return True
            while move_stack:
                st.undo(move_stack.pop())
        return False

    def n2_ejection_chain(st: SolutionState, trials_bins: int = 26, tries_per_item: int = 10) -> bool:
        # Bounded ejection chain length 2: x -> t (needs eject y), then y -> u (direct or needs eject z)
        src_bins = st.alive_bins_sorted_by_load(limit=trials_bins)
        for b in src_bins:
            if now() >= deadline:
                return False
            if not st.alive[b] or not st.bins[b]:
                continue
            items = st.bins[b][:]
            items.sort(key=lambda it: -w[it])
            for x in items:
                if now() >= deadline:
                    return False
                if st.pos_bin[x] != b:
                    continue
                wx = w[x]

                targets = st.candidate_bins(wx, exclude=b, window=22, limit=32)
                random.shuffle(targets)
                for t in targets[:tries_per_item]:
                    if not st.alive[t] or t == b:
                        continue
                    slack_t = C - st.loads[t]
                    if slack_t >= wx:
                        # direct relocate is handled by N1 mostly; still allow
                        rec0 = st.relocate(x, t)
                        if rec0 is not None:
                            st.maybe_rebuild()
                            return True
                        continue

                    need = wx - slack_t
                    # choose eject y in t: smallest y with wy>=need (min disruption)
                    y = None
                    best_wy = 10**9
                    for cand in st.bins[t]:
                        wy = w[cand]
                        if wy >= need and wy < best_wy:
                            y = cand
                            best_wy = wy
                    if y is None:
                        continue

                    rec_s = st.swap(x, y)
                    if rec_s is None:
                        continue

                    # now y is in bin b; try place y elsewhere; if not, allow 2nd ejection
                    wy = w[y]
                    cands2 = st.candidate_bins(wy, exclude=b, window=22, limit=36)
                    # consider also very slack bins
                    if random.random() < 0.25:
                        cands2.append(len(st.bins))

                    placed = False
                    for u in cands2:
                        if u == b:
                            continue
                        if u == len(st.bins):
                            rec1 = st.relocate(y, u)
                            if rec1 is not None:
                                placed = True
                                break
                            continue

                        slack_u = C - st.loads[u]
                        if slack_u >= wy:
                            rec1 = st.relocate(y, u)
                            if rec1 is not None:
                                placed = True
                                break
                        else:
                            # second ejection: swap y with z in u, then try to place z
                            need2 = wy - slack_u
                            z = None
                            best_wz = 10**9
                            for candz in st.bins[u]:
                                wz = w[candz]
                                if wz >= need2 and wz < best_wz:
                                    z = candz
                                    best_wz = wz
                            if z is None:
                                continue
                            rec2 = st.swap(y, z)
                            if rec2 is None:
                                continue
                            # now z is in bin b. Try relocate z (or new bin)
                            wz = w[z]
                            cands3 = st.candidate_bins(wz, exclude=b, window=20, limit=30)
                            if random.random() < 0.35:
                                cands3.append(len(st.bins))
                            okz = False
                            for v in cands3:
                                if v == b:
                                    continue
                                if st.relocate(z, v) is not None:
                                    okz = True
                                    break
                            if okz:
                                placed = True
                                break
                            st.undo(rec2)

                    if placed:
                        st.maybe_rebuild()
                        return True

                    st.undo(rec_s)
        return False

    def n3_guided_swaps(st: SolutionState, attempts: int = 420) -> bool:
        alive_bins = [b for b in range(len(st.bins)) if st.alive[b] and st.bins[b]]
        if len(alive_bins) < 2:
            return False
        alive_bins.sort(key=lambda b: C - st.loads[b], reverse=True)
        slack_bins = alive_bins[: min(16, len(alive_bins))]
        light_bins = st.alive_bins_sorted_by_load(limit=min(40, len(alive_bins)))

        for _ in range(attempts):
            if now() >= deadline:
                return False
            b1 = random.choice(slack_bins)
            b2 = random.choice(light_bins)
            if b1 == b2 or not st.bins[b1] or not st.bins[b2]:
                continue

            # pick a heavy from b2, and a candidate from b1 that improves both remainders
            a = max(st.bins[b2], key=lambda it: w[it])
            wa = w[a]
            slack2 = C - st.loads[b2]
            # try a few candidates from b1 close to slack2+wa
            cand_list = st.bins[b1]
            if not cand_list:
                continue
            # sample small subset
            sample = cand_list if len(cand_list) <= 8 else random.sample(cand_list, 8)
            best = None
            for b_it in sample:
                wb = w[b_it]
                # feasibility checked in swap
                # score: reduce squared slack in both bins
                oldLa, oldLb = st.loads[b2], st.loads[b1]
                newLa = oldLa - wa + wb
                newLb = oldLb - wb + wa
                if newLa > C or newLb > C:
                    continue
                old = (C - oldLa) * (C - oldLa) + (C - oldLb) * (C - oldLb)
                new = (C - newLa) * (C - newLa) + (C - newLb) * (C - newLb)
                score = old - new
                if best is None or score > best[0]:
                    best = (score, b_it)
            if best is None:
                continue
            if best[0] <= 0 and random.random() > 0.12:
                continue
            rec = st.swap(a, best[1])
            if rec is not None:
                st.maybe_rebuild()
                return True
        return False

    def vnd(st: SolutionState) -> None:
        # Intensify: repeat until no change
        while now() < deadline:
            before = st.key()
            if n1_empty_light_bin(st):
                continue
            if n2_ejection_chain(st):
                continue
            if n3_guided_swaps(st):
                continue
            st.maybe_rebuild()
            # no move
            if st.key() == before:
                break

    # ---------------------------
    # Destroy/repair for shaking (standard in VNS for BPP)
    # ---------------------------
    def remove_items(st: SolutionState, items: List[int]) -> List[int]:
        removed: List[int] = []
        for it in items:
            b = st.pos_bin[it]
            if b < 0 or not st.alive[b]:
                continue
            wi = w[it]
            oldL = st.loads[b]
            idx = st.pos_idx[it]
            last = st.bins[b][-1]
            st.bins[b][idx] = last
            st.pos_idx[last] = idx
            st.bins[b].pop()
            st.loads[b] = oldL - wi
            st._update_bin_load(b, oldL, st.loads[b])
            st.pos_bin[it] = -1
            st.pos_idx[it] = -1
            removed.append(it)
            if len(st.bins[b]) == 0 and st.alive[b]:
                st.alive[b] = False
                st.B -= 1
                st.empties += 1
        st.moves_since_rebuild += len(removed)
        return removed

    def repair_regret2(st: SolutionState, removed: List[int]) -> None:
        unplaced = removed[:]
        unplaced.sort(key=lambda it: -w[it])

        # small helper: evaluate best insertion options
        def best_two_bins(it: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
            wi = w[it]
            cands = st.candidate_bins(wi, exclude=-1, window=20, limit=44)
            fits: List[Tuple[int, int]] = []
            for b in cands:
                if st.loads[b] + wi <= C:
                    rem = C - (st.loads[b] + wi)
                    fits.append((rem, b))
            fits.sort()
            if not fits:
                return ((C - wi, len(st.bins)), (10**9, -1))
            best = fits[0]
            second = fits[1] if len(fits) >= 2 else (C - wi, len(st.bins))
            return best, second

        while unplaced and now() < deadline:
            # evaluate a subset for speed
            subset = unplaced if len(unplaced) <= 90 else unplaced[:90]
            best_choice = None  # (regret, weight, it, best_rem, best_bin)
            for it in subset:
                wi = w[it]
                (r1, b1), (r2, _b2) = best_two_bins(it)
                regret = r2 - r1
                choice = (regret, wi, it, r1, b1)
                if best_choice is None or choice > best_choice:
                    best_choice = choice

            it = best_choice[2] if best_choice is not None else unplaced[0]
            wi = w[it]
            if best_choice is None:
                b = len(st.bins)
            else:
                b = best_choice[4]

            unplaced.remove(it)
            if b == len(st.bins):
                b = st.ensure_bin()

            # insert
            oldL = st.loads[b]
            st.pos_bin[it] = b
            st.pos_idx[it] = len(st.bins[b])
            st.bins[b].append(it)
            st.loads[b] = oldL + wi
            st._update_bin_load(b, oldL, st.loads[b])

        st.maybe_rebuild()

    def shake(st: SolutionState, k: int) -> None:
        st.maybe_rebuild()
        if st.empties:
            st.compact()

        alive_bins = [b for b in range(len(st.bins)) if st.alive[b] and st.bins[b]]
        if not alive_bins:
            return

        # choose bins to destroy
        alive_bins_by_load = sorted(alive_bins, key=lambda b: st.loads[b])
        alive_bins_by_slack = sorted(alive_bins, key=lambda b: C - st.loads[b], reverse=True)

        m = min(len(alive_bins), 2 + k // 2)
        chosen = []
        # light bins are targets to eliminate
        chosen.extend(alive_bins_by_load[: min(m, len(alive_bins_by_load))])
        # add one slack-heavy bin sometimes to reshuffle
        if alive_bins_by_slack and random.random() < 0.35:
            chosen.append(alive_bins_by_slack[0])
        # add random bins for diversification
        if len(alive_bins) > 3 and random.random() < 0.55:
            chosen.append(random.choice(alive_bins))

        # unique
        seen = set()
        chosen2 = []
        for b in chosen:
            if b not in seen:
                seen.add(b)
                chosen2.append(b)
        chosen = chosen2

        removed_items: List[int] = []
        for b in chosen:
            removed_items.extend(st.bins[b][:])

        # add extra "difficult" items: heavy or complement-rare
        extra = 2 + 2 * k
        for _ in range(extra):
            it = random.randrange(n)
            if st.pos_bin[it] >= 0:
                removed_items.append(it)

        # unique items
        removed_items = list(dict.fromkeys(removed_items))
        removed_items = [it for it in removed_items if st.pos_bin[it] >= 0]

        remove_items(st, removed_items)
        st.maybe_rebuild()
        repair_regret2(st, removed_items)
        st.maybe_rebuild()

    # ---------------------------
    # Initial multi-start
    # ---------------------------
    init_best_bins: List[List[int]] = []
    init_best_loads: List[int] = []
    init_best_key = (10**9, (10**18, 10**18, 10**9))

    init_deadline = start + 0.06 * (deadline - start)  # 6% of budget
    starts = 0
    while starts < 28 and now() < init_deadline:
        starts += 1
        if starts == 1:
            order = order_desc_deterministic(); mode = "BFD"
        elif starts == 2:
            order = order_desc_deterministic(); mode = "FFD"
        elif starts % 5 == 0:
            order = order_wave(); mode = "BFD"
        elif starts % 3 == 0:
            order = order_desc_shuffle_blocks(); mode = "BFD"
        else:
            order = order_desc_noise(); mode = "BFD" if random.random() < 0.75 else "FFD"

        bins, loads = construct(order, mode)
        key = eval_key(bins, loads)
        if key < init_best_key:
            init_best_key = key
            init_best_bins, init_best_loads = bins, loads

    curr = SolutionState(init_best_bins, init_best_loads)
    vnd(curr)
    curr.compact()
    curr.recompute_max_slack()

    best_bins = [b[:] for b in curr.bins if b]
    best_loads = [L for i, L in enumerate(curr.loads) if i < len(curr.bins) and curr.bins[i]]
    best_key = eval_key(best_bins, best_loads)

    # ---------------------------
    # Main VNS loop
    # ---------------------------
    iter_budget = min(250000, 6000 + 120 * n)
    check_every = 160
    ops = 0

    # Sideways acceptance among equal-bin solutions (very limited)
    def accept(curr_key, cand_key) -> bool:
        (cB, cSec) = curr_key
        (nB, nSec) = cand_key
        if nB < cB:
            return True
        if nB > cB:
            return False
        # equal bins: accept improvements
        if nSec < cSec:
            return True
        # rare sideways
        if random.random() < 0.03:
            # allow small slack2 worsening
            delta = max(50, (C * C) // 40)
            return nSec[0] <= cSec[0] + delta
        return False

    curr_key = curr.key()
    stagn = 0

    for it_outer in range(iter_budget):
        ops += 1
        if ops % check_every == 0 and now() >= deadline:
            break

        # adapt kmax by stagnation
        if stagn < 60:
            kmax = 7
        elif stagn < 140:
            kmax = 10
        else:
            kmax = 14

        k = 1
        while k <= kmax:
            ops += 1
            if ops % check_every == 0 and now() >= deadline:
                break

            # build candidate from current compact snapshot
            curr.compact() if curr.empties else None
            cbins = [b[:] for b in curr.bins if b and curr.alive[curr.pos_bin[b[0]]]] if False else None  # no-op placeholder
            # cheaper: clone via compacted representation
            curr.compact()
            base_bins = [b[:] for b in curr.bins]
            base_loads = curr.loads[:]
            cand = SolutionState(base_bins, base_loads)

            shake(cand, k)
            vnd(cand)
            cand.compact()
            cand.recompute_max_slack()

            cand_key = cand.key()

            if accept(curr_key, cand_key):
                curr = cand
                curr_key = cand_key
                k = 1
            else:
                k += 1

            # update best
            if cand_key < best_key:
                best_key = cand_key
                curr.compact()
                best_bins = [b[:] for b in cand.bins if b]
                best_loads = [cand.loads[i] for i in range(len(cand.bins)) if cand.bins[i]]
                stagn = 0
            else:
                stagn += 1

        # periodic restart from best
        if ops % check_every == 0 and now() >= deadline:
            break
        if it_outer % 55 == 0 or (stagn > 120 and it_outer % 18 == 0):
            # restart from best and intensify
            curr = SolutionState([b[:] for b in best_bins], best_loads[:])
            shake(curr, random.choice([3, 4, 6, 7]))
            vnd(curr)
            curr.compact()
            curr.recompute_max_slack()
            curr_key = curr.key()

    return {"packing": best_bins, "bin_weights": best_loads}
