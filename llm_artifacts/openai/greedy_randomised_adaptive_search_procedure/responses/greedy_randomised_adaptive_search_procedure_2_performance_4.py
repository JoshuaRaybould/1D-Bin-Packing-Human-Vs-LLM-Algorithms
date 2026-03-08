import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    hard_cap = 100.0
    time_limit = min(float(time_limit), hard_cap)

    C = int(bin_capacity)
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ----------------- helpers / score -----------------
    def score(residuals: List[int]) -> Tuple[int, int, int]:
        # lexicographic: fewer bins, tighter (sum sq residual), then max residual
        m = len(residuals)
        s2 = 0
        mx = 0
        for r in residuals:
            s2 += r * r
            if r > mx:
                mx = r
        return (m, s2, mx)

    total_w = sum(weights)
    max_w = max(weights)
    lb = max((total_w + C - 1) // C, (max_w + C - 1) // C)

    # Pre-bucket indices by weight for reproducible diversification.
    by_w: Dict[int, List[int]] = {}
    for i, w in enumerate(weights):
        by_w.setdefault(w, []).append(i)
    distinct_desc = sorted(by_w.keys(), reverse=True)

    def order_bucket_shuffled() -> List[int]:
        seq: List[int] = []
        for w in distinct_desc:
            bucket = by_w[w]
            if len(bucket) <= 1:
                seq.extend(bucket)
            else:
                tmp = bucket[:]
                random.shuffle(tmp)
                seq.extend(tmp)
        return seq

    def order_perturbed() -> List[int]:
        seq = order_bucket_shuffled()
        if n < 40:
            return seq
        # block shuffle (keeps large items early, but moves groups)
        if random.random() < 0.70:
            bsz = 10 if n < 250 else 20
            blocks = [seq[i:i + bsz] for i in range(0, n, bsz)]
            random.shuffle(blocks)
            out: List[int] = []
            for bl in blocks:
                if len(bl) > 2 and random.random() < 0.30:
                    random.shuffle(bl)
                out.extend(bl)
            return out
        return seq

    # ----------------- baseline (BFD) -----------------
    def bfd(items: List[int]) -> Tuple[List[List[int]], List[int], List[int]]:
        packing: List[List[int]] = []
        bw: List[int] = []
        res: List[int] = []
        for it in items:
            w = weights[it]
            if w > C:
                packing.append([it])
                bw.append(w)
                res.append(0)
                continue
            best_b = -1
            best_ra = None
            for b, r in enumerate(res):
                if r >= w:
                    ra = r - w
                    if best_ra is None or ra < best_ra:
                        best_ra = ra
                        best_b = b
            if best_b >= 0:
                packing[best_b].append(it)
                bw[best_b] += w
                res[best_b] -= w
            else:
                packing.append([it])
                bw.append(w)
                res.append(C - w)
        return packing, bw, res

    # ----------------- GRASP construction -----------------
    # We build a restricted candidate list (RCL) among feasible bins.
    # Candidate evaluation tries to minimize "future waste" with a shaped penalty.
    def shaped_penalty(ra: int) -> int:
        # Penalize awkward mid-gaps more than tiny gaps.
        if ra == 0:
            return -200
        if ra <= 2:
            return -25
        if ra <= 5:
            return -8
        # mid gaps: (C/3..C/2) are often bad
        if C >= 10:
            if (C // 3) < ra <= (C // 2):
                return 10
            if (C // 5) < ra <= (C // 3):
                return 5
        return 0

    def construct(alpha: float) -> Tuple[List[List[int]], List[int], List[int]]:
        items = order_perturbed()
        packing: List[List[int]] = []
        bw: List[int] = []
        res: List[int] = []

        # Mild diversification: sometimes open a new bin even if feasible.
        # Keep this smaller than before; rely on RCL for diversification.
        open_new_bias = 0.005 + 0.06 * alpha

        for it in items:
            w = weights[it]
            if w > C:
                packing.append([it])
                bw.append(w)
                res.append(0)
                continue

            feas: List[Tuple[int, int]] = []
            for b, r in enumerate(res):
                if r >= w:
                    ra = r - w
                    # base: leftover; add shaped penalty
                    sc = ra + shaped_penalty(ra)
                    # also encourage packing into already tight bins (smaller r)
                    sc += (r // max(1, C // 10))
                    feas.append((sc, b))

            if not feas or (packing and random.random() < open_new_bias):
                packing.append([it])
                bw.append(w)
                res.append(C - w)
                continue

            feas.sort(key=lambda x: x[0])
            m = len(feas)
            # RCL size: 1..m; alpha near 0 is greedy, alpha near 1 is random
            k = 1 + int(alpha * (m - 1))
            if k > m:
                k = m
            chosen = random.choice(feas[:k])[1]
            packing[chosen].append(it)
            bw[chosen] += w
            res[chosen] -= w

        return packing, bw, res

    # ----------------- state -----------------
    class State:
        __slots__ = ("packing", "bw", "res", "item_to_bin", "pos_in_bin")

        def __init__(self, packing: List[List[int]], bw: List[int]):
            self.packing = [b[:] for b in packing]
            self.bw = bw[:]
            self.res = [max(0, C - x) for x in self.bw]
            self.item_to_bin = [-1] * n
            self.pos_in_bin = [-1] * n
            for b, items in enumerate(self.packing):
                for p, it in enumerate(items):
                    self.item_to_bin[it] = b
                    self.pos_in_bin[it] = p

        def bins(self) -> int:
            return len(self.packing)

        def snapshot(self):
            return (
                [b[:] for b in self.packing],
                self.bw[:],
                self.res[:],
                self.item_to_bin[:],
                self.pos_in_bin[:],
            )

        def restore(self, snap) -> None:
            self.packing, self.bw, self.res, self.item_to_bin, self.pos_in_bin = snap

        def remove_item_from_bin(self, it: int, b: int) -> None:
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

        def remove_bin(self, b: int) -> None:
            last = len(self.packing) - 1
            if b != last:
                self.packing[b] = self.packing[last]
                self.bw[b] = self.bw[last]
                self.res[b] = self.res[last]
                for pos, it in enumerate(self.packing[b]):
                    self.item_to_bin[it] = b
                    self.pos_in_bin[it] = pos
            self.packing.pop()
            self.bw.pop()
            self.res.pop()

    # ----------------- local search (GRASP essential) -----------------
    def best_fit_targets(res: List[int], w: int, exclude: int, limit: int) -> List[int]:
        cand: List[Tuple[int, int]] = []
        for b, r in enumerate(res):
            if b == exclude:
                continue
            if r >= w:
                cand.append((r - w, b))
        cand.sort(key=lambda x: x[0])
        return [b for _, b in cand[:limit]]

    def greedy_reinsert_items(state: State, items: List[int], forbid_bin: int) -> bool:
        # Try to reinsert items one by one best-fit; no new bins allowed.
        # Returns True if all inserted.
        # Uses state.res dynamically.
        items = sorted(items, key=lambda it: weights[it], reverse=True)
        for it in items:
            w = weights[it]
            targets = best_fit_targets(state.res, w, exclude=forbid_bin, limit=20)
            if not targets:
                return False
            bt = targets[0]
            state.move_item(it, forbid_bin, bt)
        return True

    def subset_fill_move(state: State, b0: int, max_k: int = 4, pool_cap: int = 46) -> bool:
        # Try to find 1..max_k items from b0 that exactly fill some other bin residual.
        if b0 >= state.bins() or not state.packing[b0]:
            return False

        pool = state.packing[b0][:]
        if len(pool) > pool_cap:
            pool.sort(key=lambda it: weights[it])
            k = pool_cap // 2
            pool = pool[:k] + pool[-(pool_cap - k):]

        targets = [(state.res[b], b) for b in range(state.bins()) if b != b0 and state.res[b] > 0]
        if not targets:
            return False
        targets.sort(reverse=True)
        targets = targets[: min(40, len(targets))]

        # build maps for 1,2,3 sums (fast) and a limited 4-sum via meet-in-the-middle on small pool.
        single: Dict[int, int] = {}
        for it in pool:
            single.setdefault(weights[it], it)

        pair: Dict[int, Tuple[int, int]] = {}
        if max_k >= 2:
            m = len(pool)
            for i in range(m):
                wi = weights[pool[i]]
                for j in range(i + 1, m):
                    s = wi + weights[pool[j]]
                    if s not in pair:
                        pair[s] = (pool[i], pool[j])

        triple: Dict[int, Tuple[int, int, int]] = {}
        if max_k >= 3:
            # create triples by combining single+pair (limited)
            # limit scan to keep runtime bounded
            small = pool[: min(len(pool), 28)]
            for x in small:
                wx = weights[x]
                for s, (a, b) in list(pair.items())[:1500]:
                    if x == a or x == b or a == b:
                        continue
                    tot = wx + s
                    if tot not in triple:
                        triple[tot] = (x, a, b)

        quad: Dict[int, Tuple[int, int, int, int]] = {}
        if max_k >= 4:
            # meet-in-the-middle with limited pairs
            pair_items = list(pair.items())
            pair_items.sort(key=lambda x: x[0])
            pair_items = pair_items[: min(len(pair_items), 1200)]
            # map sum->(i,j)
            for s1, (a, b) in pair_items:
                for s2, (c, d) in pair_items:
                    tot = s1 + s2
                    if tot in quad:
                        continue
                    if len({a, b, c, d}) < 4:
                        continue
                    quad[tot] = (a, b, c, d)

        for r, b in targets:
            it1 = single.get(r)
            if it1 is not None and state.item_to_bin[it1] == b0 and state.res[b] >= r:
                state.move_item(it1, b0, b)
                return True
            pr = pair.get(r)
            if pr is not None:
                a, c = pr
                if a != c and state.item_to_bin[a] == b0 and state.item_to_bin[c] == b0 and state.res[b] >= r:
                    state.move_item(a, b0, b)
                    state.move_item(c, b0, b)
                    return True
            tr = triple.get(r)
            if tr is not None:
                x, y, z = tr
                if len({x, y, z}) == 3 and state.item_to_bin[x] == b0 and state.item_to_bin[y] == b0 and state.item_to_bin[z] == b0 and state.res[b] >= r:
                    state.move_item(x, b0, b)
                    state.move_item(y, b0, b)
                    state.move_item(z, b0, b)
                    return True
            qr = quad.get(r)
            if qr is not None:
                a, c, d, e = qr
                if state.res[b] >= r and all(state.item_to_bin[it] == b0 for it in (a, c, d, e)):
                    state.move_item(a, b0, b)
                    state.move_item(c, b0, b)
                    state.move_item(d, b0, b)
                    state.move_item(e, b0, b)
                    return True

        return False

    def ejection_chain(state: State, b0: int) -> bool:
        if b0 >= state.bins() or not state.packing[b0]:
            return False

        # focus on a few large items of b0
        items0 = sorted(state.packing[b0], key=lambda it: weights[it], reverse=True)[:12]
        m = state.bins()
        tgt_bins = [b for b in range(m) if b != b0 and state.res[b] < C]
        tgt_bins.sort(key=lambda b: state.res[b])
        tgt_bins = tgt_bins[: min(18, len(tgt_bins))]
        if not tgt_bins:
            return False

        for x in items0:
            wx = weights[x]
            for bt in tgt_bins:
                if state.res[bt] >= wx:
                    continue
                need = wx - state.res[bt]
                # choose a few large items from bt as eject candidates
                cand_y = sorted(state.packing[bt], key=lambda it: weights[it], reverse=True)[:10]
                for y in cand_y:
                    wy = weights[y]
                    if wy < need:
                        continue

                    snap = state.snapshot()

                    # relocate y to best-fit elsewhere
                    ty = best_fit_targets(state.res, wy, exclude=bt, limit=16)
                    ty = [b for b in ty if b != b0]
                    moved = False
                    for by in ty[:6]:
                        state.move_item(y, bt, by)
                        if state.res[bt] >= wx:
                            state.move_item(x, b0, bt)
                            moved = True
                            break
                        # undo partial
                        state.restore(snap)

                    if moved:
                        return True

                    # 2-step: y goes to by after ejecting z
                    state.restore(snap)
                    near = [b for b in range(state.bins()) if b not in (b0, bt)]
                    near.sort(key=lambda b: (max(0, wy - state.res[b]), state.res[b]))
                    near = near[:14]
                    for by in near:
                        if state.res[by] >= wy:
                            continue
                        need2 = wy - state.res[by]
                        cand_z = sorted(state.packing[by], key=lambda it: weights[it], reverse=True)[:8]
                        for z in cand_z:
                            wz = weights[z]
                            if wz < need2:
                                continue
                            tz = best_fit_targets(state.res, wz, exclude=by, limit=16)
                            tz = [b for b in tz if b not in (b0, bt)]
                            if not tz:
                                continue
                            bz = tz[0]
                            state.move_item(z, by, bz)
                            state.move_item(y, bt, by)
                            if state.res[bt] >= wx:
                                state.move_item(x, b0, bt)
                                return True
                            state.restore(snap)

        return False

    def try_eliminate_bin(state: State, b0: int, max_tries: int = 2) -> bool:
        if b0 >= state.bins() or not state.packing[b0]:
            return False
        if any(weights[it] > C for it in state.packing[b0]):
            return False

        # First try: direct greedy reinsertion (very effective if bin is light)
        snap = state.snapshot()
        items = state.packing[b0][:]
        if greedy_reinsert_items(state, items, forbid_bin=b0):
            if state.bw[b0] == 0:
                state.remove_bin(b0)
                return True
        state.restore(snap)

        # Second: use subset exact fills to create room, then reinsert.
        for _ in range(max_tries):
            snap = state.snapshot()
            if subset_fill_move(state, b0, max_k=4):
                # after making another bin tighter, try greedy reinsertion again
                if greedy_reinsert_items(state, state.packing[b0][:], forbid_bin=b0):
                    if state.bw[b0] == 0:
                        state.remove_bin(b0)
                        return True
            state.restore(snap)

        # Third: ejection chains
        snap = state.snapshot()
        if ejection_chain(state, b0):
            # after a successful chain, try finishing elimination
            if greedy_reinsert_items(state, state.packing[b0][:], forbid_bin=b0):
                if state.bw[b0] == 0:
                    state.remove_bin(b0)
                    return True
        state.restore(snap)

        return False

    def destroy_repair(state: State, k_bins: int) -> bool:
        # Standard GRASP intensification style: perturb then improve (still local search flavor).
        m = state.bins()
        if m <= 3:
            return False
        k_bins = min(k_bins, m)

        # pick bins: lightest + a few highest residual + randoms
        idx = list(range(m))
        idx.sort(key=lambda b: state.bw[b])
        chosen = [idx[0]]
        rest = [b for b in range(m) if b != chosen[0]]
        rest.sort(key=lambda b: state.res[b], reverse=True)
        chosen.extend(rest[: max(0, k_bins - 2)])
        while len(chosen) < k_bins:
            b = random.randrange(m)
            if b not in chosen:
                chosen.append(b)
        chosen = sorted(set(chosen))

        pool: List[int] = []
        for b in chosen:
            pool.extend(state.packing[b])

        target_bins = len(chosen) - 1
        if target_bins <= 0:
            return False

        # Attempt to repack pool into fewer bins via repeated randomized best-fit.
        base = sorted(pool, key=lambda it: weights[it], reverse=True)

        def attempt(seq: List[int]) -> Optional[List[List[int]]]:
            bins: List[List[int]] = [[] for _ in range(target_bins)]
            rem = [C] * target_bins
            for it in seq:
                w = weights[it]
                if w > C:
                    return None
                best_b = -1
                best_ra = None
                for b in range(target_bins):
                    if rem[b] >= w:
                        ra = rem[b] - w
                        if best_ra is None or ra < best_ra:
                            best_ra = ra
                            best_b = b
                if best_b < 0:
                    return None
                bins[best_b].append(it)
                rem[best_b] -= w
            return [b for b in bins if b]

        rep = attempt(base)
        if rep is None:
            # trials bounded
            for _ in range(70):
                seq = base[:]
                cut = 10 if len(seq) > 35 else 6
                if len(seq) > cut:
                    suf = seq[-cut:]
                    random.shuffle(suf)
                    seq[-cut:] = suf
                for _s in range(4):
                    i = random.randrange(len(seq))
                    j = random.randrange(len(seq))
                    seq[i], seq[j] = seq[j], seq[i]
                rep = attempt(seq)
                if rep is not None:
                    break
        if rep is None:
            return False

        # commit: remove chosen bins, add repacked bins
        snap = state.snapshot()
        for b in sorted(chosen, reverse=True):
            state.remove_bin(b)

        # add rep bins
        for items in rep:
            state.packing.append([])
            state.bw.append(0)
            state.res.append(C)
            nb = state.bins() - 1
            for it in items:
                state.add_item_to_bin(it, nb)
                state.bw[nb] += weights[it]
            state.res[nb] = max(0, C - state.bw[nb])

        # accept only if bins reduced
        if state.bins() < len(snap[0]):
            return True
        state.restore(snap)
        return False

    def local_search(state: State, end_time: float, heavy: bool) -> State:
        # bin-elimination driven LS
        no_imp = 0
        max_no = 18 if heavy else 10

        while time.time() < end_time and no_imp < max_no:
            if state.bins() <= lb:
                break
            m0 = state.bins()
            improved = False

            # Try eliminate light bins first.
            cand = list(range(state.bins()))
            cand.sort(key=lambda b: state.bw[b])
            cand = cand[: min(state.bins(), 18 if heavy else 12)]
            # mix order
            if len(cand) > 2:
                head = cand[:]
                random.shuffle(head)
                cand = head

            for b0 in cand:
                if b0 >= state.bins():
                    continue
                if try_eliminate_bin(state, b0, max_tries=3 if heavy else 2):
                    improved = True
                    break

            if improved:
                no_imp = 0
                continue

            # occasional destroy/repair intensification
            if heavy and random.random() < 0.45:
                if destroy_repair(state, k_bins=7):
                    no_imp = 0
                    continue

            # if nothing improved, stop after some failures
            no_imp += 1

            # tiny safeguard: if state got worse (shouldn't), revert by doing nothing.
            if state.bins() > m0:
                no_imp += 2

        return state

    # ----------------- Path relinking (standard GRASP intensification) -----------------
    def canonical_signature(packing: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
        # Sort items inside bins, then sort bins by tuple.
        bins = [tuple(sorted(b)) for b in packing if b]
        bins.sort()
        return tuple(bins)

    def path_relink(a: State, b: State, time_cap: float) -> Optional[State]:
        # Relink from a towards b by moving items to match bin membership of b.
        # We implement a pragmatic variant: pick items that are in a different bin than in b,
        # attempt to move them to the target bin if feasible, otherwise skip.
        if time.time() >= time_cap:
            return None

        # Build target bin for each item according to b.
        target = b.item_to_bin
        st = State(a.packing, a.bw)

        diff = [it for it in range(n) if st.item_to_bin[it] != target[it] and target[it] != -1]
        # Move larger items first.
        diff.sort(key=lambda it: weights[it], reverse=True)

        # To map target bin indices to actual bins in st: bins differ. We approximate by
        # creating a mapping based on overlap with b bins.
        # Build signatures for b bins.
        b_bins = [set(bin_items) for bin_items in b.packing]
        # Map each current bin in st to best matching b bin.
        map_bin = [-1] * st.bins()
        used = set()
        for i in range(st.bins()):
            sset = set(st.packing[i])
            best_j = -1
            best_ov = -1
            for j, tset in enumerate(b_bins):
                if j in used:
                    continue
                ov = len(sset & tset)
                if ov > best_ov:
                    best_ov = ov
                    best_j = j
            map_bin[i] = best_j
            if best_j >= 0:
                used.add(best_j)

        # Inverse mapping: target b-bin -> current st-bin
        inv: Dict[int, int] = {}
        for i, j in enumerate(map_bin):
            if j is not None and j >= 0 and j not in inv:
                inv[j] = i

        # Relink steps
        for it in diff:
            if time.time() >= time_cap:
                break
            bt = target[it]
            if bt not in inv:
                continue
            b_to = inv[bt]
            b_from = st.item_to_bin[it]
            if b_from == -1 or b_from == b_to:
                continue
            w = weights[it]
            if st.res[b_to] >= w:
                st.move_item(it, b_from, b_to)
                # clean empty bins
                if st.bw[b_from] == 0:
                    st.remove_bin(b_from)
                    # indices shifted; abort mapping correctness cheaply
                    break

        # Final quick improvement: try eliminate a light bin.
        endt = min(time_cap, time.time() + 0.08)
        st = local_search(st, end_time=endt, heavy=False)
        return st

    # ----------------- Reactive alpha selection -----------------
    alphas = [0.0, 0.03, 0.08, 0.15, 0.28, 0.45, 0.65, 0.85, 1.0]
    a_idx = {a: i for i, a in enumerate(alphas)}
    cnt = [0] * len(alphas)
    mean_bins = [0.0] * len(alphas)
    best_bins_seen = [10**9] * len(alphas)

    def choose_alpha(phase2: bool) -> float:
        seen = sum(cnt)
        if seen < len(alphas):
            return random.choice(alphas)
        best_mb = min(mean_bins)
        best_bb = min(best_bins_seen)
        util = []
        for i in range(len(alphas)):
            mb = mean_bins[i]
            bb = best_bins_seen[i]
            u = -(0.70 * (mb - best_mb) + 0.30 * (bb - best_bb))
            util.append(u)
        beta = 3.0 if phase2 else 1.6
        mx = max(util)
        probs = []
        s = 0.0
        for u in util:
            x = beta * (u - mx)
            if x < -30:
                x = -30
            p = pow(2.718281828, x)
            probs.append(p)
            s += p
        r = random.random() * s
        acc = 0.0
        for a, p in zip(alphas, probs):
            acc += p
            if acc >= r:
                return a
        return alphas[-1]

    # ----------------- initial solution + elite pool -----------------
    base = order_bucket_shuffled()
    p0, bw0, _ = bfd(base)
    best_state = State(p0, bw0)

    # use initial time slice for a strong improvement
    init_end = min(start + time_limit, time.time() + min(0.8, 0.06 * time_limit))
    best_state = local_search(best_state, end_time=init_end, heavy=True)
    best_sc = score(best_state.res)

    # elite solutions for path relinking
    elite: List[State] = [best_state]
    elite_sig = {canonical_signature(best_state.packing)}
    ELITE_MAX = 10

    def consider_elite(st: State) -> None:
        nonlocal elite, elite_sig
        sig = canonical_signature(st.packing)
        if sig in elite_sig:
            return
        elite_sig.add(sig)
        elite.append(st)
        elite.sort(key=lambda s: score(s.res))
        if len(elite) > ELITE_MAX:
            rm = elite.pop()
            elite_sig.discard(canonical_signature(rm.packing))

    # ----------------- main GRASP loop -----------------
    # Fixed iteration budget + periodic time checks.
    # Increase iterations; LS slices are time-governed.
    if n <= 200:
        max_iter = 42000
    elif n <= 1000:
        max_iter = 30000
    else:
        max_iter = 22000

    phase2_start = int(max_iter * 0.55)

    for it in range(1, max_iter + 1):
        if (it & 63) == 0 and (time.time() - start) >= time_limit:
            break

        phase2 = it >= phase2_start
        a = choose_alpha(phase2)

        p, bw, _res = construct(a)
        st = State(p, bw)

        # time slice per iteration; increase in phase2 to intensify
        # also adapt to instance size
        if n <= 250:
            base_slice = 0.030
        elif n <= 1500:
            base_slice = 0.022
        else:
            base_slice = 0.016

        if phase2:
            base_slice *= 2.2

        # heavier when promising
        heavy = phase2 or (st.bins() <= best_state.bins() + 1)
        slice_mult = 1.7 if heavy else 1.0
        t_slice = base_slice * slice_mult

        remaining = start + time_limit - time.time()
        if remaining <= 0:
            break
        if t_slice > remaining:
            t_slice = remaining

        st = local_search(st, end_time=time.time() + t_slice, heavy=heavy)
        sc = score(st.res)

        # update reactive stats
        i = a_idx[a]
        cnt[i] += 1
        mean_bins[i] += (sc[0] - mean_bins[i]) / cnt[i]
        if sc[0] < best_bins_seen[i]:
            best_bins_seen[i] = sc[0]

        # path relinking occasionally with elite
        if elite and (phase2 or random.random() < 0.22):
            partner = random.choice(elite)
            # cap relinking time
            pr_cap = min(start + time_limit, time.time() + (0.06 if not phase2 else 0.11))
            pr = path_relink(st, partner, time_cap=pr_cap)
            if pr is not None:
                pr_sc = score(pr.res)
                if pr_sc < sc:
                    st = pr
                    sc = pr_sc

        if sc < best_sc:
            best_sc = sc
            best_state = st
            consider_elite(st)
        else:
            # keep some diverse elites close to best
            if sc[0] <= best_sc[0] + 1:
                consider_elite(st)

        if best_sc[0] <= lb:
            # can't do better than LB, but LB can be weak; keep searching.
            pass

    final_packing = [b[:] for b in best_state.packing]
    final_bw = [sum(weights[i] for i in b) for b in final_packing]
    return {"packing": final_packing, "bin_weights": final_bw}
