# openai
# greedy_randomised_adaptive_search_procedure_2_performance_3.py

import time
import random
from typing import List, Dict, Tuple, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.time()
    # Allow solver to run beyond nominal limits if caller provides >40; hard stop at 100s as stated.
    hard_cap = 100.0
    time_limit = min(time_limit, hard_cap)

    C = bin_capacity
    n = len(weights)
    if n == 0:
        return {"packing": [], "bin_weights": []}

    # ---------- scoring ----------
    def score_from_residuals(residuals: List[int]) -> Tuple[int, int, int]:
        # Primary: #bins. Secondary: sum(res^2) to encourage tight packing. Tertiary: max residual.
        if not residuals:
            return (0, 0, 0)
        s2 = 0
        mx = 0
        for r in residuals:
            s2 += r * r
            if r > mx:
                mx = r
        return (len(residuals), s2, mx)

    # Lower bound (simple but useful): max(total/C, maxitem/C)
    total_w = sum(weights)
    max_w = max(weights)
    lb = max((total_w + C - 1) // C, (max_w + C - 1) // C)

    # ---------- item order diversification ----------
    # Pre-bucket by weight to shuffle within equal weights.
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
                tmp = bucket[:]
                random.shuffle(tmp)
                seq.extend(tmp)
        return seq

    def iter_items_perturbed() -> List[int]:
        # Occasionally do a mild perturbation: take blocks and shuffle blocks.
        seq = iter_items_bucket_shuffled()
        if n <= 30:
            return seq
        if random.random() < 0.55:
            b = 10 if n < 200 else 20
            blocks = [seq[i:i + b] for i in range(0, n, b)]
            random.shuffle(blocks)
            out: List[int] = []
            for bl in blocks:
                # small in-block shuffle
                if len(bl) > 2 and random.random() < 0.25:
                    random.shuffle(bl)
                out.extend(bl)
            return out
        return seq

    # ---------- baseline: deterministic BFD ----------
    def bfd(items_seq: List[int]) -> Tuple[List[List[int]], List[int]]:
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
            best_ra = None
            for b, r in enumerate(rem):
                if r >= w:
                    ra = r - w
                    if best_ra is None or ra < best_ra:
                        best_ra = ra
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

    # ---------- GRASP construction ----------
    # Penalize awkward gaps; reward exact/near-exact fills.
    def gap_penalty(res_after: int) -> int:
        if res_after <= 0:
            return -6  # perfect
        if res_after <= 2:
            return -3
        # avoid mid gaps
        if C > 0:
            if (C // 3) < res_after <= (C // 2):
                return 4
            if (C // 5) < res_after <= (C // 3):
                return 2
        return 0

    def construct(alpha: float) -> Tuple[List[List[int]], List[int], List[int]]:
        # alpha controls RCL aggressiveness; smaller is greedier.
        items_seq = iter_items_perturbed()
        packing: List[List[int]] = []
        bw: List[int] = []
        rem: List[int] = []

        # Diversification knob: sometimes allow opening a new bin even when feasible.
        open_new_bias = 0.02 + 0.10 * alpha

        for it in items_seq:
            w = weights[it]
            if w > C:
                packing.append([it])
                bw.append(w)
                rem.append(0)
                continue

            feas: List[Tuple[int, int]] = []  # (score, bin)
            for b, r in enumerate(rem):
                if r >= w:
                    ra = r - w
                    # score: primary residual after, plus penalty shaping.
                    sc = ra + 3 * max(0, gap_penalty(ra))
                    # strong reward for exact/near-exact
                    if ra == 0:
                        sc -= 100
                    elif ra <= 2:
                        sc -= 10
                    feas.append((sc, b))

            if not feas or (random.random() < open_new_bias and len(packing) > 0):
                packing.append([it])
                bw.append(w)
                rem.append(C - w)
                continue

            feas.sort(key=lambda x: x[0])
            # RCL by percentile: take first k where k grows with alpha.
            m = len(feas)
            k = 1 + int(alpha * (m - 1))
            if k < 1:
                k = 1
            if k > m:
                k = m
            chosen_b = random.choice(feas[:k])[1]
            packing[chosen_b].append(it)
            bw[chosen_b] += w
            rem[chosen_b] -= w

        return packing, bw, rem

    # ---------- Local search state ----------
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

        def snapshot(self):
            return (
                [lst[:] for lst in self.packing],
                self.bw[:],
                self.res[:],
                self.item_to_bin[:],
                self.pos_in_bin[:],
            )

        def restore(self, snap) -> None:
            self.packing, self.bw, self.res, self.item_to_bin, self.pos_in_bin = snap

    # ---------- Local search primitives ----------
    def best_fit_bins_by_res(res_list: List[int], w: int, exclude: int, limit: int) -> List[int]:
        cand: List[Tuple[int, int]] = []
        for b, r in enumerate(res_list):
            if b == exclude:
                continue
            if r >= w:
                cand.append((r - w, b))
        cand.sort(key=lambda x: x[0])
        return [b for _, b in cand[:limit]]

    def try_empty_bin_beam(state: State, b0: int, beam: int = 10) -> bool:
        # Beam assignment: try to place all items of b0 into other bins.
        if b0 >= state.bins() or not state.packing[b0]:
            return False
        items0 = state.packing[b0][:]
        items0.sort(key=lambda it: weights[it], reverse=True)
        if any(weights[it] > C for it in items0):
            return False

        # Each beam element: (tmp_res, planned_moves)
        init_res = state.res[:]
        beams: List[Tuple[List[int], List[Tuple[int, int]]]] = [(init_res, [])]

        for it in items0:
            w = weights[it]
            new_beams: List[Tuple[List[int], List[Tuple[int, int]]]] = []
            for tmp_res, plan in beams:
                # candidate target bins using tmp_res
                targets = best_fit_bins_by_res(tmp_res, w, exclude=b0, limit=6)
                if not targets:
                    continue
                # diversify among top few
                for bt in targets[: min(3, len(targets))]:
                    tr = tmp_res[:]
                    tr[bt] -= w
                    new_plan = plan + [(it, bt)]
                    new_beams.append((tr, new_plan))

            if not new_beams:
                return False
            # keep best beams by heuristic: minimize sum(res^2) on affected bins
            def beam_key(elem):
                tr, _pl = elem
                # approximate: sum squares of smallest 12 residuals (tighter is better)
                s = 0
                # sample a bit for speed
                sample = tr if len(tr) <= 18 else random.sample(tr, 18)
                for r in sample:
                    s += r * r
                return s

            new_beams.sort(key=beam_key)
            beams = new_beams[:beam]

        # commit best plan
        _tmp_res, plan = beams[0]
        for it, bt in plan:
            state.move_item(it, b0, bt)
        if state.bw[b0] == 0:
            state.remove_bin(b0)
            return True
        return False

    def subset_sum_exact_moves(state: State, b0: int, max_k: int = 3, pool_cap: int = 36) -> bool:
        if b0 >= state.bins() or not state.packing[b0]:
            return False

        pool = state.packing[b0][:]
        if len(pool) > pool_cap:
            pool.sort(key=lambda it: weights[it])
            k = pool_cap // 2
            pool = pool[:k] + pool[-(pool_cap - k):]

        # residual targets (focus on large residuals)
        targets = [(state.res[b], b) for b in range(state.bins()) if b != b0 and state.res[b] > 0]
        if not targets:
            return False
        targets.sort(reverse=True)
        targets = targets[: min(30, len(targets))]

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

        for r, b in targets:
            it1 = single.get(r)
            if it1 is not None and state.item_to_bin[it1] == b0 and state.res[b] >= r:
                state.move_item(it1, b0, b)
                if state.bw[b0] == 0:
                    state.remove_bin(b0)
                return True

            if max_k >= 2:
                pr = pair.get(r)
                if pr is not None:
                    a, c = pr
                    if a != c and state.item_to_bin[a] == b0 and state.item_to_bin[c] == b0 and state.res[b] >= r:
                        state.move_item(a, b0, b)
                        state.move_item(c, b0, b)
                        if state.bw[b0] == 0:
                            state.remove_bin(b0)
                        return True

            if max_k >= 3:
                # x + (y+z) = r
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

    def ejection_chain(state: State, b0: int) -> bool:
        if b0 >= state.bins() or not state.packing[b0]:
            return False

        items0 = state.packing[b0][:]
        items0.sort(key=lambda it: weights[it], reverse=True)
        items0 = items0[: min(10, len(items0))]

        m = state.bins()
        tgt_bins = [b for b in range(m) if b != b0 and state.res[b] < C]
        tgt_bins.sort(key=lambda b: state.res[b])
        tgt_bins = tgt_bins[: min(14, len(tgt_bins))]
        if not tgt_bins:
            return False

        for x in items0:
            wx = weights[x]
            for bt in tgt_bins:
                if state.res[bt] >= wx:
                    continue
                need = wx - state.res[bt]

                cand_y = state.packing[bt][:]
                cand_y.sort(key=lambda it: weights[it], reverse=True)
                cand_y = cand_y[: min(10, len(cand_y))]

                for y in cand_y:
                    wy = weights[y]
                    if wy < need:
                        continue
                    if state.res[bt] + wy < wx:
                        continue

                    # Try place y somewhere; if not, try 2-step: y ejects z.
                    snap = state.snapshot()

                    # option 1: relocate y directly
                    targets_y = best_fit_bins_by_res(state.res, wy, exclude=bt, limit=10)
                    targets_y = [b for b in targets_y if b != b0]
                    moved = False
                    for by in targets_y[: min(5, len(targets_y))]:
                        state.move_item(y, bt, by)
                        state.move_item(x, b0, bt)
                        moved = True
                        break

                    if moved:
                        if state.bw[b0] == 0:
                            state.remove_bin(b0)
                            return True
                        # try finish with beam
                        if try_empty_bin_beam(state, b0, beam=8):
                            return True
                        state.restore(snap)
                        continue

                    # option 2: two-step ejection y -> by by ejecting z
                    # pick a few by where y almost fits
                    near_bins = [b for b in range(state.bins()) if b not in (b0, bt)]
                    near_bins.sort(key=lambda b: (max(0, wy - state.res[b]), state.res[b]))
                    near_bins = near_bins[:12]

                    success = False
                    for by in near_bins:
                        if by == b0 or by == bt:
                            continue
                        if state.res[by] >= wy:
                            continue  # direct would have been caught
                        need2 = wy - state.res[by]
                        cand_z = state.packing[by][:]
                        cand_z.sort(key=lambda it: weights[it], reverse=True)
                        cand_z = cand_z[:8]
                        for z in cand_z:
                            wz = weights[z]
                            if wz < need2:
                                continue
                            if state.res[by] + wz < wy:
                                continue
                            # place z elsewhere
                            targets_z = best_fit_bins_by_res(state.res, wz, exclude=by, limit=10)
                            targets_z = [b for b in targets_z if b not in (b0, bt)]
                            if not targets_z:
                                continue
                            bz = targets_z[0]
                            # commit chain: z->bz, y->by, x->bt
                            state.move_item(z, by, bz)
                            state.move_item(y, bt, by)
                            state.move_item(x, b0, bt)
                            success = True
                            break
                        if success:
                            break

                    if success:
                        if state.bw[b0] == 0:
                            state.remove_bin(b0)
                            return True
                        if try_empty_bin_beam(state, b0, beam=8):
                            return True

                    state.restore(snap)

        return False

    def randomized_repack(pool: List[int], max_bins: int, trials: int = 40) -> Optional[List[List[int]]]:
        # Attempt to repack pool into <= max_bins using multiple randomized best-fit orders.
        # Returns packing if succeeds.
        if max_bins < 1:
            return None
        base = sorted(pool, key=lambda it: weights[it], reverse=True)

        def attempt(seq: List[int]) -> Optional[List[List[int]]]:
            bins: List[List[int]] = [[] for _ in range(max_bins)]
            rem = [C] * max_bins
            for it in seq:
                w = weights[it]
                if w > C:
                    return None
                # best-fit
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
                rem[best_b] -= w
            out = [b for b in bins if b]
            return out

        # deterministic first
        rep = attempt(base)
        if rep is not None:
            return rep

        for _ in range(trials):
            seq = base[:]
            # perturb: shuffle some suffix and occasional swaps
            cut = 5 if len(seq) < 40 else 12
            if len(seq) > cut:
                suf = seq[-cut:]
                random.shuffle(suf)
                seq[-cut:] = suf
            for _s in range(3):
                i = random.randrange(len(seq))
                j = random.randrange(len(seq))
                seq[i], seq[j] = seq[j], seq[i]
            rep = attempt(seq)
            if rep is not None:
                return rep
        return None

    def lns_repack(state: State, k_bins: int = 6) -> bool:
        m = state.bins()
        if m <= 2:
            return False
        k_bins = min(k_bins, m)

        # Choose: lightest bin + largest residual bins + a random bin
        idxs = list(range(m))
        idxs.sort(key=lambda b: state.bw[b])
        chosen = [idxs[0]]
        others = [b for b in range(m) if b != chosen[0]]
        others.sort(key=lambda b: state.res[b], reverse=True)
        chosen.extend(others[: max(0, k_bins - 2)])
        if m > 3:
            chosen.append(random.choice([b for b in range(m) if b not in chosen]))
        chosen = list(dict.fromkeys(chosen))
        if len(chosen) < 3:
            return False

        pool: List[int] = []
        for b in chosen:
            pool.extend(state.packing[b])

        target_bins = len(chosen) - 1
        rep = randomized_repack(pool, target_bins, trials=60)
        if rep is None:
            return False

        chosen_sorted = sorted(chosen, reverse=True)
        for b in chosen_sorted:
            state.remove_bin(b)

        for b_items in rep:
            state.packing.append([])
            state.bw.append(0)
            state.res.append(C)
            new_b = state.bins() - 1
            for it in b_items:
                state.add_item_to_bin(it, new_b)
                state.bw[new_b] += weights[it]
            state.res[new_b] = max(0, C - state.bw[new_b])

        return True

    def tighten_swaps(state: State, attempts: int = 120) -> bool:
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

    def local_search(state: State, t_slice: float, heavy: bool) -> State:
        end_t = min(start + time_limit, time.time() + t_slice)
        no_imp = 0
        max_no = 10 if heavy else 6

        while time.time() < end_t and no_imp < max_no:
            m = state.bins()
            if m <= lb:
                break
            improved = False

            # Candidate bins to eliminate: focus on light bins.
            cand = list(range(m))
            cand.sort(key=lambda b: state.bw[b])
            cand = cand[: min(14 if heavy else 9, m)]
            random.shuffle(cand)

            for b0 in cand:
                if b0 >= state.bins():
                    continue
                if not state.packing[b0]:
                    state.remove_bin(b0)
                    improved = True
                    break

                # 1) beam emptying
                if try_empty_bin_beam(state, b0, beam=12 if heavy else 8):
                    improved = True
                    break

                # 2) exact fill moves
                if subset_sum_exact_moves(state, b0, max_k=3 if heavy else 2):
                    if state.bins() < m:
                        improved = True
                        break

                # 3) ejection chains
                if heavy and ejection_chain(state, b0):
                    improved = True
                    break

                # 4) LNS repack
                if heavy and lns_repack(state, k_bins=6):
                    improved = True
                    break

            if improved:
                no_imp = 0
                continue

            # Tighten within same bin count to enable later eliminations.
            if tighten_swaps(state, attempts=150 if heavy else 80):
                no_imp = 0
                continue

            no_imp += 1

        return state

    # ---------- Reactive GRASP alpha selection ----------
    alphas = [0.0, 0.05, 0.12, 0.25, 0.45, 0.7, 1.0]
    a_idx = {a: i for i, a in enumerate(alphas)}
    cnt = [0] * len(alphas)
    mean_bins = [0.0] * len(alphas)
    best_bins_seen = [10**9] * len(alphas)

    def choose_alpha(phase2: bool) -> float:
        seen = sum(cnt)
        if seen < len(alphas):
            return random.choice(alphas)
        best_mb = min(mean_bins)
        # Utility mixes mean and best achieved.
        util = []
        for i, a in enumerate(alphas):
            mb = mean_bins[i]
            bb = best_bins_seen[i]
            # lower is better; convert to higher utility
            u = -(0.75 * (mb - best_mb) + 0.25 * (bb - min(best_bins_seen)))
            util.append(u)
        beta = 2.6 if phase2 else 1.4
        # softmax
        mx = max(util)
        probs = []
        s = 0.0
        for u in util:
            x = beta * (u - mx)
            if x < -25:
                x = -25
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

    # ---------- initial solution ----------
    base_items = iter_items_bucket_shuffled()
    bp, bbw = bfd(base_items)
    best_state = State(bp, bbw)
    # Spend a bit of time improving initial
    best_state = local_search(best_state, t_slice=min(0.35, time_limit * 0.08), heavy=True)
    best_score = score_from_residuals(best_state.res)

    # ---------- main GRASP loop ----------
    # Fixed iteration budget; time is checked periodically.
    if n <= 200:
        max_iter = 22000
    elif n <= 1000:
        max_iter = 16000
    else:
        max_iter = 12000

    phase2_start = int(max_iter * 0.60)

    for it in range(1, max_iter + 1):
        if (it & 31) == 0 and (time.time() - start) >= time_limit:
            break

        phase2 = it >= phase2_start
        a = choose_alpha(phase2)

        p, bw, _rem = construct(a)
        st = State(p, bw)

        # heavier LS when close to best or in phase2
        heavy = phase2 or (st.bins() <= best_state.bins() + 1)
        # dynamic time slice
        if phase2:
            t_slice = 0.05 if not heavy else 0.14
        else:
            t_slice = 0.02 if not heavy else 0.07

        remaining = start + time_limit - time.time()
        if remaining <= 0:
            break
        if t_slice > remaining:
            t_slice = remaining

        st = local_search(st, t_slice=t_slice, heavy=heavy)

        sc = score_from_residuals(st.res)

        # update reactive stats
        i = a_idx[a]
        cnt[i] += 1
        mean_bins[i] += (sc[0] - mean_bins[i]) / cnt[i]
        if sc[0] < best_bins_seen[i]:
            best_bins_seen[i] = sc[0]

        if sc < best_score:
            best_score = sc
            best_state = st
            # If we hit LB, we're essentially optimal under this LB; keep intensifying briefly.
            if best_score[0] == lb and (time.time() - start) > 0.5:
                # still keep searching (LB may be weak), but reduce randomness a bit later.
                pass

    final_packing = [bin_items[:] for bin_items in best_state.packing]
    final_bw = [sum(weights[i] for i in b) for b in final_packing]
    return {"packing": final_packing, "bin_weights": final_bw}
