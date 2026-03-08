# openai
# tabu_search_0_initial.py

import time
import random
from typing import List, Tuple, Dict, Optional


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    C = int(bin_capacity)
    w = weights
    n = len(w)
    start = time.time()

    # ---------- Helpers ----------
    def now() -> float:
        return time.time() - start

    def bfd_initial() -> Tuple[List[List[int]], List[int], List[int]]:
        # Best-Fit Decreasing; returns (bins, loads, assignment[item]=bin)
        order = sorted(range(n), key=lambda i: w[i], reverse=True)
        bins: List[List[int]] = []
        loads: List[int] = []
        assign = [-1] * n
        for i in order:
            wi = w[i]
            best_b = -1
            best_rem = C + 1
            for b, lb in enumerate(loads):
                rem = C - lb
                if wi <= rem:
                    new_rem = rem - wi
                    if new_rem < best_rem:
                        best_rem = new_rem
                        best_b = b
                        if best_rem == 0:
                            break
            if best_b == -1:
                best_b = len(bins)
                bins.append([i])
                loads.append(wi)
            else:
                bins[best_b].append(i)
                loads[best_b] += wi
            assign[i] = best_b
        return bins, loads, assign

    def score(bins: List[List[int]], loads: List[int]) -> Tuple[int, int, int]:
        # Primary: number of bins
        # Secondary: total slack (unused capacity)
        # Tertiary: encourage high utilization (sum of squared slacks)
        m = len(bins)
        slack = sum(C - lb for lb in loads)
        slack2 = sum((C - lb) * (C - lb) for lb in loads)
        return (m, slack, slack2)

    def clone_solution(bins: List[List[int]], loads: List[int], assign: List[int]):
        return [lst[:] for lst in bins], loads[:], assign[:]

    def rebuild_assign(bins: List[List[int]]) -> List[int]:
        a = [-1] * n
        for b, items in enumerate(bins):
            for i in items:
                a[i] = b
        return a

    def remove_empty_bins(bins: List[List[int]], loads: List[int], assign: List[int]) -> None:
        # Remove empty bins and renumber; keep solution consistent
        mapping = [-1] * len(bins)
        new_bins: List[List[int]] = []
        new_loads: List[int] = []
        for b, items in enumerate(bins):
            if items:
                mapping[b] = len(new_bins)
                new_bins.append(items)
                new_loads.append(loads[b])
        if len(new_bins) != len(bins):
            for i in range(n):
                if assign[i] != -1:
                    assign[i] = mapping[assign[i]]
            bins[:] = new_bins
            loads[:] = new_loads

    # Move application
    def apply_relocate(item: int, src: int, dst: int,
                       bins: List[List[int]], loads: List[int], assign: List[int]) -> None:
        # assumes feasible
        bins[src].remove(item)
        loads[src] -= w[item]
        bins[dst].append(item)
        loads[dst] += w[item]
        assign[item] = dst
        # cleanup empty bin if created
        if not bins[src]:
            remove_empty_bins(bins, loads, assign)

    def apply_swap(a: int, b: int, bin_a: int, bin_b: int,
                   bins: List[List[int]], loads: List[int], assign: List[int]) -> None:
        # assumes feasible and bin_a != bin_b
        bins[bin_a].remove(a)
        bins[bin_b].remove(b)
        bins[bin_a].append(b)
        bins[bin_b].append(a)
        loads[bin_a] += w[b] - w[a]
        loads[bin_b] += w[a] - w[b]
        assign[a] = bin_b
        assign[b] = bin_a

    # ---------- Initial solution ----------
    bins, loads, assign = bfd_initial()
    best_bins, best_loads, best_assign = clone_solution(bins, loads, assign)
    best_score = score(best_bins, best_loads)

    # ---------- Tabu Search parameters ----------
    # Iteration budget: scale with n; also bounded.
    max_iters = max(2000, min(50000, 800 * n))

    # Candidate evaluation limits
    cand_reloc = max(30, min(250, 8 * n))
    cand_swap = max(30, min(250, 6 * n))

    # Tabu tenure
    base_tenure = 7
    tenure_span = 10

    # Stagnation and diversification
    stagnation_limit = max(200, 20 * n)
    kick_moves = max(3, min(25, n // 5 + 1))

    # Tabu structures
    # tabu_move[(item, dst_bin)] = iteration_until_allowed
    tabu_move: Dict[Tuple[int, int], int] = {}

    def is_tabu(item: int, dst_bin: int, it: int) -> bool:
        return tabu_move.get((item, dst_bin), -1) > it

    def set_tabu(item: int, dst_bin: int, it: int) -> None:
        tenure = base_tenure + random.randint(0, tenure_span)
        tabu_move[(item, dst_bin)] = it + tenure

    # ---------- Neighborhood sampling ----------
    def pick_bins_for_attention() -> List[int]:
        # prioritize low-filled bins to try to empty them
        m = len(bins)
        if m <= 1:
            return [0] if m == 1 else []
        fills = [(loads[b], b) for b in range(m)]
        fills.sort()  # ascending load
        k = min(m, max(3, m // 3))
        return [b for _, b in fills[:k]]

    def sample_item_from_bin(b: int) -> Optional[int]:
        if not bins[b]:
            return None
        # bias toward larger items for stronger impact
        items = bins[b]
        if len(items) == 1:
            return items[0]
        # pick among top few by weight
        top = sorted(items, key=lambda i: w[i], reverse=True)
        k = min(len(top), 4)
        return random.choice(top[:k])

    def evaluate_relocate(item: int, src: int, dst: int) -> Optional[Tuple[Tuple[int, int, int], int, int]]:
        # returns (new_score, delta_slack, delta_slack2) to allow tie-breaking; None if infeasible
        if src == dst:
            return None
        wi = w[item]
        if loads[dst] + wi > C:
            return None
        # compute score delta cheaply
        m = len(bins)
        # Whether src becomes empty
        src_size = len(bins[src])
        new_m = m - 1 if src_size == 1 else m

        # slack changes: slack = sum(C - load)
        # move decreases slack in dst by wi (less unused), increases slack in src by wi (more unused)
        # if src becomes empty bin removed: slack removes (C - new_load_src= C) since load becomes 0
        # careful: if bin removed, total slack decreases by C (removing its slack C)
        delta_slack = 0
        # src load decreases by wi => slack increases by wi
        delta_slack += wi
        # dst load increases by wi => slack decreases by wi
        delta_slack -= wi
        # net 0 unless bin removed
        if src_size == 1:
            delta_slack -= C

        # slack2 change
        s_src = C - loads[src]
        s_dst = C - loads[dst]
        # before: s_src^2 + s_dst^2 (+ maybe empty bin)
        # after: (s_src+wi)^2 + (s_dst-wi)^2
        delta_slack2 = (s_src + wi) * (s_src + wi) + (s_dst - wi) * (s_dst - wi) - (s_src * s_src + s_dst * s_dst)
        if src_size == 1:
            # remove bin with slack C
            delta_slack2 -= C * C

        # Construct new score tuple
        new_slack = sum(C - lb for lb in loads) + delta_slack
        new_slack2 = sum((C - lb) * (C - lb) for lb in loads) + delta_slack2
        new_score = (new_m, int(new_slack), int(new_slack2))
        return new_score, delta_slack, delta_slack2

    def evaluate_swap(a: int, b: int, bin_a: int, bin_b: int) -> Optional[Tuple[Tuple[int, int, int], int, int]]:
        if bin_a == bin_b:
            return None
        wa, wb = w[a], w[b]
        # feasibility
        if loads[bin_a] - wa + wb > C:
            return None
        if loads[bin_b] - wb + wa > C:
            return None
        m = len(bins)
        # swaps don't change bin count
        new_m = m

        s_a = C - loads[bin_a]
        s_b = C - loads[bin_b]
        # after loads: load_a' = load_a - wa + wb => slack_a' = s_a + wa - wb
        # load_b' = load_b - wb + wa => slack_b' = s_b + wb - wa
        sa2 = s_a + wa - wb
        sb2 = s_b + wb - wa

        delta_slack = (sa2 + sb2) - (s_a + s_b)  # should be 0
        delta_slack2 = (sa2 * sa2 + sb2 * sb2) - (s_a * s_a + s_b * s_b)

        new_slack = sum(C - lb for lb in loads) + delta_slack
        new_slack2 = sum((C - lb) * (C - lb) for lb in loads) + delta_slack2
        new_score = (new_m, int(new_slack), int(new_slack2))
        return new_score, delta_slack, delta_slack2

    # ---------- Search loop ----------
    it = 0
    last_improve = 0

    # Precompute for speed: keep these sums updated approximately by recomputing when needed.
    # Since neighborhood sizes are moderate, recomputation cost is acceptable.

    while it < max_iters:
        it += 1
        if (it & 0x3F) == 0 and now() >= time_limit:
            break

        cur_score = score(bins, loads)

        # Build candidate moves
        attention_bins = pick_bins_for_attention()
        all_bins = list(range(len(bins)))

        best_move = None  # (move_type, data..., new_score)
        best_move_score = None

        # --- Relocate candidates ---
        for _ in range(cand_reloc):
            if not attention_bins:
                break
            src = random.choice(attention_bins)
            item = sample_item_from_bin(src)
            if item is None:
                continue
            # Try destinations biased toward tight bins (high load)
            if len(bins) <= 1:
                continue
            # choose dst distinct
            dst = random.choice(all_bins)
            if dst == src:
                continue
            wi = w[item]
            if loads[dst] + wi > C:
                # try a couple alternatives quickly
                tried = 0
                while tried < 2:
                    dst2 = random.choice(all_bins)
                    if dst2 != src and loads[dst2] + wi <= C:
                        dst = dst2
                        break
                    tried += 1
                else:
                    continue

            evalr = evaluate_relocate(item, src, dst)
            if evalr is None:
                continue
            new_sc, _, _ = evalr

            tabu = is_tabu(item, dst, it)
            # aspiration: accept tabu if improves global best
            if tabu and not (new_sc < best_score):
                continue

            if best_move_score is None or new_sc < best_move_score:
                best_move_score = new_sc
                best_move = ("relocate", item, src, dst, new_sc)

        # --- Swap candidates ---
        for _ in range(cand_swap):
            if len(bins) <= 1:
                break
            b1 = random.choice(attention_bins) if attention_bins else random.choice(all_bins)
            b2 = random.choice(all_bins)
            if b1 == b2 or not bins[b1] or not bins[b2]:
                continue
            a = sample_item_from_bin(b1)
            bitem = sample_item_from_bin(b2)
            if a is None or bitem is None or a == bitem:
                continue

            evals = evaluate_swap(a, bitem, b1, b2)
            if evals is None:
                continue
            new_sc, _, _ = evals

            tabu = is_tabu(a, b2, it) or is_tabu(bitem, b1, it)
            if tabu and not (new_sc < best_score):
                continue

            if best_move_score is None or new_sc < best_move_score:
                best_move_score = new_sc
                best_move = ("swap", a, bitem, b1, b2, new_sc)

        # If no admissible move found, diversify a bit
        if best_move is None:
            # random kick: relocate random items to random feasible bins
            for _ in range(kick_moves):
                if now() >= time_limit:
                    break
                item = random.randrange(n)
                src = assign[item]
                if src == -1 or len(bins) <= 1:
                    continue
                # pick a feasible dst
                candidates = list(range(len(bins)))
                random.shuffle(candidates)
                for dst in candidates:
                    if dst != src and loads[dst] + w[item] <= C:
                        apply_relocate(item, src, dst, bins, loads, assign)
                        set_tabu(item, dst, it)
                        break
            continue

        # Apply best move
        if best_move[0] == "relocate":
            _, item, src, dst, new_sc = best_move
            # re-fetch src in case bins were compacted (should not have changed since selection)
            apply_relocate(item, src, dst, bins, loads, assign)
            set_tabu(item, dst, it)

        else:
            _, a, bitem, b1, b2, new_sc = best_move
            apply_swap(a, bitem, b1, b2, bins, loads, assign)
            set_tabu(a, b2, it)
            set_tabu(bitem, b1, it)

        # Update best
        cur_sc2 = score(bins, loads)
        if cur_sc2 < best_score:
            best_score = cur_sc2
            best_bins, best_loads, best_assign = clone_solution(bins, loads, assign)
            last_improve = it
        else:
            # stagnation diversification
            if it - last_improve > stagnation_limit:
                last_improve = it
                # heavier kick: shuffle a few items from the emptiest bins
                empties = pick_bins_for_attention()
                moved = 0
                for b in empties:
                    # move up to 2 items out
                    for _ in range(2):
                        if moved >= kick_moves:
                            break
                        if not bins[b]:
                            break
                        item = sample_item_from_bin(b)
                        if item is None:
                            break
                        # choose random feasible destination
                        dests = list(range(len(bins)))
                        random.shuffle(dests)
                        for dst in dests:
                            if dst != b and loads[dst] + w[item] <= C:
                                apply_relocate(item, b, dst, bins, loads, assign)
                                set_tabu(item, dst, it)
                                moved += 1
                                break
                    if moved >= kick_moves:
                        break

        if (it & 0xFF) == 0 and now() >= time_limit:
            break

    # Return best found
    # Ensure consistency and compute bin weights
    remove_empty_bins(best_bins, best_loads, best_assign)
    # best_loads already aligned but recompute to be safe
    bw = [sum(w[i] for i in items) for items in best_bins]
    return {"packing": best_bins, "bin_weights": bw}
