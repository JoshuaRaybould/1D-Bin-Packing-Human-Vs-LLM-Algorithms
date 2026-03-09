import time
import random
from typing import List, Dict, Tuple


def solve(bin_capacity: int, weights: list[int], time_limit: float) -> dict:
    start = time.perf_counter()
    n = len(weights)

    if n == 0:
        return {"packing": [], "bin_weights": []}

    # --- Helpers ---
    def elapsed() -> float:
        return time.perf_counter() - start

    def better(sol_a: Tuple[List[List[int]], List[int]], sol_b: Tuple[List[List[int]], List[int]]) -> bool:
        """Return True if sol_a is better than sol_b."""
        pack_a, bw_a = sol_a
        pack_b, bw_b = sol_b
        if sol_b is None:
            return True
        if len(pack_a) != len(pack_b):
            return len(pack_a) < len(pack_b)
        # tie-break: maximize sum of squared fills (more compact), then minimize max slack
        sqa = sum(x * x for x in bw_a)
        sqb = sum(x * x for x in bw_b)
        if sqa != sqb:
            return sqa > sqb
        msa = max(bin_capacity - x for x in bw_a) if bw_a else 0
        msb = max(bin_capacity - x for x in bw_b) if bw_b else 0
        return msa < msb

    def compute_bin_weights(packing: List[List[int]]) -> List[int]:
        return [sum(weights[i] for i in b) for b in packing]

    # --- GRASP construction ---
    def construct(alpha: float) -> Tuple[List[List[int]], List[int]]:
        # Slightly randomized order around non-increasing weights
        items = list(range(n))
        # Add small noise to break ties and diversify
        items.sort(key=lambda i: (-weights[i], random.random()))

        packing: List[List[int]] = []
        bin_w: List[int] = []

        for it in items:
            w = weights[it]

            # generate candidate placements: (score, bin_index or -1 for new bin)
            candidates = []
            # existing bins
            for b, bw in enumerate(bin_w):
                if bw + w <= bin_capacity:
                    slack_after = bin_capacity - (bw + w)
                    # best-fit: smaller slack is better
                    # add tiny randomness for diversification
                    score = slack_after + 1e-6 * random.random()
                    candidates.append((score, b))

            # new bin option: generally worse than any feasible placement,
            # but required if none feasible.
            if not candidates:
                packing.append([it])
                bin_w.append(w)
                continue

            # Build RCL by threshold based on alpha
            candidates.sort(key=lambda x: x[0])
            best = candidates[0][0]
            worst = candidates[-1][0]
            thresh = best + alpha * (worst - best)
            rcl = [c for c in candidates if c[0] <= thresh]

            # Alternatively cap RCL size to keep it meaningful
            cap = 10
            if len(rcl) > cap:
                rcl = rcl[:cap]

            _, chosen_bin = random.choice(rcl)
            packing[chosen_bin].append(it)
            bin_w[chosen_bin] += w

        return packing, bin_w

    # --- Local search: try to remove bins by relocating all items from a bin ---
    def try_empty_one_bin(packing: List[List[int]], bin_w: List[int], bidx: int) -> bool:
        # Attempt to move all items from bin bidx into other bins (one pass), best-fit.
        items = packing[bidx][:]
        random.shuffle(items)

        # precompute slacks
        slacks = [bin_capacity - bw for bw in bin_w]

        # record moves to rollback if fail
        moves = []  # (item, from_bin, to_bin)

        for it in items:
            w = weights[it]
            best_bin = -1
            best_slack_after = None
            # try best-fit among other bins
            for b in range(len(packing)):
                if b == bidx:
                    continue
                if slacks[b] >= w:
                    slack_after = slacks[b] - w
                    if best_slack_after is None or slack_after < best_slack_after:
                        best_slack_after = slack_after
                        best_bin = b
            if best_bin == -1:
                # rollback
                for item, fb, tb in reversed(moves):
                    packing[tb].remove(item)
                    bin_w[tb] -= weights[item]
                    slacks[tb] += weights[item]
                    packing[fb].append(item)
                    bin_w[fb] += weights[item]
                    slacks[fb] -= weights[item]
                return False

            # perform move
            packing[bidx].remove(it)
            bin_w[bidx] -= w
            slacks[bidx] += w

            packing[best_bin].append(it)
            bin_w[best_bin] += w
            slacks[best_bin] -= w
            moves.append((it, bidx, best_bin))

        # bin emptied
        if packing[bidx]:
            return False

        # remove bin
        packing.pop(bidx)
        bin_w.pop(bidx)
        return True

    def local_search(packing: List[List[int]], bin_w: List[int], max_passes: int = 3) -> Tuple[List[List[int]], List[int]]:
        # A few passes attempting to eliminate lightly loaded bins.
        for _ in range(max_passes):
            if elapsed() >= time_limit:
                break

            # order bins by increasing load to attempt emptying small bins first
            order = list(range(len(packing)))
            order.sort(key=lambda b: (bin_w[b], len(packing[b])))

            improved = False
            j = 0
            while j < len(order):
                if elapsed() >= time_limit:
                    break
                b = order[j]
                # b might have shifted if prior removals happened; check bounds
                if b >= len(packing):
                    j += 1
                    continue
                if len(packing) <= 1:
                    break

                if try_empty_one_bin(packing, bin_w, b):
                    improved = True
                    # need to rebuild order because indices changed
                    order = list(range(len(packing)))
                    order.sort(key=lambda bb: (bin_w[bb], len(packing[bb])))
                    j = 0
                    continue
                j += 1

            if not improved:
                break

        return packing, bin_w

    # --- Parameterization ---
    # Fixed iteration cap, but also must check time.
    # Make it scale with n but remain bounded.
    max_iter = min(2000, max(200, 20 * (n.bit_length() + 1)))

    best_sol = None  # type: ignore

    # A few alpha values cycled to balance greediness/diversification
    alphas = [0.05, 0.1, 0.2, 0.35]

    it = 0
    while it < max_iter and elapsed() < time_limit:
        alpha = alphas[it % len(alphas)]
        packing, bin_w = construct(alpha)

        # local search (essential GRASP component)
        packing, bin_w = local_search(packing, bin_w, max_passes=3)

        sol = (packing, bin_w)
        if best_sol is None or better(sol, best_sol):
            best_sol = ( [b[:] for b in packing], bin_w[:] )

        it += 1
        # periodic time check is already in loop condition/local search

    if best_sol is None:
        # Fallback (should not happen)
        packing, bin_w = construct(0.1)
        best_sol = (packing, bin_w)

    best_packing, best_bin_w = best_sol

    # Final sanity: ensure alignment and feasibility
    # (Keep it lightweight; no heavy checks)
    return {"packing": best_packing, "bin_weights": best_bin_w}
