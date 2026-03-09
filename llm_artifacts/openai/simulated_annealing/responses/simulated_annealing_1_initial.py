import time
import math
import random
from typing import List, Dict


def solve(bin_capacity: int, weights: List[int], time_limit: float) -> Dict:
    C = int(bin_capacity)
    w = weights
    n = len(w)

    start = time.perf_counter()
    deadline = start + float(time_limit)

    # ---------- Helpers ----------
    def time_up() -> bool:
        return time.perf_counter() >= deadline

    def ffd_initial() -> tuple[list[list[int]], list[int], list[int]]:
        order = list(range(n))
        order.sort(key=lambda i: w[i], reverse=True)
        bins: list[list[int]] = []
        bw: list[int] = []
        item_bin = [-1] * n
        for i in order:
            wi = w[i]
            best_b = -1
            best_rem = C + 1
            for b in range(len(bins)):
                rem = C - bw[b]
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
                bw.append(wi)
            else:
                bins[best_b].append(i)
                bw[best_b] += wi
            item_bin[i] = best_b
        return bins, bw, item_bin

    def normalize(bins: list[list[int]], bw: list[int], item_bin: list[int]) -> None:
        # Remove empty bins by swapping with end for O(1)
        b = 0
        while b < len(bins):
            if bins[b]:
                b += 1
                continue
            last = len(bins) - 1
            if b == last:
                bins.pop()
                bw.pop()
                break
            # move last into b
            bins[b] = bins[last]
            bw[b] = bw[last]
            for it in bins[b]:
                item_bin[it] = b
            bins.pop()
            bw.pop()

    # Energy: prioritize few bins, then waste
    # waste = sum (C - load)
    # energy = bins*BIG + waste
    BIG = C * (n + 1) + 10_000

    def energy(bw: list[int]) -> int:
        waste = 0
        for load in bw:
            waste += (C - load)
        return len(bw) * BIG + waste

    # ---------- Initial solution ----------
    bins, bw, item_bin = ffd_initial()
    normalize(bins, bw, item_bin)

    best_bins = [b[:] for b in bins]
    best_bw = bw[:]
    best_item_bin = item_bin[:]
    best_E = energy(best_bw)

    cur_E = best_E

    # ---------- SA parameters ----------
    # Temperature scale: based on typical waste changes
    avg_w = sum(w) / n if n else 1.0
    T0 = max(1.0, 0.5 * avg_w)
    Tmin = 1e-4

    # Choose a fixed iteration budget; still respects time limit.
    # Scale with n, but cap to keep reasonable.
    base_iters = 30_000 + 400 * n
    max_iters = min(base_iters, 600_000)

    # Cooling schedule: exponential
    alpha = math.exp(math.log(Tmin / T0) / max(1, max_iters))

    # Stagnation / reheating
    stagnation = 0
    stag_limit = max(2_000, 20 * n)
    reheat_factor = 1.8

    # Neighborhood move probabilities
    # 0: relocate, 1: swap
    p_relocate = 0.72

    # For faster candidate selection: maintain list of non-empty bins indices (implicitly all)

    def try_relocate(T: float) -> bool:
        nonlocal cur_E

        if len(bins) == 0:
            return False

        # pick an item; bias toward items in light bins (more likely to empty a bin)
        b1 = random.randrange(len(bins))
        # retry a couple times to find a bin with items
        for _ in range(3):
            if bins[b1]:
                break
            b1 = random.randrange(len(bins))
        if not bins[b1]:
            return False

        i = random.choice(bins[b1])
        wi = w[i]

        # Target bin selection: try a handful of random bins + best-fit among them
        best_b2 = -1
        best_rem = C + 1

        # include possibility of new bin (always feasible)
        # but we only create if not accepted elsewhere

        k = 10 if len(bins) > 10 else len(bins)
        for _ in range(k):
            b2 = random.randrange(len(bins))
            if b2 == b1:
                continue
            rem = C - bw[b2]
            if wi <= rem:
                new_rem = rem - wi
                if new_rem < best_rem:
                    best_rem = new_rem
                    best_b2 = b2
                    if best_rem == 0:
                        break

        # If no feasible existing bin, optionally consider opening a new bin
        # (usually worse) but can help escape constraints for later swaps.
        open_new = (best_b2 == -1)

        # Compute delta energy efficiently
        # Current waste changes only for affected bins; bin count may change if b1 becomes empty
        before_bins = len(bins)
        before_waste = (C - bw[b1]) + (0 if open_new else (C - bw[best_b2]))
        after_waste = (C - (bw[b1] - wi)) + (0 if open_new else (C - (bw[best_b2] + wi)))

        # If opening new bin, add its waste too
        if open_new:
            before_waste += 0
            after_waste += (C - wi)

        # Bin count change if b1 empties and we do not open_new?
        b1_will_empty = (len(bins[b1]) == 1)
        # If open_new and b1 empties, bin count stays same (close one, open one)
        # If open_new and b1 not empty, bin count +1
        # If not open_new and b1 empties, bin count -1
        after_bins = before_bins
        if open_new:
            if not b1_will_empty:
                after_bins = before_bins + 1
        else:
            if b1_will_empty:
                after_bins = before_bins - 1

        dE = (after_bins - before_bins) * BIG + (after_waste - before_waste)

        if dE <= 0 or random.random() < math.exp(-dE / max(T, 1e-12)):
            # apply move
            # remove from b1
            bins[b1].remove(i)
            bw[b1] -= wi

            if open_new:
                b2 = len(bins)
                bins.append([i])
                bw.append(wi)
                item_bin[i] = b2
            else:
                b2 = best_b2
                bins[b2].append(i)
                bw[b2] += wi
                item_bin[i] = b2

            # If b1 empty, normalize to drop it (and fix indices)
            if not bins[b1]:
                normalize(bins, bw, item_bin)

            cur_E += dE
            return True
        return False

    def try_swap(T: float) -> bool:
        nonlocal cur_E

        if len(bins) < 2:
            return False

        b1 = random.randrange(len(bins))
        b2 = random.randrange(len(bins) - 1)
        if b2 >= b1:
            b2 += 1

        if not bins[b1] or not bins[b2]:
            return False

        i = random.choice(bins[b1])
        j = random.choice(bins[b2])
        wi, wj = w[i], w[j]

        # feasibility after swap
        load1 = bw[b1] - wi + wj
        load2 = bw[b2] - wj + wi
        if load1 > C or load2 > C:
            return False

        before_waste = (C - bw[b1]) + (C - bw[b2])
        after_waste = (C - load1) + (C - load2)
        dE = (after_waste - before_waste)  # bin count unchanged

        if dE <= 0 or random.random() < math.exp(-dE / max(T, 1e-12)):
            # apply
            bins[b1].remove(i)
            bins[b2].remove(j)
            bins[b1].append(j)
            bins[b2].append(i)
            bw[b1] = load1
            bw[b2] = load2
            item_bin[i] = b2
            item_bin[j] = b1
            cur_E += dE
            return True
        return False

    # ---------- Main SA loop ----------
    T = T0
    for it in range(max_iters):
        if (it & 1023) == 0 and time_up():
            break

        moved = False
        if random.random() < p_relocate:
            moved = try_relocate(T)
        else:
            moved = try_swap(T)

        if moved:
            stagnation = 0
        else:
            stagnation += 1

        # Track best
        if cur_E < best_E:
            best_E = cur_E
            best_bins = [b[:] for b in bins]
            best_bw = bw[:]
            best_item_bin = item_bin[:]

        # Cooling
        T *= alpha
        if T < Tmin:
            T = Tmin

        # Reheat if stuck
        if stagnation >= stag_limit:
            T = min(T0, T * reheat_factor)
            stagnation = 0

    # ---------- Return best found ----------
    # Ensure output aligned and valid
    packing = best_bins
    bin_weights = best_bw

    return {"packing": packing, "bin_weights": bin_weights}
