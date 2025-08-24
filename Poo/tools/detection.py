import numpy as np


def find_top_local_extrema(signal, n, sign="both"):
    candidates = []
    for i in range(1, len(signal) - 1):
        if sign == "pos" and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            candidates.append((i, abs(signal[i])))
        elif sign == "neg" and signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            candidates.append((i, abs(signal[i])))
        elif sign == "both" and (
            (signal[i] > signal[i - 1] and signal[i] > signal[i + 1]) or
            (signal[i] < signal[i - 1] and signal[i] < signal[i + 1])
        ):
            candidates.append((i, abs(signal[i])))

    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    best_indices = [idx for idx, _ in sorted_candidates[:n]]

    while len(best_indices) < n:
        best_indices.append(best_indices[-1] if best_indices else 0)

    return best_indices
