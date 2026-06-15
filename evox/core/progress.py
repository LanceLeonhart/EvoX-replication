"""Window-level progress metrics for the outer loop.

EvoX evaluates a strategy over a window of W inner iterations:

    delta = s_end - s_start                       (raw improvement in the window)
    J     = delta * log(1 + s_start) / sqrt(W)    (strategy score)

``s_start`` / ``s_end`` are the best fitness (higher-is-better) at the window
boundaries. The ``log(1 + s_start)`` term discounts improvements made when the
search is already strong, and the ``1/sqrt(W)`` term normalises by window size
so windows of different lengths are comparable. Stagnation is declared when the
window's improvement falls at or below the threshold tau.
"""

from __future__ import annotations

import math


def window_delta(s_start: float, s_end: float) -> float:
    return s_end - s_start


def strategy_score(delta: float, s_start: float, window: int) -> float:
    w = max(1, int(window))
    # guard log against negative/zero fitness so the score stays well-defined
    return delta * math.log(1.0 + max(0.0, s_start)) / math.sqrt(w)


def is_stagnant(delta: float, tau: float) -> bool:
    return delta <= tau
