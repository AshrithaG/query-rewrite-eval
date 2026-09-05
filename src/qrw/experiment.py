"""How much traffic does it take to see a ranking change?

Offline retrieval numbers say system A beats system B on a fixed set. Shipping
asks a harder question: how many real queries must you serve before that
difference is distinguishable from noise, and is there a cheaper design than
splitting traffic in half.

Three things are implemented here, in increasing order of how much they buy:

  paired vs unpaired A/B   both systems answer the same query, so the paired
                           test removes query difficulty from the variance
  team-draft interleaving  one result list built from both systems, so a single
                           impression is a direct comparison
  power curves             the sample size each design needs for a given effect
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def ndcg_from_rank(rank: int, k: int = 10) -> float:
    """0 for a miss, otherwise the single-relevant-document nDCG."""
    return 0.0 if (rank <= 0 or rank > k) else float(1.0 / np.log2(rank + 1))


@dataclass
class PowerResult:
    n_required: int | None      # queries needed for the target power
    effect: float               # observed mean difference
    sd: float                   # sd of the quantity the test uses
    cohens_d: float
    design: str

    def __str__(self) -> str:
        n = f"{self.n_required:,}" if self.n_required else "not detectable"
        return f"{self.design:<22} effect {self.effect:+.4f}  d {self.cohens_d:+.3f}  n={n}"


def _n_for_power(d: float, power: float = 0.80, alpha: float = 0.05) -> int | None:
    """Sample size for a two-sided test at the given effect size.

    Normal approximation, which is what everyone uses for traffic planning and
    is accurate well before the sample sizes search systems deal in.
    """
    if not np.isfinite(d) or abs(d) < 1e-9:
        return None
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(((z_a + z_b) / abs(d)) ** 2))


def paired_ab(a: np.ndarray, b: np.ndarray, power: float = 0.80) -> PowerResult:
    """Both systems answer every query; the test is on per-query differences.

    This is the offline situation, and it is the most sensitive design because
    query difficulty cancels out.
    """
    diff = a - b
    sd = float(diff.std(ddof=1))
    d = float(diff.mean() / sd) if sd > 0 else 0.0
    return PowerResult(_n_for_power(d, power), float(diff.mean()), sd, d, "paired A/B")


def unpaired_ab(a: np.ndarray, b: np.ndarray, power: float = 0.80) -> PowerResult:
    """Traffic is split, so each query is seen by one system only.

    This is what an online A/B test actually is, and the variance is the pooled
    spread of query difficulty rather than the spread of the difference. `n` is
    per arm.
    """
    sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    effect = float(a.mean() - b.mean())
    d = effect / sd if sd > 0 else 0.0
    n = _n_for_power(d, power)
    return PowerResult(2 * n if n else None, effect, sd, d, "unpaired A/B (total)")


def team_draft_interleave(rank_a: int, rank_b: int, k: int = 10,
                          rng: np.random.Generator | None = None) -> int:
    """One impression of team-draft interleaving, returning +1 if A wins.

    Both systems contribute to a single result list, alternating picks with a
    coin flip for who picks first. The user clicks the relevant document if it
    appears; credit goes to whichever system contributed that position. Because
    a single impression compares the systems directly, interleaving is far more
    sensitive per impression than splitting traffic.

    Simplified to the one-relevant-document case this dataset provides: the team
    that placed the relevant document earlier wins the impression.
    """
    rng = rng or np.random.default_rng()
    a_hit = 0 < rank_a <= k
    b_hit = 0 < rank_b <= k
    if not a_hit and not b_hit:
        return 0                      # no click, no information
    if a_hit and not b_hit:
        return 1
    if b_hit and not a_hit:
        return -1
    if rank_a == rank_b:
        return int(rng.integers(0, 2)) * 2 - 1   # tie broken by the coin flip
    return 1 if rank_a < rank_b else -1


def interleaving_power(ranks_a: np.ndarray, ranks_b: np.ndarray, k: int = 10,
                       power: float = 0.80, seed: int = 0) -> PowerResult:
    """Sample size for interleaving, where the statistic is the win rate."""
    rng = np.random.default_rng(seed)
    outcomes = np.array([team_draft_interleave(int(x), int(y), k, rng)
                         for x, y in zip(ranks_a, ranks_b)], dtype=float)
    informative = outcomes[outcomes != 0]
    if informative.size == 0:
        return PowerResult(None, 0.0, 0.0, 0.0, "interleaving")
    # Deviation from a 50/50 win rate, expressed on the same scale as the tests
    # above so the sample sizes are comparable.
    sd = float(informative.std(ddof=1))
    d = float(informative.mean() / sd) if sd > 0 else 0.0
    n = _n_for_power(d, power)
    # impressions that produced no click carry no signal, so the traffic needed
    # is larger than the number of informative impressions
    inflate = len(outcomes) / informative.size
    return PowerResult(int(np.ceil(n * inflate)) if n else None,
                       float(informative.mean()), sd, d, "interleaving")


def detectable_effect(sd: float, n: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """The smallest difference n queries can resolve. The planning question."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) * sd / np.sqrt(n))
