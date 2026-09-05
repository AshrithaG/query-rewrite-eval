# How much traffic does it take to ship a ranking change?

Offline evaluation asks which system scores higher on a fixed set. Shipping asks
how many real queries you must serve before that difference is distinguishable
from noise, and whether there is a cheaper design than splitting traffic. Run
with `PYTHONPATH=src python scripts/power_analysis.py` over 12,561 QReCC turns.

## Traffic required, versus `concat_2`, at 80% power and alpha 0.05

| system | ΔnDCG@10 | paired | A/B split | interleaved | interleaving vs A/B |
|---|---|---|---|---|---|
| raw | -0.2179 | 25 | 34 | 25 | 1.4x |
| concat_1 | -0.0373 | 190 | 1,088 | 859 | 1.3x |
| concat_all | -0.0162 | 985 | 5,092 | 299 | 17.0x |
| first_turn | -0.0121 | 6,073 | 13,064 | 27,259 | **0.5x** |
| concat_4 | -0.0084 | 2,371 | 18,648 | 344 | **54.2x** |
| gold | +0.1161 | 60 | 144 | 77 | 1.9x |

## Interleaving is not uniformly better, and the reason is not size

The usual claim is that interleaving is an order of magnitude more sensitive
than an A/B split. Two rows here disagree with each other, and the disagreement
is the interesting part.

| comparison vs `concat_2` | ΔnDCG | head-to-head win rate | interleaving vs A/B |
|---|---|---|---|
| `concat_4` | -0.0084 | 0.415 | 54.2x |
| `first_turn` | -0.0121 | 0.490 | 0.5x |

`first_turn` has the **larger** mean difference of the two and interleaving still
does worse than a traffic split on it.

What separates them is consistency of direction. `concat_4` loses to `concat_2`
on 58% of informative impressions, steadily, while their means differ by almost
nothing because its wins and losses are of similar size and cancel. Interleaving
counts wins, so it sees a clear signal where the mean sees a coin flip.

`first_turn` is the mirror image. It wins 49% of impressions, an actual coin
flip, but its losses are larger than its wins, so the mean separates while the
win count does not.

So the rule is not that interleaving is more sensitive. It is that **interleaving
measures a different quantity**: pairwise preference rather than mean utility.
Those coincide when one system dominates the other consistently, and come apart
when systems trade wins of unequal magnitude. Choosing interleaving because a
paper reported a 10x sensitivity gain is choosing it for the wrong reason; the
question is whether you expect your change to be consistent or to be a
redistribution.

Roughly a quarter of impressions produce no click under this click model,
because neither system retrieves the relevant document in the top 10. Those
carry no information, and the traffic numbers above include that inflation.

## Why real search launches report such small numbers

Smallest nDCG difference each traffic level can resolve, paired design:

| queries | detectable ΔnDCG | as % of baseline |
|---|---|---|
| 1,000 | 0.01301 | 3.34% |
| 10,000 | 0.00411 | 1.06% |
| 100,000 | 0.00130 | 0.33% |
| 1,000,000 | 0.00041 | 0.11% |
| 10,000,000 | 0.00013 | 0.03% |

Production search teams report relevance lifts in the range of 0.15% to 0.44%.
The table explains why that is the normal size of a real launch rather than a
disappointing one: **detecting a 0.15% lift takes on the order of a million
queries**, and anything smaller than about 1% is invisible below 10,000. An
offline benchmark of a few thousand queries cannot resolve the effects that
production A/B tests are built to find, which is the actual reason offline and
online results disagree so often.

## The design choice, stated plainly

Pairing is worth 5x to 8x on its own, because the same query answered by both
systems removes query difficulty from the variance. That is free offline and
impossible online, which is most of the gap between offline and online
sensitivity. Interleaving recovers some of it by putting both systems in one
impression, but only for changes that are consistent in direction.
