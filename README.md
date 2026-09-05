# Does ROUGE measure query rewriting?

Conversational query rewriting turns "what does it cost?" into "what does it
cost to become a physician's assistant?" so a retrieval system can answer it.
The field reports ROUGE against a human reference rewrite. Nobody retrieves
anything with ROUGE.

This measures both on the same 12,561 QReCC turns: ROUGE against the reference,
and actual retrieval against a fixed corpus. They disagree, and the shape of the
disagreement turns out to matter more than the fact of it.

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/evaluate.py        # baselines, no GPU, ~10 min
bash scripts/run_all.sh                          # + fine-tuning, one GPU, ~3 h
```

## The headline

Across systems that genuinely differ in approach, **ROUGE-L ranks them close to
backwards from recall@10**: Kendall tau between -0.73 and -0.87 depending on
corpus size, stable across a fivefold change in collection size.

| system | ROUGE-L | recall@10 | MRR@10 | nDCG@10 | words |
|---|---|---|---|---|---|
| gold (human reference) | 1.000 | 0.715 | 0.439 | 0.505 | 10.3 |
| fine-tuned Qwen3-1.7B | 0.844 | 0.691 | 0.415 | 0.481 | 9.6 |
| base Qwen3-1.7B, zero-shot | 0.544 | 0.595 | 0.339 | 0.400 | 19.2 |
| concat last 2 turns | 0.317 | 0.702 | 0.289 | 0.389 | 35.2 |
| **concat last 4 turns** | **0.230** | **0.737** | 0.269 | 0.381 | 58.4 |
| first turn + question | 0.500 | 0.559 | 0.320 | 0.377 | 14.3 |
| **concat whole conversation** | **0.178** | **0.735** | 0.261 | 0.373 | 105.8 |
| **no rewriting at all** | **0.665** | **0.240** | 0.150 | 0.171 | 6.4 |

On this corpus the two systems with the **worst** ROUGE retrieve **more**
relevant documents than the human reference does. That specific result turns out
to be corpus-size dependent and is corrected below; the correlation itself is
not.

The finding that does hold at every scale tested: the system with the third-best
ROUGE is the one that does no rewriting at all. The unmodified question is mostly a substring of the reference rewrite,
so it inherits high n-gram overlap while retrieving at 0.240.

ROUGE is measuring resemblance to a short human sentence. Retrieval rewards
having the right terms present. Those are not the same objective, and on this
data they are close to opposed.


## The claim that did not survive, and the one that did

The obvious attack on the table above is corpus size. With only 12,353
documents, a verbose query that matches many terms can still surface the right
document in the top ten because there is little competition. On a realistic
collection it should pull in far more noise.

That attack is correct, so it is worth running rather than acknowledging. Adding
answers from the training split as distractors, holding the queries fixed:

| corpus | concat_4 recall | gold recall | concat_4 advantage |
|---|---|---|---|
| 12,353 | 0.737 | 0.715 | **+0.022** |
| 24,353 | 0.683 | 0.667 | +0.016 |
| 52,353 | 0.599 | 0.599 | +0.000 |
| 59,149 | 0.590 | 0.592 | **-0.002** |

**Concatenation beating the human rewrite is an artifact of a small corpus.** It
decays to nothing by 52k documents and would be gone entirely on a real
collection. Any claim resting on it should be discarded.

The correlation is a different matter:

| corpus | tau, ROUGE vs recall@10 | p |
|---|---|---|
| 12,353 | -0.867 | 0.017 |
| 24,353 | -0.867 | 0.017 |
| 52,353 | -0.733 | 0.056 |
| 59,149 | -0.733 | 0.056 |

Across a fivefold increase in collection size the anti-correlation is stable in
direction and magnitude. What changes is which system sits at the top, not
whether ROUGE orders them backwards. With six systems the p-value is marginal at
the larger sizes, and it should be read as such.

## The part that nearly fooled me

Adding seven fine-tuning checkpoints flips the correlation to **+0.714**, and
the regret of choosing by ROUGE drops to zero. Taken at face value, that says
ROUGE is fine after all.

It is an artifact of pooling, and it is worth spelling out because it is an easy
trap in any metric-versus-metric study.

| group | ROUGE spread | nDCG spread |
|---|---|---|
| 7 checkpoints of one run | 0.013 | 0.003 |
| 7 genuinely different systems | 0.487 | 0.229 |

The checkpoints are one system measured seven times. Of the 21 checkpoint pairs,
**3 differ significantly** under a Bonferroni-corrected paired t-test over
12,561 queries, and the largest gap between any two is 0.0037 nDCG. Two of them
are the same model saved one step apart (+0.00012, p=0.718). For contrast, a
comparison that is real, the fine-tuned model against concat-2, gives +0.092
with p = 6e-229.

Correlating a tight cluster together with a spread-out group measures the gap
between the groups, not agreement within either. Collapse the checkpoints to one
representative and the correlation falls apart again:

| systems correlated | vs nDCG@10 | vs recall@10 |
|---|---|---|
| all 14 | +0.714 (p=0.000) | +0.045 (p=0.826) |
| checkpoints collapsed to 1 | +0.214 (p=0.548) | -0.571 (p=0.061) |
| the 7 distinct systems | -0.048 (p=1.000) | **-0.810 (p=0.011)** |
| the 7 checkpoints only | +0.714 (p=0.030) | +0.720 (p=0.028) |

That last row is the honest nuance. Within a training run, ROUGE and retrieval
do move together. They are also both moving across a range too small to act on.


## The retrieval layer, and two things measuring it revealed

The BM25 index here is written rather than imported: numpy postings with a
per-term offset table, agreeing with a reference implementation on 97% of top-1
results under matched parameters. It was written by hand for one reason, which
turned into the first finding.

### Sharding a BM25 index silently changes the answer

BM25 weights a term by how rare it is in the collection. A shard holds a slice,
so a shard computing IDF from its own documents weights terms by their rarity
*there*, and the same document scores differently depending on which shard it
landed on. Nothing raises an error. The results look fine.

Top-10 agreement against a single unsharded index over the same collection:

| shards | global IDF | per-shard IDF | top-1 preserved |
|---|---|---|---|
| 2 | 97.1% | 90.6% | 93.6% |
| 4 | 97.1% | 85.9% | 88.8% |
| 8 | 96.9% | 83.3% | 86.9% |
| 16 | 96.8% | 79.6% | 83.9% |

At 16 shards a fifth of the top-10 is wrong and **one query in six gets a
different top result**, silently. Distributing global document frequencies
before building any shard holds fidelity flat at ~97% regardless of shard count,
and costs one pass over the collection. Off-the-shelf BM25 libraries do not
expose the document frequencies, which is why this index accepts them as input.

### Fan-out made it 25 times slower, and that is the useful number

| documents | build | index size | search p50 |
|---|---|---|---|
| 12,353 | 0.1 s | 3 MB | 0.07 ms |
| 50,000 | 1.2 s | 18 MB | 0.17 ms |
| 100,000 | 2.5 s | 32 MB | 0.52 ms |
| 200,000 | 5.4 s | 60 MB | 0.72 ms |
| 400,000 | 10.8 s | 114 MB | 1.48 ms |

A loopback HTTP round trip costs about 1 ms, so a shard doing less than roughly
1 ms of work spends more on being called than on searching. At the scale this
evaluation actually runs, splitting the collection eight ways saves 0.04 ms of
search and adds about 1 ms of coordination.

Measured rather than predicted: fan-out to eight local processes took p50 at
concurrency 32 from **21.6 ms to 776 ms**, because eight uvicorn workers and a
coordinator were competing for the same twelve cores in order to avoid 0.07 ms
of work.

The crossover is near 400k documents per shard. Below it, distributing this is
strictly worse, and the correct engineering decision is the one that looks least
impressive on a diagram.

## What this means if you are shipping something

**Choosing between approaches on ROUGE will mislead you.** The metric prefers
short outputs resembling one human sentence; retrieval prefers the right terms
being present at all. On this data the preference is close to inverted.

**Choosing a checkpoint on ROUGE is fine, and mostly pointless.** After the
first 500 steps the checkpoints are within 0.004 nDCG of each other. Early
stopping on ROUGE will not hurt you, because there is almost nothing left to
decide.

**Recall and ranking quality come apart.** Concatenation matches the human
rewrite on recall@10 while losing badly on MRR (0.269 against 0.439). It puts
the right document in the top ten and then buries it, because the extra
conversational terms match many documents. If your pipeline reranks, cheap
concatenation may be enough; if it does not, the ranking difference is the whole
game.

## The fine-tuning does work

Qwen3-1.7B with LoRA on 48,415 QReCC pairs reaches 0.481 nDCG against the human
reference's 0.505, at 9.6 words against 10.3, from a base of 0.400 zero-shot.
The model learned the task. That is the point: a genuinely good system and a
crude string concatenation sit at opposite ends of ROUGE while landing much
closer on the thing anyone cares about.

## Method

QReCC turns that have conversational context, so the ones where rewriting can do
anything. The retrieval task is self-contained: the corpus is every unique
answer in the split (12,353 documents), a query's relevant document is its own
answer, and every other answer is a distractor. That is smaller and easier than
the official 54M-passage QReCC collection, so these absolute numbers are not
comparable to published QReCC retrieval results. The comparison the study needs
is between metrics on identical queries, which this supports.

BM25 rather than a dense retriever, deliberately: a dense model would fold its
own semantics into the outcome, so a rewrite could score well because the encoder
liked it. BM25 is transparent and lexical, so a rewrite that retrieves better did
so by putting better terms in the query.

Full detail on the index and sharding work is in `results/idf_sharding.md` and
`results/scaling.md`, and the experiment-design analysis of how much traffic it
takes to detect these differences online is in `results/experiment_design.md`.

## What this is not

The retrieval task is constructed, not the official QReCC benchmark: 12k to 59k
short answers rather than 54M passages. The correlation held across the range
tested, but 59k is still two to three orders of magnitude below a production
collection, and nothing here establishes what happens there.

BM25 only. A dense retriever may behave differently, and most modern systems use
one. The choice was deliberate, since a dense encoder would fold its own
semantics into the result, but it does limit what the finding covers.

Six to seven distinct systems is a small sample for a rank correlation. The
p-values are 0.017 to 0.056 and should be read as suggestive rather than
settled.

Metric validity in query rewriting is not a new concern in the IR literature.
What this offers is a controlled measurement of it on one dataset, with the
confounds tested rather than assumed.
