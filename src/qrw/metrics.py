"""ROUGE against the reference rewrite, and how it relates to retrieval.

The question this project asks is whether ROUGE, which is what query-rewriting
papers report, tells you anything about whether the rewrite retrieves better,
which is what anyone actually wants.
"""

from __future__ import annotations

import numpy as np
from rouge_score import rouge_scorer
from scipy import stats

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge(prediction: str, reference: str) -> dict[str, float]:
    s = _SCORER.score(reference, prediction)
    return {k: v.fmeasure for k, v in s.items()}


def rouge_all(predictions: list[str], references: list[str]) -> dict[str, float]:
    per = [rouge(p, r) for p, r in zip(predictions, references)]
    return {k: float(np.mean([d[k] for d in per])) for k in per[0]}


def per_query_rouge(predictions: list[str], references: list[str]) -> np.ndarray:
    return np.array([rouge(p, r)["rougeL"] for p, r in zip(predictions, references)])


def agreement(rouge_scores: np.ndarray, retrieval_scores: np.ndarray) -> dict[str, float]:
    """How well per-query ROUGE tracks per-query retrieval quality.

    Pearson assumes a linear relationship, Spearman only a monotone one, and
    Kendall is the one to quote when the question is about ordering, which it is
    here: does ROUGE put queries in the same order as retrieval does.
    """
    ok = np.isfinite(rouge_scores) & np.isfinite(retrieval_scores)
    r, rp = stats.pearsonr(rouge_scores[ok], retrieval_scores[ok])
    s, sp = stats.spearmanr(rouge_scores[ok], retrieval_scores[ok])
    t, tp = stats.kendalltau(rouge_scores[ok], retrieval_scores[ok])
    return {"pearson": float(r), "pearson_p": float(rp),
            "spearman": float(s), "spearman_p": float(sp),
            "kendall": float(t), "kendall_p": float(tp), "n": int(ok.sum())}


def system_ranking_agreement(rouge_by_system: dict[str, float],
                             retrieval_by_system: dict[str, float]) -> dict[str, float]:
    """Do the two metrics rank whole systems the same way?

    This is the question a practitioner faces: given several rewriters, does
    picking the best by ROUGE pick the best by retrieval?
    """
    names = sorted(rouge_by_system)
    a = np.array([rouge_by_system[n] for n in names])
    b = np.array([retrieval_by_system[n] for n in names])
    tau, p = stats.kendalltau(a, b)
    best_rouge = names[int(np.argmax(a))]
    best_retrieval = names[int(np.argmax(b))]
    # what you give up by trusting ROUGE to pick the system
    regret = float(max(b) - b[names.index(best_rouge)])
    return {"kendall_tau": float(tau), "p": float(p),
            "best_by_rouge": best_rouge, "best_by_retrieval": best_retrieval,
            "regret": regret, "n_systems": len(names)}
