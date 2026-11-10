"""
src/search.py - Path aggregation, Best-of-N search, and compute-matched search.
"""
from typing import Any, Callable

import time


from .generation import compute_majority_vote, run_self_consistency
from .prm_judge import score_path

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]


def aggregate_path_score(
    step_scores: list[float], 
    strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> float:
    """
    Aggregates a list of step scores (+1.0 or -1.0) into a scalar trajectory score.
    
    Rules:
      - An empty list returns -1.0 for every strategy.
      - 'min': min(step_scores).
      - 'product': p_t = clamp((r_t + 1)/2, 0.01, 0.99); returns the raw product.
      - 'step_penalty': mean(step_scores) - penalty_lambda * count(r_t == -1.0).
      - An unknown strategy raises ValueError.
    """
    # TODO: Implement min, product, and step_penalty aggregation

    
    if strategy not in ("min", "product", "step_penalty"):
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
    if not step_scores:
        return -1.0
    if strategy == "min":
        return float(min(step_scores))
    if strategy == "product":
        prod = 1.0
        for r in step_scores:
            prod *= min(max((r + 1.0) / 2.0, 0.01), 0.99)
        return prod
    n_bad = sum(1 for r in step_scores if r == -1.0)
    return sum(step_scores) / len(step_scores) - penalty_lambda * n_bad



def select_best_path(
    candidate_paths: list[dict[str, Any]], 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> dict[str, Any]:
    """
    Selects the winning candidate path from candidate_paths.
    
    Input Dictionary Contract:
      Each candidate dictionary in candidate_paths must contain 'step_scores' (list[float])
      and 'answer' (int | None). Any other keys present in the candidate dictionary
      are passed through unchanged in the returned copy.
    
    Selection & Tie-Breaking Rules:
      1. Compute path_score for each candidate with aggregate_path_score.
         A candidate with an empty step_scores list gets -1.0.
      2. Eligibility: if at least one candidate has a non-None answer, candidates
         with answer=None are excluded. If all answers are None, all remain eligible.
      3. Find maximum path_score among eligible candidates (tolerance 1e-9).
         The tied set is all eligible candidates within tolerance of the maximum.
      4. If the tied set has more than one candidate, compute compute_majority_vote
         over their answers. Return the lowest-index candidate in the tied set whose
         answer equals the vote. If the vote is None, return the lowest-index tied candidate.
      5. Returns a shallow COPY of the chosen dict with two added keys:
         'path_score' (float) and 'num_tied' (int, size of the tied set).
         Does not mutate the inputs.
    """
    # TODO: Implement best path selection with answer=None filtering and majority tie-breaking

    if not candidate_paths:
        raise ValueError("candidate_paths must be non-empty")

    scores = [
        aggregate_path_score(c.get("step_scores", []), aggregation_strategy, penalty_lambda)
        for c in candidate_paths
    ]
    eligible = list(range(len(candidate_paths)))
    if any(c.get("answer") is not None for c in candidate_paths):
        eligible = [i for i in eligible if candidate_paths[i].get("answer") is not None]

    best = max(scores[i] for i in eligible)
    tied = [i for i in eligible if abs(scores[i] - best) <= 1e-9]

    chosen = tied[0]
    if len(tied) > 1:
        vote = compute_majority_vote([candidate_paths[i].get("answer") for i in tied])
        if vote is not None:
            chosen = next(i for i in tied if candidate_paths[i].get("answer") == vote)

    out = dict(candidate_paths[chosen])
    out["path_score"] = float(scores[chosen])
    out["num_tied"] = len(tied)
    return out


def run_prm_best_of_n(
    model: Any = None, 
    judge_model: Any = None, 
    gen_tokenizer: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0, 
    candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Executes PRM-guided Best-of-N search over reasoning trajectories.
    
    If candidates is provided (shared candidate pool), does NOT sample new candidates
    (generation_calls = 0, generated_tokens = 0). Provided candidates use the
    run_self_consistency candidate format (at minimum 'steps' and 'answer').
    Otherwise, samples n candidates using run_self_consistency.
    
    Scores each candidate's steps using score_path.
    Selects the winning candidate via select_best_path.
    
    Returns:
      selected, answer, candidates [each augmented with step_scores, rationales],
      generation_calls, judge_calls, generated_tokens, judge_tokens, time_seconds
    """
    # TODO: Implement PRM Best-of-N search over candidate trajectories
    t0 = time.perf_counter()
    generation_calls, generated_tokens = 0, 0

    if candidates is None:
        sc = run_self_consistency(model=model, tokenizer=gen_tokenizer, question=question,
                                  n=n, generate_fn=generate_fn)
        cands = [dict(c) for c in sc["candidates"]]
        generation_calls = sc["generation_calls"]
        generated_tokens = sc["generated_tokens"]
    else:
        cands = [dict(c) for c in candidates]   # copies, caller's dicts untouched

    judge_calls, judge_tokens = 0, 0
    for c in cands:
        r = score_path(judge_model, judge_tokenizer, question, c, judge_fn=judge_fn)
        c["step_scores"] = r["step_scores"]
        c["rationales"] = r["rationales"]
        judge_calls += r["judge_calls"]
        judge_tokens += r["judge_tokens"]

    selected = select_best_path(cands, aggregation_strategy, penalty_lambda)
    return {
        "selected": selected,
        "answer": selected.get("answer"),
        "candidates": cands,
        "generation_calls": generation_calls,
        "judge_calls": judge_calls,
        "generated_tokens": generated_tokens,
        "judge_tokens": judge_tokens,
        "time_seconds": time.perf_counter() - t0,
    }


def run_compute_matched_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    target_calls: int = 15, 
    temperature: float = 0.7, 
    existing_candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    
    """
    Executes compute-matched Self-Consistency aggregating target_calls paths.
    If existing_candidates are supplied, samples only the deficit:
    target_calls - len(existing_candidates). Returned candidates = existing_candidates followed by new samples, in sampling order.
    
    Returns:
      candidates, answer, generation_calls (new samples only), total_samples,
      generated_tokens (new samples only), time_seconds
    """
    # TODO: Implement compute-matched Self-Consistency with sample reuse
    t0 = time.perf_counter()
    existing = [dict(c) for c in (existing_candidates or [])]
    deficit = max(0, target_calls - len(existing))

    new_cands, new_tokens = [], 0
    if deficit > 0:
        sc = run_self_consistency(model=model, tokenizer=tokenizer, question=question,
                                  n=deficit, temperature=temperature, generate_fn=generate_fn)
        new_cands = sc["candidates"]
        new_tokens = sc["generated_tokens"]

    allc = existing + new_cands
    return {
        "candidates": allc,
        "answer": compute_majority_vote([c.get("answer") for c in allc]),
        "generation_calls": len(new_cands),
        "total_samples": len(allc),
        "generated_tokens": new_tokens,
        "time_seconds": time.perf_counter() - t0,
    }
