"""
src/generation.py - Prompt formatting and baseline generation routines.
"""
from typing import Any, Callable
from collections import Counter
from .extraction import extract_answer, parse_steps
import time
from .utils import (make_hf_generate_fn, GREEDY_GEN_KWARGS, SAMPLING_GEN_KWARGS)
GenerateFn = Callable[[str, int, dict], list[dict]]
# generate_fn(raw_prompt, num_samples, gen_kwargs) -> list of {"text": str, "num_tokens": int}
# gen_kwargs carries do_sample/temperature/top_p/top_k/repetition_penalty/max_new_tokens


def format_step_prompt(question: str) -> str:
    
    """
    Formats the raw problem question into a prompt that instructs the 
    model to emit numbered steps ('Step 1: ...') and 'Final Answer: <int>'.
    
    Returns a raw instruction string (no chat template applied).
    """
    # TODO: Build zero-shot prompt enforcing step-structured reasoning
    prompt=(
        
        f"Question: {question}\n\n"
       
        "Give me the solution to the math question. Show all the numbered steps short explanation and provide a final answer as an integer only.\n"
       
        "Follow the Format below \n"
        
       
        "Step 1: one short plain sentence with one calculation written like 1 * 5 = 5.\n"
        "Step 2: next short sentence with one calculation.\n"
        "Step 3: Continue with the next step and state the current values\n"
       
        "(continue, one step per line)\n"
        "Final Answer: <integer>\n\n"

        "Do not deviate from the following rules:"

         "Use plain text only. No markdown, no LaTeX, no \\[ \\], no \\boxed{}, no brackets.\n"
    
        "no units in the final answer.\n"
        "Every line before the last starts with 'Step k:'. No concluding sentence.\n"
        
        
        "write nothing after the last line that is the Final Answer line\n"
        "The last line must be only be 'Final Answer: number' replace the number placeholder with the actual answer value only"
        

        
   


       )
    
    return prompt

    
    

def compute_majority_vote(answers: list[int | None]) -> int | None:
    """
    Given a list of extracted answers, returns the majority answer.
    Ignores None values. Breaks ties deterministically in favor of the answer
    that appeared earliest in the list.
    """
    # TODO: Implement deterministic majority voting
    votes = [a for a in answers if a is not None]
    if not votes:
        return None
    counts = Counter(votes)
    highest = max(counts.values())
    for a in votes:                  
        if counts[a] == highest:
            return a
    

def run_greedy(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes greedy decoding for a single question.
    Calls generate_fn with num_samples=1 and gen_kwargs containing do_sample=False,
    temperature=None, top_p=None, top_k=None.
    
    Returns:
      prompt, completion, steps, answer, generation_calls, generated_tokens, time_seconds
    """
    # TODO: Implement greedy baseline decoding
    if generate_fn is None:
        generate_fn = make_hf_generate_fn(model, tokenizer)
    gen_kwargs = dict(GREEDY_GEN_KWARGS)
    prompt = format_step_prompt(question)
    start = time.perf_counter()
    out = generate_fn(prompt, 1, gen_kwargs)[0]
    elapsed = time.perf_counter() - start
    completion = out.get("text", "")
    return {
        "prompt": prompt,
        "completion": completion,
        "steps": parse_steps(completion),
        "answer": extract_answer(completion),
        "generation_calls": 1,
        "generated_tokens": int(out.get("num_tokens", 0)),
        "time_seconds": elapsed,
    }




def run_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    temperature: float = 0.7, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Samples n diverse reasoning trajectories and aggregates candidate answers via majority vote.
    
    Returns:
      candidates [each: completion, steps, answer, num_tokens], answer, 
      generation_calls, generated_tokens, time_seconds
    """
    # TODO: Implement Self-Consistency sampling and majority voting
    if n < 1:
        raise ValueError("n must be >= 1")
    if generate_fn is None:
        generate_fn = make_hf_generate_fn(model, tokenizer)

    prompt = format_step_prompt(question)
    gen_kwargs = dict(SAMPLING_GEN_KWARGS)
    gen_kwargs["temperature"] = temperature

    start = time.perf_counter()
    outputs = generate_fn(prompt, n, gen_kwargs)
    elapsed = time.perf_counter() - start

    candidates = []
    answers = []
    generated_tokens = 0

    for output in outputs:
        completion = output.get("text", "")
        steps = parse_steps(completion)
        answer = extract_answer(completion)
        num_tokens = int(output.get("num_tokens", 0))

        candidates.append({
            "completion": completion,
            "steps": steps,
            "answer": answer,
            "num_tokens": num_tokens,
        })
        answers.append(answer)
        generated_tokens += num_tokens

    return {
        "candidates": candidates,
        "answer": compute_majority_vote(answers),   
        "generation_calls": len(candidates),
        "generated_tokens": generated_tokens,
        "time_seconds": elapsed,
    }
