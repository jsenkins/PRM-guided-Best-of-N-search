

"""
src/prm_judge.py - PRM judge prompt construction, parsing, and step scoring.
"""
from typing import Any, Callable
from .utils import make_hf_judge_fn
import re

JudgeFn = Callable[[str], dict]
# judge_fn(judge_prompt) -> {"text": str, "num_tokens": int}


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    
    """
    Builds the verification prompt for the PRM judge to classify current_step
    as VALID or INVALID given the problem question and preceding_steps.
    
    Judge Semantics (Local Validity):
    The judge evaluates current_step ASSUMING ALL PRECEDING STEPS ARE CORRECT.
    A step is VALID if its arithmetic is sound and it follows logically from the
    problem statement plus preceding steps as written. An error in an earlier step
    does NOT make subsequent steps invalid by itself.
    
    Autograder Literal Checks:
    - Preceding steps are rendered as 'Step 1: <text>', 'Step 2: <text>', ... in order.
    - The current step is rendered with the label 'Current Step:'.
    - The prompt contains the phrase 'assume all preceding steps are correct' (case-insensitive).
    - The prompt contains the literal strings 'Verdict: VALID', 'Verdict: INVALID', and 'Rationale:'.
    - With an empty preceding_steps list, the prompt states '(no preceding steps)'.
    """
    # TODO: Build verification prompt instructing 7B judge under local validity
   
    
    if preceding_steps:
        preceding = "\n".join(f"Step {i}: {s}" for i, s in enumerate(preceding_steps, start=1))
    else:
        preceding = "(no preceding steps)"


    judge_prompt = (
        f"For the problem {question}, classify whether the current step {current_step} is valid or invalid given the preceeding steps {preceding_steps}\n"
        
        "Do not deviate from the following instructions: \n"
        "Check the arithmetic and logical validity of each step. A VALID current step has sound arithmetic and it follows logically from the problem statement plus preceding steps as written so\n"
        "Validity depends on the current_step only. \n"
        "Preceeding INVALID step does not make current step INVALID. \n"
        "Preceding steps are rendered as 'Step 1: <text>', 'Step 2: <text>', ... in order.\n"
        "The current step is rendered with the label 'Current Step:'.\n"
        "Evaluate the Current Step. Assume all preceding steps are correct. \n"
        "With an empty preceding_steps list, the prompt states '(no preceding steps)'.\n"
        f"Problem:\n{question}\n\n"
        f"Preceding steps:\n{preceding}\n\n"
        f"Current Step:\n{current_step}\n\n"
       
        
        "Give the answer in the following format if it is VALID: \n"
        "Verdict: VALID \n"
        "Rationale: briefly state why it is valid. \n"

        "Give the answer in the following format if it is INVALID: \n"
        "Verdict: INVALID \n"
        "Rationale: briefly state why it is invalid. \n"

                
    )
    return judge_prompt



def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    r"""
    Parses the judge model's raw text completion into (score, rationale).
    
    Rules:
      1. If a line matches /verdict\s*:\s*\**\s*(VALID|INVALID)\b/i, use it.
         If several exist, use the first.
      2. Otherwise, if standalone word INVALID or phrase 'NOT VALID' appears
         (case-insensitive, word boundaries) -> -1.0.
      3. Otherwise, if standalone word VALID appears -> +1.0.
      4. Otherwise -> -1.0 (unparseable defaults to INVALID).
      
    Warning: A naive '"VALID" in text' check is wrong because "INVALID" contains "VALID".
    Rationale: Text after 'Rationale:' up to the first sentence end (".", "!", or "?"
    followed by whitespace or end of string, so decimals like "2.5" are not split),
    or a newline, whichever comes first, stripped. If absent, returns empty string "".
    """
    # TODO: Implement regex parsing for judge verdict and rationale
    VERDICT_RE   = r"verdict\s*:\s*\**\s*(VALID|INVALID)\b"   # re.IGNORECASE, re.search (first match wins)
    INVALID_RE   = r"\b(?:INVALID|NOT\s+VALID)\b"             # re.IGNORECASE
    VALID_RE     = r"\bVALID\b"                               # re.IGNORECASE
    RATIONALE_RE = r"rationale\s*:[ \t]*(.*?(?:[.!?](?=\s|$)|(?=\n)|$))"
    try:
            
        if not isinstance(judge_output, str):
            return -1.0, ""
        text = judge_output
    
        rm = re.search(RATIONALE_RE, text, flags=re.IGNORECASE)
        rationale = rm.group(1).strip() if rm else ""
    
        vm = re.search(VERDICT_RE, text, flags=re.IGNORECASE)      
        if vm:
            return (1.0 if vm.group(1).upper() == "VALID" else -1.0), rationale
        if re.search(INVALID_RE, text, flags=re.IGNORECASE):       
            return -1.0, rationale
        if re.search(VALID_RE, text, flags=re.IGNORECASE):         
            return 1.0, rationale
        return -1.0, rationale       
    except Exception:
        return -1.0, ""
        


        
        




def score_path(
    judge_model: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    candidate: dict[str, Any] | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Sequentially scores each step in candidate["steps"] using judge_fn.
    Step t is judged with preceding_steps = steps[:t].
    
    The student's code MUST call build_prm_judge_prompt to build the string passed to judge_fn,
    and MUST call parse_judge_decision on the returned text.
    
    Returns:
      step_scores, rationales, judge_calls, judge_tokens
    """
    # TODO: Implement sequential step scoring under local validity
    
    if judge_fn is None:
        judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)

    steps = list((candidate or {}).get("steps", []))
    step_scores, rationales = [], []
    judge_tokens = 0

    for t, step in enumerate(steps):
        prompt = build_prm_judge_prompt(question, steps[:t], step)
        out = judge_fn(prompt)
        score, rationale = parse_judge_decision(out.get("text", ""))
        step_scores.append(score)
        rationales.append(rationale)
        judge_tokens += int(out.get("num_tokens", 0))

    return {
        "step_scores": step_scores,
        "rationales": rationales,
        "judge_calls": len(steps),
        "judge_tokens": judge_tokens,
    }
