"""
src/extraction.py - Step and answer extraction utilities.
"""
import re


def parse_steps(completion_text: str) -> list[str]:
    """
    Parses completion text and returns a list of individual step strings.
    Must tolerate markdown decoration around labels (e.g. '**Step 1:**', '### Step 2:')
    and strip leftover '*' or '#' from captured text.
    """
    if not completion_text or not isinstance(completion_text, str):
        return []

    # Pattern matching Step markers with optional markdown decoration (#, *, `, :, .)
    pattern = r"(?:^|\n)\s*(?:[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)(.*?)(?=(?:\n\s*[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)|(?:\n\s*[#*`\s]*Final Answer)|$)"
    matches = re.findall(pattern, completion_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = []
    for m in matches:
        text = m.strip().strip("*#`").strip()
        if text:
            cleaned.append(text)
    return cleaned


def extract_answer(completion_text: str) -> int | None:
    """
    Extracts the final numeric integer from the completion.

    Required Behaviors:
      - 'Final Answer: 18' -> 18
      - 'Final Answer: $1,234' -> 1234
      - 'Final Answer: 18.00' -> 18
      - 'Final Answer: 18.5' -> None (non-integer)
      - 'Final Answer: -5' -> -5
      - 'Final Answer: 18 dollars.' -> 18
      - '**Final Answer:** 18' -> 18
      - 'final answer 18' (case-insensitive, colon optional) -> 18
      - Multiple 'Final Answer' lines -> use the LAST one
      - No 'Final Answer' present -> None
      - Empty string or None-like garbage -> None, and never raise
    """
    # TODO: Implement robust regex answer extraction
        
    
    try:
        
        if not completion_text or not isinstance(completion_text, str):
            return None

        delimiters = list(re.finditer(r"final[ \t]*answer", completion_text, flags=re.IGNORECASE))
        if not delimiters:
            return None
        line = completion_text[delimiters[-1].end():].split("\n", 1)[0]

        #used claude to help with the following part 

        pattern = (
            r"(?:\s*is\b)?[\s:=*]*"
            r"\$?\s*(?P<sign>[-\u2212])?\s*\$?\s*"
            r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)"
            r"(?:\.(?P<frac>\d+))?"
            )
        m = re.match(pattern, line, flags=re.IGNORECASE)
        if not m:
            return None
        if m.group("frac") and m.group("frac").strip("0"):
            return None

        val = int(m.group("int").replace(",", ""))
        return -val if m.group("sign") else val
    
    except Exception:
        return None

      
