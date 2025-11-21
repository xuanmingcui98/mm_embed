import re, json, ast
from typing import Dict, Any

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

def _first_braced(s: str) -> str:
    """Return first balanced {...} chunk or the whole string."""
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return s[start:]  # best effort

def _clean(s: str) -> str:
    # strip stray backticks & normalize smart quotes
    s = s.strip().strip("`").replace("“","\"").replace("”","\"").replace("’","'").replace("‘","'")
    # remove // and /* */ comments
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)
    # remove trailing commas before ] or }
    s = re.sub(r",(\s*[\]}])", r"\1", s)
    return s.strip()

def parse_llm_json(text: str) -> Dict[str, Any]:
    """Parse a depth-1 dict from messy LLM output."""
    m = _FENCE.search(text)
    cand = m.group(1) if m else _first_braced(text)
    cand = _clean(cand)

    # 1) strict JSON
    try:
        obj = json.loads(cand)
    except Exception:
        # 2) tiny fallback: allow single quotes & true/false/null
        pyish = (cand
                 .replace(": null", ": None")
                 .replace(": true", ": True")
                 .replace(": false", ": False"))
        try:
            obj = ast.literal_eval(pyish)
        except Exception as e:
            raise ValueError(f"Could not parse JSON-like output:\n{e}\n---\n{cand[:400]}") from e

    if not isinstance(obj, dict):
        raise ValueError("Expected a top-level object (dict).")
    # Optional: enforce depth-1 (no nested dicts)
    for v in obj.values():
        if isinstance(v, dict):
            raise ValueError("Nested dicts found; expected depth-1.")
    return obj