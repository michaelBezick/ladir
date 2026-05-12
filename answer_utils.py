import math
import re
from fractions import Fraction
from typing import Any, Iterable, Optional


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def _strip_latex(text: str) -> str:
    replacements = {
        "$": "",
        "\\$": "",
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\!": "",
        "\\ ": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    return text


def extract_final_answer(text: Any) -> Optional[str]:
    """Extract the most likely final math answer from decoded reasoning text."""
    if text is None:
        return None
    text = _strip_latex(str(text)).strip()
    if not text:
        return None

    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()

    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()

    answer_patterns = [
        r"(?:final answer|answer|therefore|so)\s*(?:is|=|:)?\s*([^\n\.]+)",
        r"(?:=)\s*([^\n=]+)$",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            candidate = matches[-1].strip()
            number = _NUMBER_RE.findall(candidate)
            return number[-1] if number else candidate

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].strip()
    return text.splitlines()[-1].strip()


def _to_number(value: str) -> Optional[float]:
    value = value.strip().replace(",", "")
    if not value:
        return None
    value = value.rstrip(".")
    try:
        if "/" in value and not any(ch in value for ch in "()[]"):
            return float(Fraction(value))
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def normalize_answer(answer: Any) -> Optional[str]:
    if answer is None:
        return None
    answer = extract_final_answer(answer)
    if answer is None:
        return None
    answer = _strip_latex(str(answer)).strip().lower()
    answer = answer.replace(",", "")
    answer = re.sub(r"\s+", " ", answer)
    answer = answer.strip(" .,:;")

    number = _to_number(answer)
    if number is not None and math.isfinite(number):
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return f"{number:.10g}"
    return answer


def answers_match(prediction: Any, targets: Any, atol: float = 1e-6) -> bool:
    pred = normalize_answer(prediction)
    if pred is None:
        return False

    if isinstance(targets, str) or not isinstance(targets, Iterable):
        target_iter = [targets]
    else:
        target_iter = targets

    pred_num = _to_number(pred)
    for target in target_iter:
        gold = normalize_answer(target)
        if gold is None:
            continue
        if pred == gold:
            return True
        gold_num = _to_number(gold)
        if pred_num is not None and gold_num is not None and abs(pred_num - gold_num) <= atol:
            return True
    return False
