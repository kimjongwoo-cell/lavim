"""MultiPathQA 답 일치 판정 — 세 level, 전부 "정리한 뒤 문자열이 같은가"만 본다.

  exact    " ".join(casefold().strip().split())
           0912/live_dashboard.py 의 normalized() 와 같은 동작이다.
  norm     NFKC, 같은 구두점 연속(",,")을 하나로, 앞뒤 구두점(. , ; :) 제거, 그 뒤 exact.
  nospace  norm 에서 공백을 전부 지운다.

지우지 않는 것: 하이픈·+/-·괄호·소수점 같은 가운데 기호("B-cell", "ER+", "2.5"),
관사(a/an/the). 유사도(quick_ratio 등)는 쓰지 않는다 — eval/metrics.py acc_of_seq 가
형식 실패 출력을 정답 처리한 원인이 그것이다.

충돌: 한 문항의 보기 둘이 같은 level 에서 같은 문자열이 되면 그 level 은 그 문항을 가를 수
없다. 정답 보기가 충돌에 걸리면 그 문항은 exact 로 판정한다.
단조성: exact 로 맞으면 모든 level 에서 맞다.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

LEVELS = ("exact", "norm", "nospace")
_EDGE = " .,;:"
_REPEAT = re.compile(r"([.,;:])\1+")


def normalize(text: object, level: str = "exact") -> str:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
    value = str(text)
    if level == "exact":
        return " ".join(value.casefold().strip().split())
    value = unicodedata.normalize("NFKC", value)
    value = " ".join(value.casefold().split())
    value = _REPEAT.sub(r"\1", value)
    value = " ".join(value.strip(_EDGE).split())
    if level == "nospace":
        value = "".join(value.split())
    return value


def collisions(choices: Sequence[object], level: str) -> list[list[int]]:
    """같은 문자열로 합쳐지는 보기 인덱스 묶음(크기 2 이상)."""
    groups: dict[str, list[int]] = {}
    for position, choice in enumerate(choices):
        groups.setdefault(normalize(choice, level), []).append(position)
    return [members for members in groups.values() if len(members) > 1]


def judge(
    prediction: object,
    gold: object,
    choices: Sequence[object] | None,
    level: str = "exact",
) -> tuple[bool, bool]:
    """(맞음, exact 로 되돌아감).

    choices 가 None 이면 충돌을 확인할 수 없으므로 exact 로 되돌아간다.
    """
    exact_hit = normalize(prediction) == normalize(gold)
    if level == "exact":
        return exact_hit, False
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
    if choices is None:
        return exact_hit, True
    target = normalize(gold, level)
    if sum(normalize(choice, level) == target for choice in choices) > 1:
        return exact_hit, True
    return exact_hit or normalize(prediction, level) == target, False
