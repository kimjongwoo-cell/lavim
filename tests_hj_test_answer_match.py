#!/usr/bin/env python3
"""answer_match / rescore 유닛 + 0912 대시보드 Base 재현.

1) exact 가 0912/live_dashboard.py normalized() 와 같은 문자열을 내는지
2) 실제로 exact 가 놓친 4건이 어느 level 에서 살아나는지, 틀린 답은 어느 level 에서도 안 살아나는지
3) 보기 충돌이면 exact 로 되돌아가는지
4) rescore.py exact 가 live_dashboard 의 Base 5행(n·정답수·점수·초)과 같은지 (0912/runs 가 있을 때)

usage: python3 tests_hj_test_answer_match.py
"""
import importlib.util
import sys
from pathlib import Path

TREE = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # live_dashboard 의 dataclass 가 모듈 등록을 요구한다
    spec.loader.exec_module(module)
    return module


am = load("answer_match", TREE / "eval" / "answer_match.py")
rescore = load("rescore", TREE / "scripts" / "rescore.py")
live = load("live", TREE / "0912" / "live_dashboard.py")

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


def hit(pred, gold, level, choices=None):
    return am.judge(pred, gold, choices if choices is not None else [gold, "zzz other"], level)[0]


print("== exact == live_dashboard.normalized")
samples = ["Chronic gastritis.", "  Skin ", "A\tB\nC", "ＳＫＩＮ", "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma",
           "pT2 pNX pMX", "ER+", "", "ß", "İstanbul", "a b"]
for s in samples:
    check(f"exact {s!r}", am.normalize(s) == live.normalized(s), (am.normalize(s), live.normalized(s)))

print("== 실제 4건")
check("gastritis. : exact 틀림", not hit("Chronic gastritis", "Chronic gastritis.", "exact"))
check("gastritis. : norm 맞음", hit("Chronic gastritis", "Chronic gastritis.", "norm"))
check(",, NOS : norm 맞음", hit("Adenocarcinoma,, NOS", "Adenocarcinoma, NOS", "norm"))
check("pNXpMX : norm 틀림", not hit("pT2 pNXpMX", "pT2 pNX pMX", "norm"))
check("pNXpMX : nospace 맞음", hit("pT2 pNXpMX", "pT2 pNX pMX", "nospace"))
check("MultipleSmall : norm 틀림", not hit("MultipleSmall foci", "Multiple small foci", "norm"))
check("MultipleSmall : nospace 맞음", hit("MultipleSmall foci", "Multiple small foci", "nospace"))

print("== 틀린 답은 어느 level 에서도 틀림")
for pred in ["CHOICE", "text", "{", "", "B", "Skin Skin"]:
    for level in am.LEVELS:
        check(f"{pred!r} @{level}", not hit(pred, "Skin", level))
check("ER+ vs ER- @nospace", not hit("ER+", "ER-", "nospace"))
check("B-cell vs B cell @norm (하이픈 유지)", not hit("B-cell", "B cell", "norm"))
check("Grade vs Grade A @nospace (관사 유지)", not hit("Grade", "Grade A", "nospace"))
check("2.5 vs 25 @nospace (가운데 소수점 유지)", not hit("25", "2.5", "nospace"))

print("== NFKC·대소문자")
check("전각 SKIN @exact 틀림", not hit("ＳＫＩＮ", "Skin", "exact"))
check("전각 SKIN @norm 맞음", hit("ＳＫＩＮ", "Skin", "norm"))
check("대소문자 @exact 맞음", hit("skin", "SKIN", "exact"))

print("== 충돌")
choices = ["Grade 2.", "Grade 2", "Grade 3"]
check("collisions 탐지", am.collisions(choices, "norm") == [[0, 1]], am.collisions(choices, "norm"))
check("exact 에선 충돌 없음", am.collisions(choices, "exact") == [])
ok, fell = am.judge("grade 2.", "Grade 2", choices, "norm")
check("정답이 충돌에 걸리면 exact 로 되돌아감", (ok, fell) == (False, True), (ok, fell))
ok, fell = am.judge("Grade 3.", "Grade 3", choices, "norm")
check("충돌 밖 정답은 norm 판정", (ok, fell) == (True, False), (ok, fell))
ok, fell = am.judge("Skin.", "Skin", None, "norm")
check("choices 없으면 exact 로 되돌아감", (ok, fell) == (False, True), (ok, fell))

print("== 단조성 (exact 맞음 → 전 level 맞음)")
for pred, gold in [("Skin", "skin"), ("  a  b ", "A B"), ("Chronic gastritis.", "chronic  gastritis.")]:
    check(f"{pred!r}", all(hit(pred, gold, lv) for lv in am.LEVELS))

try:
    am.normalize("x", "fuzzy")
    check("알 수 없는 level ValueError", False)
except ValueError:
    check("알 수 없는 level ValueError", True)

print("== 0912 대시보드 Base 재현 (rescore exact vs live_dashboard)")
if not (TREE / "0912" / "runs").is_dir():
    print("  skip (0912/runs 없음)")
else:
    selections = {
        "tcga_expert_vqa": (["0912/runs/nav2v2_base_expert_*"], []),
        "tcga_slidebench": (["0912/runs/tcga_slidebench_base_12patch_*"], []),
        "gtex": (["0912/runs/*nav2*"], ["prompt2"]),
        "tcga": (["0912/runs/*nav2*"], ["prompt2"]),
        "panda": (["0912/runs/*nav2*"], ["prompt2"]),
    }
    live_rows = {row.dataset: row for row in live.collect_uncached() if row.variant == "Base"}
    for dataset, (roots, exclude) in selections.items():
        dirs = rescore.run_dirs(rescore.resolve_roots(roots), "base", exclude)
        cases, timings = rescore.collect(dirs, {dataset})
        mine = rescore.score_dataset(dataset, cases.get(dataset, {}), rescore.load_records(dataset),
                                     timings.get(dataset))
        ref = live_rows.get(dataset)
        got = (mine["n"], mine["exact"]["correct"], mine["exact"]["score"], mine["sec"])
        want = (ref.completed, ref.correct, ref.score, ref.seconds_per_case) if ref else None
        check(f"{dataset} (n, correct, score, sec)", got == want, f"rescore={got} live={want}")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
