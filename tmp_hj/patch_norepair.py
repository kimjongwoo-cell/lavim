"""0915: remove the digit-forced terminal repair (user decision).

Default = removed: an empty terminal answer is recorded as NO_ANSWER_SENTINEL and
scores as wrong; no second decode, no fabricated choice. VLMAS_TERMINAL_DIGIT_REPAIR=1
restores the 0806 behaviour for reproducing old runs.
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
p = os.path.join(root, "vision_text_mas/latent_answerer.py")
s = open(p).read()

old = '''MAX_TERMINAL_REPAIRS: Final = int(os.getenv("VLMAS_MAX_TERMINAL_REPAIRS", "3"))
'''
new = '''MAX_TERMINAL_REPAIRS: Final = int(os.getenv("VLMAS_MAX_TERMINAL_REPAIRS", "3"))
# 0915: the 0806 "empty answer -> re-decode with a forced single digit" repair is
# REMOVED by default (it fabricated choice 1/2 and could never pick choice >= 10).
# An empty terminal answer is stored as this sentinel and scores as wrong.
# VLMAS_TERMINAL_DIGIT_REPAIR=1 restores the old behaviour for reproduction only.
NO_ANSWER_SENTINEL: Final = "<no-answer>"


def _digit_repair_enabled() -> bool:
    return os.environ.get("VLMAS_TERMINAL_DIGIT_REPAIR", "0").strip() == "1"
'''
assert s.count(old) == 1, ("anchor const", s.count(old))
s = s.replace(old, new)

old = '''        answer_text = _normalize_choice(answer_text, effective_choices)
        if not answer_text:
            if MAX_TERMINAL_REPAIRS == 0:
                raise PipelineFailure(
                    code=FailureCode.ANSWER_CONTRACT,
                    stage="answerer",
                    detail="structured Judger returned an empty answer",
                )
            repair_prompt = latent_answerer_prompt('''
new = '''        answer_text = _normalize_choice(answer_text, effective_choices)
        if not answer_text and not _digit_repair_enabled():
            # 0915: no fabricated re-decode. Keep the raw so the empty readout is
            # inspectable, and let the case score as wrong.
            print(
                f"[Answerer] EMPTY terminal answer (raw={raw[:80]!r}); "
                "digit repair disabled -> recorded as no-answer",
                flush=True,
            )
            answer_text = NO_ANSWER_SENTINEL
        if not answer_text:
            if MAX_TERMINAL_REPAIRS == 0:
                raise PipelineFailure(
                    code=FailureCode.ANSWER_CONTRACT,
                    stage="answerer",
                    detail="structured Judger returned an empty answer",
                )
            repair_prompt = latent_answerer_prompt('''
assert s.count(old) == 1, ("anchor repair", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("answerer: digit repair disabled by default")
