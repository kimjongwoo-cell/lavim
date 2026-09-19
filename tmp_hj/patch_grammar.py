"""0915: grammar-constrained terminal decode (port of 0902 grammar_json), env-gated.

VLMAS_TERMINAL_GRAMMAR=1  Answerer decode is masked token-by-token by an xgrammar JSON schema
                          {answer: enum(choices), rationale, confidence}. The teacher-forced
                          opener is dropped by the engine (the grammar generates the object from
                          "{"), greedy is unchanged, so a case whose unconstrained greedy path is
                          already grammar-valid decodes identically; an invalid path is forced onto
                          the nearest valid continuation -> the output is always one of the choices.
Off = byte-identical.
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."

# ------------------------------------------------------------- onepass: thread json_schema
p = os.path.join(root, "vision_text_mas/latent_onepass.py")
s = open(p).read()
old = '''        final_tokens: int,
        terminal_prefix: str | None = "<answer>",
    ) -> RoleCall:
        """Generate the final answer directly from the accumulated latent cache."""
'''
new = '''        final_tokens: int,
        terminal_prefix: str | None = "<answer>",
        json_schema: str | None = None,
    ) -> RoleCall:
        """Generate the final answer directly from the accumulated latent cache."""
'''
assert s.count(old) == 1, ("onepass sig", s.count(old))
s = s.replace(old, new)
old = '''            max_new_tokens=final_tokens,
            json_prefix=terminal_prefix,
        )
        return RoleCall(
            role="answerer",
'''
new = '''            max_new_tokens=final_tokens,
            json_prefix=terminal_prefix,
            json_schema=json_schema,
        )
        return RoleCall(
            role="answerer",
'''
assert s.count(old) == 1, ("onepass call", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("onepass: json_schema threaded")

# ------------------------------------------------------------- answerer: build schema when enabled
p = os.path.join(root, "vision_text_mas/latent_answerer.py")
s = open(p).read()
old = '''        terminal_prefix: str | None,
    ) -> RoleCall: ...
'''
new = '''        terminal_prefix: str | None,
        json_schema: str | None = None,
    ) -> RoleCall: ...
'''
assert s.count(old) == 1, ("protocol sig", s.count(old))
s = s.replace(old, new)
old = '''            final_tokens=self._final_tokens,
            terminal_prefix=self._terminal_prefix(),
        )
        raw = record.final_outputs[0].strip()
'''
new = '''            final_tokens=self._final_tokens,
            terminal_prefix=self._terminal_prefix(),
            # 0915 VLMAS_TERMINAL_GRAMMAR=1: xgrammar-constrained readout (0902 grammar_json
            # port). answer is an enum of the exact choice texts, so garbage is impossible.
            json_schema=(
                latent_answerer_json_schema(answer_options=tuple(effective_choices or ()))
                if os.environ.get("VLMAS_TERMINAL_GRAMMAR", "").strip() == "1"
                else None
            ),
        )
        raw = record.final_outputs[0].strip()
'''
assert s.count(old) == 1, ("answerer call", s.count(old))
s = s.replace(old, new)
old = '''from vision_text_mas.prompts import canonical_open_answer_options
'''
new = '''from vision_text_mas.prompts import canonical_open_answer_options
from vision_text_mas.answerer_schema import latent_answerer_json_schema
'''
assert s.count(old) == 1, ("answerer import", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("answerer: VLMAS_TERMINAL_GRAMMAR schema pass-through added")

# ------------------------------------------------------------- engine: accept the terminal gate
p = os.path.join(root, "vision_text_mas/latent_qwen_engine.py")
s = open(p).read()
old = '''            and os.environ.get("VLMAS_NAV_GRAMMAR") == "1"
'''
new = '''            and (
                os.environ.get("VLMAS_NAV_GRAMMAR") == "1"
                or os.environ.get("VLMAS_TERMINAL_GRAMMAR", "").strip() == "1"
            )
'''
assert s.count(old) == 1, ("engine gate", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("engine: VLMAS_TERMINAL_GRAMMAR gate added")
