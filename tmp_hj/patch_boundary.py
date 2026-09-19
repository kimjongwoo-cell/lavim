"""0915 LatentMAS-faithful Answerer boundary (env-gated, off = byte-identical).

VLMAS_ANSWERER_BOUNDARY=1  Insert ONE text-only user/assistant turn (no images, ZERO latent
                           steps) between the Reasoner's image-bearing turn and the Answerer,
                           the way 0717 always had a text-only Verifier turn before Diagnosis.
                           The answering agent itself still runs no latent steps (LatentMAS
                           judger semantics); total latent budget is unchanged.
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
p = os.path.join(root, "vision_text_mas/latent_onepass.py")
s = open(p).read()
old = '''        answerer_steps = int(_os.environ.get("VLMAS_ANSWERER_LATENT_STEPS", "0") or 0)
        if answerer_steps > 0 and getattr(self, "_answerer_latent_state_id", None) != id(self._state):
            self._state = self._backend.append_multimodal_latent(
                self._state,
                stage="answerer_latent",
                images=(),
                system=system_prompt,
                prompt=("Before answering, deliberate over the visual evidence "
                        "already in context for this question: " + user_prompt[:600]),
                latent_steps=answerer_steps,
            )
            self._answerer_latent_state_id = id(self._state)
            print(f"[AnswererLatent] {answerer_steps} latent steps appended "
                  f"(cache {self._state.cache_length})", flush=True)
'''
new = '''        answerer_steps = int(_os.environ.get("VLMAS_ANSWERER_LATENT_STEPS", "0") or 0)
        # 0915 VLMAS_ANSWERER_BOUNDARY=1: text-only boundary turn with ZERO latent steps
        # (LatentMAS-faithful: the answering agent runs no latent; this only restores the
        # 0717 structure "image turn -> text-only turn -> final decode").
        boundary_only = (
            answerer_steps <= 0
            and _os.environ.get("VLMAS_ANSWERER_BOUNDARY", "").strip() == "1"
        )
        if (answerer_steps > 0 or boundary_only) and getattr(
            self, "_answerer_latent_state_id", None
        ) != id(self._state):
            self._state = self._backend.append_multimodal_latent(
                self._state,
                stage="answerer_latent",
                images=(),
                system=system_prompt,
                prompt=(
                    ("The evidence review for this question is complete; the final "
                     "answer follows in the next turn. Question: " + user_prompt[:300])
                    if boundary_only
                    else ("Before answering, deliberate over the visual evidence "
                          "already in context for this question: " + user_prompt[:600])
                ),
                latent_steps=max(0, answerer_steps),
            )
            self._answerer_latent_state_id = id(self._state)
            print(f"[AnswererLatent] {max(0, answerer_steps)} latent steps appended "
                  f"(boundary_only={boundary_only}, cache {self._state.cache_length})",
                  flush=True)
'''
assert s.count(old) == 1, ("answerer latent anchor", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("onepass: VLMAS_ANSWERER_BOUNDARY added")
