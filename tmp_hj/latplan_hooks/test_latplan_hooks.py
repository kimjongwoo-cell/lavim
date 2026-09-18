"""usage: python tmp_hj/latplan_hooks/test_latplan_hooks.py  (spawns subprocesses with/without the flag)."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
PROBE = r'''
import json
from vision_text_mas.latent_onepass import fixed_patch_plan, LatentRoleClient
import vision_text_mas.onepass_navigation_nav4 as n4
plan = fixed_patch_plan("What organ type is shown in this histopathology image?", patch_budget=12)
p = n4._prompt(plan, x5_count=10, x20_count=20)
class B:
    called = 0
    def decode_on_clone(self, *a, **k):
        B.called += 1
        return 'x", "detail_target": "y"}'
c = LatentRoleClient.__new__(LatentRoleClient); c._backend = B(); c._state = None
r = c.decode_plan_targets("q")
print("JSON" + json.dumps({"prompt": p, "ret": r, "called": B.called}))
'''
def run(flag):
    env = dict(os.environ); env.pop("VLMAS_PLAN_LATENT_ONLY", None); env.pop("VLMAS_NAV3_X5_CLAMP", None); env.pop("VLMAS_C1_META_BRIDGE", None)
    if flag: env["VLMAS_PLAN_LATENT_ONLY"] = "1"
    env["PYTHONPATH"] = f"{HERE}:{REPO}"
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, cwd=REPO, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("JSON")]
    assert line, out.stderr[-2000:]
    return json.loads(line[0][4:]), out.stdout
P = F = 0
def check(n, c, x=""):
    global P, F
    P += bool(c); F += not c; print(("  ok   " if c else "  FAIL ") + n, x if not c else "")
env0 = dict(os.environ); env0.pop("VLMAS_PLAN_LATENT_ONLY", None)
env0["PYTHONPATH"] = REPO
ref = subprocess.run([sys.executable, "-c", PROBE], env=env0, cwd=REPO, capture_output=True, text=True)
ref = json.loads([l for l in ref.stdout.splitlines() if l.startswith("JSON")][0][4:])
off, so = run(False)
on, s1 = run(True)
check("flag off: prompt identical to no-hook run", off["prompt"] == ref["prompt"])
check("flag off: decode_plan_targets decodes (backend called once)", off["called"] == 1 and off["ret"] == ["x", "y"] and ref["called"] == 1)
check("flag off: no [LATPLAN] marker", "[LATPLAN]" not in so)
check("flag on: decode_plan_targets returns None, backend never called", on["ret"] is None and on["called"] == 0)
ref_lines = ref["prompt"].splitlines(); on_lines = on["prompt"].splitlines()
removed = [l for l in ref_lines if l not in on_lines]
check("flag on: exactly the two target lines removed, rest identical and ordered",
      removed == [l for l in ref_lines if l.startswith(("x5 overview target:", "x20 detail target:"))] and len(removed) == 2
      and on_lines == [l for l in ref_lines if l not in removed], removed)
check("flag on: latent-carried plan line and question focus kept", any("carried in latent state" in l for l in on_lines) and any(l.startswith("Question focus:") for l in on_lines))
check("flag on: both [LATPLAN] markers printed", s1.count("[LATPLAN] patched") == 2)
print(f"\n{P} passed, {F} failed"); sys.exit(1 if F else 0)
