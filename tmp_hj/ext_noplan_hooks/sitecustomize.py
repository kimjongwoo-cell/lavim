"""0919 PLIP -> Reasoner -> Answerer (no Planner, no Navigator). Original files and tmp_hj/ext_patches_hooks untouched.

Chains tmp_hj/ext_patches_hooks (PLIP top-K patches replace the nav4 Navigator; itself chains latplan_hooks). With
VLMAS_NO_PLANNER=1, vision_text_mas.latent_onepass_agents.LatentEvidencePlannerAgent.plan makes no model call and appends
nothing to the latent state: it only returns fixed_patch_plan(question) (the same deterministic metadata the real Planner
returns) with a placeholder RoleCall. The Reasoner therefore becomes the first role and starts the cache itself
(latent_qwen_engine: cache is None -> grounded_prefill_and_latent with the patch images).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_f = os.path.join(os.path.dirname(_HERE), "ext_patches_hooks", "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_ext_patches"})

if os.environ.get("VLMAS_NO_PLANNER", "") == "1":
    import importlib.abc
    import importlib.machinery

    def _noplan(self, *, question, thumbnail, missing_evidence, patch_budget=8):
        from vision_text_mas.contracts import RoleCall
        from vision_text_mas.latent_onepass import fixed_patch_plan
        from vision_text_mas.qwen_client import ParsedRoleCall
        plan = fixed_patch_plan(question, patch_budget=patch_budget)
        record = RoleCall(role="evidence_planner", prompt="[NOPLAN] Planner skipped (VLMAS_NO_PLANNER=1)",
                          image_labels=(), reasoning="", final_outputs=("skipped",), format_repairs=0,
                          physical_calls=1, elapsed_seconds=0.0)
        print("[NOPLAN] skipped Planner (no model call, nothing appended to the latent state)", flush=True)
        return ParsedRoleCall(value=plan, record=record)

    class _AgentsFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "vision_text_mas.latent_onepass_agents":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig = spec.loader.exec_module

            def exec_module(module):
                orig(module)
                module.LatentEvidencePlannerAgent.plan = _noplan
                print("[NOPLAN] patched LatentEvidencePlannerAgent.plan", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _AgentsFinder())
