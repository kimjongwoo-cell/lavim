"""Import hook for the GLVR diagnostic (tmp_hj/glvr).  Original files untouched.

Chains the latplan hooks first (which themselves chain lpvh_ntrs_hooks), then, when
VLMAS_GLVR is set, patches two modules:

  memory.rnlcr.apply_reasoner  -> install the replay capture while the grounding
                                  replay runs, so each grounded latent's conditional
                                  visual read u_{V,k} is recorded without an extra forward.
  memory.alsi_diag.run         -> run glvr_diag.run at the Answerer boundary
                                  (provenance masses, trajectory stats, lambda sweep).

VLMAS_GLVR_SKIP_ALSI=1 skips the original ALSI measurement afterwards.
"""
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- chain the latplan hooks (Planner text path removal) -------------------------------
# VLMAS_GLVR_CHAIN=nova_rho_latplan_hooks stacks GLVR on NOVA rho + latplan.
_CHAIN = os.environ.get("VLMAS_GLVR_CHAIN", "").strip() or "latplan_hooks"
for _sibling in (_CHAIN, "lpvh_ntrs_hooks"):
    _path = os.path.join(os.path.dirname(_HERE), _sibling, "sitecustomize.py")
    if os.path.exists(_path):
        try:
            _spec = importlib.util.spec_from_file_location(f"_glvr_chain_{_sibling}", _path)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
        except Exception as _exc:  # noqa: BLE001
            print(f"[GLVR] chain {_sibling} FAILED: {_exc!r}", flush=True)
        break

if os.environ.get("VLMAS_GLVR", "").strip():
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    def _patch_rnlcr(module):
        original = module.apply_reasoner

        def apply_reasoner(engine, result, **kwargs):
            import traceback
            try:
                import glvr_diag as GD
                bb = engine._backbone
                m = int(kwargs.get("m") or 0)
                rec = GD.on_reasoner(engine, result, m=m,
                                     case_index=int(kwargs.get("case_index", -1)))
                vis = (rec or {}).get("vis")
                if vis is not None and vis.numel():
                    with GD.capture_replay(bb, m, vis, int(result.get("past_len") or 0)) as cap:
                        out = original(engine, result, **kwargs)
                    GD.finish_reasoner(bb, cap)
                    return out
            except Exception as exc:  # noqa: BLE001 - never break the case
                print(f"[GLVR] reasoner capture FAILED: {exc!r}", flush=True)
                traceback.print_exc()
            return original(engine, result, **kwargs)

        module.apply_reasoner = apply_reasoner
        module._glvr_patched = True
        print("[GLVR] patched memory.rnlcr.apply_reasoner (replay capture)", flush=True)

    def _patch_alsi(module):
        original_run = module.run

        def run(engine, **kwargs):
            import traceback
            try:
                importlib.import_module("glvr_diag").run(engine, **kwargs)
            except Exception as exc:  # noqa: BLE001
                print(f"[GLVR] FAILED case {kwargs.get('case_index')}: {exc!r}", flush=True)
                traceback.print_exc()
            if os.environ.get("VLMAS_GLVR_SKIP_ALSI", "") == "1":
                return None
            return original_run(engine, **kwargs)

        module.run = run
        module._glvr_patched = True
        print("[GLVR] patched memory.alsi_diag.run (terminal-row diagnostic)", flush=True)

    def _patch_rnlcr_terminal(module):
        import contextlib
        original_terminal = module.terminal

        @contextlib.contextmanager
        def terminal(engine, cache, position_cursor, **kwargs):
            GD = importlib.import_module("glvr_diag")
            # In ground mode rnlcr.terminal only measures (its hooks return the output
            # untouched), yet it recomputes attention on every layer and decode step.
            # Yielding None makes the engine skip its [RNLCR] summary line.
            measure_only = not (module.redistributes() or module.readout() or module.fulr_mode())
            skip = measure_only and os.environ.get("VLMAS_GLVR_RNLCR_MEASURE", "0") != "1"
            outer = contextlib.nullcontext(None) if skip else \
                original_terminal(engine, cache, position_cursor, **kwargs)
            with outer as stats, \
                    GD.apply_terminal(engine, cache, system_prompt=kwargs.get("system_prompt"),
                                      user_prompt=kwargs.get("user_prompt"),
                                      json_prefix=kwargs.get("json_prefix"),
                                      case_index=int(kwargs.get("case_index", -1))):
                yield stats

        module.terminal = terminal
        print(f"[GLVR] patched memory.rnlcr.terminal (apply lambda={os.environ.get('VLMAS_GLVR_APPLY')})", flush=True)

    def _patch_formation(module):
        """F2 (Order 88 Phase A): per-latent disjoint visual support instead of shared P_Q/N_Q.

        pos_neg keeps the relevance scores and visual states it was handed; ground_stage then
        builds one target pair per latent from them. Steps, lr and loss shape stay F1's.
        """
        import formation as FM

        original_pos_neg = module.pos_neg
        original_ground = module.ground_stage

        def pos_neg(score, V, k):
            module._glvr_form_stash = {"score": score.detach(), "V": V.detach(), "k": int(k)}
            return original_pos_neg(score, V, k)

        def ground_stage(Z0, P, N, steps, lr_rel):
            st = getattr(module, "_glvr_form_stash", None)
            if not FM.enabled() or not st:
                return original_ground(Z0, P, N, steps, lr_rel)
            try:
                m = int(Z0.shape[0])
                Pk, Nk, top, _bot = FM.per_latent_targets(st["score"], st["V"], m, st["k"])
                Z, losses = FM.ground_stage_per_latent(Z0, Pk.to(Z0.device), Nk.to(Z0.device),
                                                       steps, lr_rel)
                print(f"[GLVR-F2] per-latent targets m={m} k={top.numel() // max(1, m)} "
                      f"support_overlap={FM.support_overlap(top, m):.3f} "
                      f"loss {losses[0]:.4g}->{losses[-1]:.4g}", flush=True)
                return Z, losses
            except Exception as exc:  # noqa: BLE001 - fall back to F1 rather than break the case
                print(f"[GLVR-F2] FAILED, using shared targets: {exc!r}", flush=True)
                return original_ground(Z0, P, N, steps, lr_rel)

        module.pos_neg = pos_neg
        module.ground_stage = ground_stage
        print(f"[GLVR] patched memory.rnlcr grounding (form={os.environ.get('VLMAS_GLVR_FORM')})",
              flush=True)

    def _patch_rnlcr_all(module):
        _patch_rnlcr(module)
        if os.environ.get("VLMAS_GLVR_FORM", "").strip().lower() == "f2":
            _patch_formation(module)
        try:
            if float(os.environ.get("VLMAS_GLVR_APPLY", "0") or 0):
                _patch_rnlcr_terminal(module)
        except ValueError:
            print("[GLVR] bad VLMAS_GLVR_APPLY, apply off", flush=True)

    def _patch_engine_coverage(module):
        """Coverage-driven Reasoner latent steps (coverage.py); VLMAS_GLVR_COV=beta > 0."""
        import glvr_coverage as CV
        cls = module.LatentQwenEngine
        original_append = cls.append

        def append(self, *args, **kwargs):
            b = CV.beta()
            if kwargs.get("stage") != "reasoner" or b == 0.0:
                return original_append(self, *args, **kwargs)
            bb = self._backbone
            with CV.install(bb, None, b) as st:
                bb._glvr_cov = st
                try:
                    out = original_append(self, *args, **kwargs)
                finally:
                    bb._glvr_cov = None
            rec = st.to_glvr(int(getattr(self, "_rpath_case_index", -1) or 0))
            if rec is not None:
                bb._glvr = rec
            print(f"[GLVR-COV] case {getattr(self, '_rpath_case_index', -1)} beta={b} "
                  f"steps={len(st.lat_cols)} n_vis={0 if st.vis is None else int(st.vis.numel())} "
                  f"recorded={rec is not None} note={st.note}", flush=True)
            return out

        cls.append = append
        print(f"[GLVR] patched LatentQwenEngine.append (coverage beta={CV.beta()})", flush=True)

    def _patch_backbone_coverage(module):
        """Hand the Reasoner's live visual columns to the coverage state at each latent step."""
        cls = module.Qwen3VLBackbone
        original_cut = cls._visual_cut_ctx

        def _visual_cut_ctx(self, cur_vis):
            st = getattr(self, "_glvr_cov", None)
            if st is not None and st.vis is None and isinstance(cur_vis, __import__("torch").Tensor):
                st.vis = cur_vis.detach().clone()
            return original_cut(self, cur_vis)

        cls._visual_cut_ctx = _visual_cut_ctx
        print("[GLVR] patched Qwen3VLBackbone._visual_cut_ctx (coverage vis hand-off)", flush=True)

    _TARGETS = {"memory.rnlcr": _patch_rnlcr_all, "memory.alsi_diag": _patch_alsi}
    try:
        if float(os.environ.get("VLMAS_GLVR_COV", "0") or 0):
            _TARGETS["vision_text_mas.latent_qwen_engine"] = _patch_engine_coverage
            _TARGETS["backbone.qwen3vl"] = _patch_backbone_coverage
    except ValueError:
        print("[GLVR] bad VLMAS_GLVR_COV, coverage off", flush=True)

    class _GlvrFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            patch = _TARGETS.get(fullname)
            if patch is None:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            original_exec = spec.loader.exec_module

            def exec_module(module):
                original_exec(module)
                try:
                    patch(module)
                except Exception as exc:  # noqa: BLE001
                    print(f"[GLVR] patch {fullname} FAILED: {exc!r}", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _GlvrFinder())
