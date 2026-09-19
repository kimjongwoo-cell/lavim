"""Dump the actual NOVA keep mask (+ pixels, features) per Reasoner prefill. Chains tmp_hj/nova_v2_hooks (which chains
lpvh_ntrs_hooks). Enabled by VLMAS_NOVA_DUMP_DIR; no existing file is modified."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.join(os.path.dirname(_HERE), "nova_v2_hooks")
_f = os.path.join(_V2, "sitecustomize.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "sitecustomize_nova_v2"})

_DIR = os.environ.get("VLMAS_NOVA_DUMP_DIR", "")
if _DIR:
    import importlib.abc
    import importlib.machinery

    class _DumpFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "memory.nova_select":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                orig = module.select_vision_features

                def wrapped(bb, pixel_values, grid_thw):
                    import torch
                    kept, keep = orig(bb, pixel_values, grid_thw)
                    os.makedirs(_DIR, exist_ok=True)
                    n = len([x for x in os.listdir(_DIR) if x.endswith(".pt")])
                    pv = getattr(bb, "_secld_pixel_values_fp32", None)
                    if pv is None or tuple(pv.shape) != tuple(pixel_values.shape):
                        pv = pixel_values
                    torch.save({"keep": keep.detach().cpu(), "grid_thw": torch.as_tensor(grid_thw).cpu(),
                                "pixel_values": pv.detach().to(torch.float16).cpu(),
                                "mags": list(getattr(bb, "_prune_image_magnifications", ()) or ()),
                                "nova_v2": os.environ.get("VLMAS_NOVA_V2", "")},
                               os.path.join(_DIR, f"{n:03d}.pt"))
                    print(f"[NOVADUMP] saved {n:03d}.pt kept={int(keep.sum())}/{keep.numel()}", flush=True)
                    return kept, keep

                module.select_vision_features = wrapped

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _DumpFinder())
