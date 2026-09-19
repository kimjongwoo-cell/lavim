import hashlib, shutil
P = "/home/users/whddn12316/wsi_latent_0915_decode_hj/tests_hj_test_rnlcr.py"
src = open(P).read()
assert hashlib.md5(src.encode()).hexdigest().startswith("27f06672")
shutil.copy(P, P + ".bak_0917_fast2")
anchor = '''    check("fulr: no V* bank -> skip", "skip" in st_sk and "bank" in st_sk["skip"])
'''
assert src.count(anchor) == 1
add = '''    # FAST=2 hook-free fused sparse path == hook reference (same sdpa backend for both), outputs and stats
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _AAF
    impl0 = lm.config._attn_implementation
    lm.config._attn_implementation = "sdpa"
    KEYS2 = ("a_s", "ms", "rR", "rperp", "cos_rot", "delta_rel", "dv_rel", "fal_mean", "falv_mean", "o")
    try:
        for rows_ in ("gen", "all"):
            bb._rnlcr_R = R
            os.environ.pop("VLMAS_RNLCR_FAST", None)
            st_ref2, out_ref2, _ = answer(copy.deepcopy(base_fu), "fulr", rows=rows_)
            bb._rnlcr_R = R
            n_hooks = sum(len(l.self_attn._forward_hooks) for l in lm.layers)
            local_before = dict(_AAF._local_mapping)
            os.environ["VLMAS_RNLCR_FAST"] = "2"
            st_f2, out_f2, _ = answer(copy.deepcopy(base_fu), "fulr", rows=rows_)
            os.environ.pop("VLMAS_RNLCR_FAST", None)
            dmax = max(float((a - b).abs().max()) for a, b in zip(out_ref2, out_f2))
            check(f"fast2 rows={rows_}: outputs == hook reference", dmax < 1e-5, dmax)
            bad = {k: (st_ref2.get(k), st_f2.get(k)) for k in KEYS2 if st_ref2.get(k) is None or st_f2.get(k) is None or abs(st_ref2[k] - st_f2[k]) > 1e-4 * max(1.0, abs(st_ref2[k]))}
            check(f"fast2 rows={rows_}: stats == hook reference", not bad, bad)
            lay = all(abs(a - b) < 1e-4 for a, b in zip(st_ref2["cos_rot_by_layer"] + st_ref2["fal_by_layer"] + st_ref2["a_s_by_layer"] + st_ref2["ms_by_layer"],
                                                           st_f2["cos_rot_by_layer"] + st_f2["fal_by_layer"] + st_f2["a_s_by_layer"] + st_f2["ms_by_layer"]))
            check(f"fast2 rows={rows_}: per-layer stats equal, calls/rows equal, no hooks installed, registry restored, fast flag 2",
                  lay and st_ref2["calls"] == st_f2["calls"] and st_ref2["rows"] == st_f2["rows"] and n_hooks == 0
                  and dict(_AAF._local_mapping) == local_before and st_f2.get("fast") == 2 and bool(_json.dumps(st_f2)))
        bb._rnlcr_R = R
        os.environ["VLMAS_RNLCR_FAST"] = "2"; os.environ["VLMAS_RNLCR_STATS"] = "none"
        st_n, out_n, _ = answer(copy.deepcopy(base_fu), "fulr", rows="gen")
        os.environ.pop("VLMAS_RNLCR_FAST", None); os.environ.pop("VLMAS_RNLCR_STATS", None)
        bb._rnlcr_R = R
        st_g, out_g, _ = answer(copy.deepcopy(base_fu), "fulr", rows="gen")
        check("fast2 stats=none: outputs unchanged vs hook reference", max(float((a - b).abs().max()) for a, b in zip(out_g, out_n)) < 1e-5)
    finally:
        lm.config._attn_implementation = impl0
    bb._rnlcr_R = R
    os.environ["VLMAS_RNLCR_FAST"] = "2"
    st_e, out_e, _ = answer(copy.deepcopy(base_fu), "fulr", rows="gen")
    os.environ.pop("VLMAS_RNLCR_FAST", None)
    check("fast2 on eager backend falls back to the fast hook path, outputs == reference", st_e.get("fast2_fallback") == "eager"
          and max(float((a - b).abs().max()) for a, b in zip(out_u, out_e)) < 1e-5)
''' + anchor
src = src.replace(anchor, add)
open(P, "w").write(src)
print("tests patched", hashlib.md5(src.encode()).hexdigest()[:8])
