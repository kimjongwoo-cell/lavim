import hashlib, shutil
P = "/home/users/whddn12316/wsi_latent_0915_decode_hj/tests_hj_test_rnlcr.py"
src = open(P).read()
assert hashlib.md5(src.encode()).hexdigest().startswith("539c410e")
shutil.copy(P, P + ".bak_0917_fast")
anchor = '''    check("fulr rows=all: prefill rows included", st_ua["rows"] == 5 and st_ua["applied"])
'''
assert src.count(anchor) == 1
add = anchor + '''    # FAST path == reference path (outputs and every stat), rows=gen and rows=all
    import json as _json
    KEYS = ("a_s", "C_s", "o", "lat_nat", "lat_new", "lat_would", "msg_nat", "msg_new", "msg_would", "ms", "rR", "rperp", "cos_rot", "delta_rel", "dv_rel", "fal_mean", "falv_mean")
    for rows_ in ("gen", "all"):
        bb._rnlcr_R = R
        os.environ.pop("VLMAS_RNLCR_FAST", None)
        st_ref, out_ref, _ = answer(copy.deepcopy(base_fu), "fulr", rows=rows_)
        bb._rnlcr_R = R
        os.environ["VLMAS_RNLCR_FAST"] = "1"
        st_fa, out_fa, _ = answer(copy.deepcopy(base_fu), "fulr", rows=rows_)
        os.environ.pop("VLMAS_RNLCR_FAST", None)
        dmax = max(float((a - b).abs().max()) for a, b in zip(out_ref, out_fa))
        check(f"fast rows={rows_}: outputs == reference", dmax < 1e-5, dmax)
        bad = {k: (st_ref.get(k), st_fa.get(k)) for k in KEYS if st_ref.get(k) is None or st_fa.get(k) is None or abs(st_ref[k] - st_fa[k]) > 1e-4 * max(1.0, abs(st_ref[k]))}
        check(f"fast rows={rows_}: stats == reference", not bad, bad)
        lay = all(abs(a - b) < 1e-4 for a, b in zip(st_ref["cos_rot_by_layer"] + st_ref["fal_by_layer"] + st_ref["a_s_by_layer"], st_fa["cos_rot_by_layer"] + st_fa["fal_by_layer"] + st_fa["a_s_by_layer"]))
        check(f"fast rows={rows_}: per-layer stats == reference, calls/rows equal, fast flag, json-serialisable",
              lay and st_ref["calls"] == st_fa["calls"] and st_ref["rows"] == st_fa["rows"] and st_fa.get("fast") is True and "fast" not in st_ref
              and bool(_json.dumps(st_fa)))
'''
src = src.replace(anchor, add)
open(P, "w").write(src)
print("tests patched", hashlib.md5(src.encode()).hexdigest()[:8])
