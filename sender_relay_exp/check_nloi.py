"""Integrity checks on finished NLOI rows (no result interpretation)."""
import glob, json, os, sys, math
import torch
K = sys.argv[1]
for jf in sorted(glob.glob(f"{K}/nloi_nat_*.jsonl")):
    name = os.path.basename(jf)[5:-6]
    rows = [json.loads(l) for l in open(jf) if l.strip()]
    cases = [r["case"] for r in rows]
    print(f"\n== {name}: rows={len(rows)} cases={cases} dup={len(cases)-len(set(cases))}")
    # result.json predictions keyed by dataset order
    res = sorted(glob.glob(f"{K}/{name}/**/result.json", recursive=True))
    for r in rows:
        P = len(r["pages"])
        flags = []
        if r["n_vis"] != sum(r["pages"]): flags.append("n_vis!=sum(pages)")
        if len(set(r["pages"])) != 1: flags.append(f"pages={r['pages']}")
        if r["mags"] is None or len(r["mags"]) != P: flags.append("mags")
        if r["boxes"] is None or len(r["boxes"]) != P: flags.append("boxes")
        par = r.get("parents") or []
        # children inside parent box
        bad_child = 0
        for c, p in enumerate(par):
            if p >= 0:
                bc, bp = r["boxes"][c], r["boxes"][p]
                if not (bp[0] <= bc[0] and bc[0] + bc[2] <= bp[0] + bp[2] and bp[1] <= bc[1] and bc[1] + bc[3] <= bp[1] + bp[3]):
                    bad_child += 1
        if bad_child: flags.append(f"child_outside_parent={bad_child}")
        if len(r["true"]["pairs"]) != P * (P - 1) // 2: flags.append("npairs")
        if r["shuffled"] is None or r["shuffled_sizes"] != r["pages"]: flags.append("shuf_sizes")
        # ident argmax vs native greedy token at each probed row
        agree = [int(a) == int(r["gen_tokens"][t]) for a, t in zip(r["argmax_ident"], r["tok_rows"])]
        # recompute psi from dump for pair 0 row 0
        dp = f"{K}/dump_{name}/nloi_case{r['case']:03d}.pt"
        psi_ok = None
        if os.path.exists(dp):
            d = torch.load(dp)["h"]
            p0 = r["true"]["pairs"][0]; i, j = p0["i"], p0["j"]
            hf, hi, hj, hij = (d[k][0].double() for k in ("ident", f"true_s{i}", f"true_s{j}", f"true_p{i}_{j}"))
            psi = float(((hf - hij) - (hf - hi) - (hf - hj)).norm())
            psi_ok = abs(psi - p0["rows"][0]["psi"]) / max(psi, 1e-9) < 1e-3   # dump is float32
            nd = len(d)
            if nd != 3 + 2 * (P + P * (P - 1) // 2): flags.append(f"dump_entries={nd}")
        mass = [m[0] for m in r["mass_true"]] if r["mass_true"] else []
        print(f" c{r['case']:02d} P={P} mags={''.join('a' if m == 5 else 'b' for m in r['mags'])} rows={r['tok_rows']} "
              f"ident==native_tok {sum(agree)}/{len(agree)} drift/d_all={r['path_drift'][0]/r['zero_all']['d'][0]:.2f} "
              f"m_V={sum(mass):.4f} psi_dump_ok={psi_ok} sec={r['sec']} ans={r['gen_text'][:14]!r} {' '.join(flags)}")
    # engine answers vs NLOI regeneration
    preds = []
    for f in res:
        try:
            j = json.load(open(f)); preds.append(str(j.get("prediction", j.get("answer", j.get("pred", ""))))[:14])
        except Exception as e:
            preds.append(f"ERR {e}")
    print(f" result.json n={len(res)} preds={preds[:10]}")
