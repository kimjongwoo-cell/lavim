"""RFNM 24-question smoke vs the C1 NOVA rho full run on the same original indices (exact judge)."""
import glob, json, os, sys
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj/scripts")
import rescore as rs
W = "runs/nav4_nlqp_dev"
for tag, ds in (("evqa24", "tcga_expert_vqa"), ("sb24", "tcga_slidebench")):
    recs = rs.load_records(ds)
    sm = json.load(open(f"smoke/{tag}.json"))
    qmap = {}
    for i, r in enumerate(recs):
        qmap.setdefault((r["Question"], r["Answer"]), i)
    ref = {}
    for f in glob.glob(f"runs/nav4_rho_latplan/nova_rho_latplan/{ds}/gpu*_attempt_*/*/result.json"):
        r = json.load(open(f)); a = r["answer"]; ref[r["dataset_index"]] = str(a.get("answer") if isinstance(a, dict) else a)
    rows, logs = [], {}
    for l in open(f"{W}/{tag}/rfnm_rho/nlqp_shardsmoke.jsonl"):
        x = json.loads(l); logs[x["case"]] = x
    for f in sorted(glob.glob(f"{W}/{tag}/rfnm_rho/smoke_rfnm_rho/*/*/result.json")):
        name = os.path.basename(os.path.dirname(f)); pos = int(name.split("_")[0])
        idx = qmap[(sm[pos]["Question"], sm[pos]["Answer"])]
        r = json.load(open(f)); a = r["answer"]; p = str(a.get("answer") if isinstance(a, dict) else a)
        ch = recs[idx].get("Choice")
        ok = lambda s: rs.am.judge(recs[idx]["Answer"], s, ch, "exact")[0]
        rows.append((idx, p, ref.get(idx), ok(p), ok(ref.get(idx, "")), logs.get(int(name.split("_")[0]))))
    same = sum(p == q for _, p, q, *_ in rows)
    gain = sum(a and not b for *_, a, b, _ in rows); loss = sum(b and not a for *_, a, b, _ in rows)
    lg = [x for *_, x in rows if x]
    mean = lambda k: sum(x.get(k, 0) for x in lg) / max(1, len(lg))
    print(f"{tag}: n={len(rows)} same_answer={same} rfnm_correct={sum(r[3] for r in rows)} nova_correct={sum(r[4] for r in rows)} "
          f"gain={gain} loss={loss} | mean r/x={mean('r_rel'):.2e} q_rel={mean('q_rel'):.2e} attn_jsd={mean('attn_jsd'):.2e} "
          f"rhoR_nat={mean('rhoR_nat'):.2e}")
    cr = [x["rfnm"]["cos_raw"] for x in lg if x.get("rfnm")]; cc = [x["rfnm"]["cos_centered"] for x in lg if x.get("rfnm")]
    er = [x["rfnm"]["erank_raw"] for x in lg if x.get("rfnm")]; ec = [x["rfnm"]["erank_centered"] for x in lg if x.get("rfnm")]
    if cr:
        print(f"   gate: cos raw {sum(cr)/len(cr):.3f} -> centered {sum(cc)/len(cc):.3f}; erank raw {sum(er)/len(er):.2f} -> centered {sum(ec)/len(ec):.2f}")
    for idx, p, q, a, b, _ in rows:
        if p != q:
            print(f"   #{idx}: NOVAρ={q!r:.45} RFNM={p!r:.45} gold={recs[idx]['Answer']!r:.35}")
