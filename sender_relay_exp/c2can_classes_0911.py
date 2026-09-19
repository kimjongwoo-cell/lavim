import json, collections, sys
src=open("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/rescore_dashboard_bacc.py").read().split("CELLS = []")[0]
ns={}; exec(compile(src,"rs","exec"),ns)
R0806=ns["R0806"]; RHJ=ns["RHJ"]; m=ns["m"]; REC=ns["DATASETS"]["gtex"]
arms={"base":[R0806/"multipathqa_gtex_qwen_base_allsteps_gpu5678_20260901/4b/step10"],
      "canonical":sorted(p for p in RHJ.glob("expq/routecanon_gtex10.s*") if p.is_dir()),
      "c2can":[RHJ/"consol_v/c2can_gtex"]}
P={}
for a,roots in arms.items():
    cs=ns["collect"](roots); d={}
    for i,r in cs.items():
        rec=REC[i]; ch=rec["Choice"]; gt=rec["Answer"]; ans=r.get("answer"); pred=ans.get("answer") if isinstance(ans,dict) else ans
        if not isinstance(pred,str): pred=""
        pe=m.expand_letter(pred.strip(),ch); ge=m.expand_letter(gt.strip(),ch)
        d[i]=(pe,ge,int(bool(m.acc_of_seq(ch,ge,pe))))
    P[a]=d; s=ns["score"]("gtex",cs)
    tot=collections.Counter(v[1] for v in d.values()); cor=collections.Counter(v[1] for v in d.values() if v[2])
    print(f"{a}: n {len(d)} BACC {s['bacc']*100:.2f} correct {sum(v[2] for v in d.values())}")
    print("   per-class", {k:f"{cor[k]}/{tot[k]}" for k in sorted(tot) if cor[k]})
    print("   top preds", collections.Counter(v[0] for v in d.values()).most_common(6))
for ref in ("base","canonical"):
    cc=set(P[ref])&set(P["c2can"])
    g=collections.Counter(P["c2can"][i][1] for i in cc if P["c2can"][i][2] and not P[ref][i][2])
    l=collections.Counter(P[ref][i][1] for i in cc if P[ref][i][2] and not P["c2can"][i][2])
    print(f"c2can vs {ref}: common {len(cc)} answers changed {sum(P[ref][i][0]!=P['c2can'][i][0] for i in cc)} gained {sum(g.values())} {dict(g)} lost {sum(l.values())} {dict(l)}")
