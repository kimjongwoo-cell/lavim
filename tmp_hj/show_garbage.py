import json, sys, glob
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs"
want = {("ntrs_nav3/gtex", 13), ("ntrs_nav3/gtex", 148), ("ntrs_nav3/gtex", 26), ("nav3_prompt1_base_gtex_gpu*", 51),
        ("nav3_prompt1_base_gtex_gpu*", 120), ("qasc_nav3/gtex", 148)}
for root, idx in sorted(want):
    for rp in glob.glob(f"{R}/{root}/**/result.json", recursive=True):
        r = json.load(open(rp))
        if int(r["dataset_index"]) != idx:
            continue
        d = rp[:-len("/result.json")]
        a = json.load(open(d + "/answerer_call.json"))
        print(f"=== {root} #{idx} gold={r['gold_answer']!r}")
        print("  keys:", sorted(a.keys()))
        print("  PROMPT_TAIL:", repr(str(a.get("prompt", ""))[-500:]))
        fo = a.get("final_outputs") or a.get("raw_output") or a.get("output")
        print("  OUT:", repr(str(fo)[:600]))
        print("  ANSWER:", repr(r.get("answer")))
        break
