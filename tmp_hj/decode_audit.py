"""Decode audit for runs/nav4_wsivqa/<arm>: latest result per dataset_index, one row per question (no double counting).
usage: decode_audit.py ARM [ARM ...]"""
import glob, json, os, re, sys
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from vision_text_mas.prompts import canonical_open_answer_options

R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp"
W = R + "/runs/nav4_wsivqa"
recs = json.load(open(R + "/data_wsivqa/wsivqa.json"))
FOREIGN = re.compile(r"[Ѐ-ӿ가-힣぀-ヿ一-鿿]")   # Cyrillic / Hangul / kana / CJK
REPEAT = re.compile(r"(\b\S{1,20}\b)(?:[\s,\"']+\1){4,}")

for arm in sys.argv[1:]:
    latest = {}
    for f in glob.glob(f"{W}/{arm}/wsivqa/gpu*_attempt_*/*/result.json"):
        try:
            r = json.load(open(f))
            c = json.load(open(os.path.join(os.path.dirname(f), "answer_call.json" if False else "answerer_call.json")))
        except Exception:
            continue
        i = int(r["dataset_index"]); t = os.path.getmtime(f)
        if i not in latest or t > latest[i][0]:
            a = r.get("answer", {}); pred = str(a.get("answer", "") if isinstance(a, dict) else a)
            latest[i] = (t, pred, " ".join(c.get("final_outputs") or []), c.get("format_repairs", 0))
    cnt = {k: 0 for k in ("n", "empty", "not_in_choices", "foreign_script", "repeat", "long_raw", "repairs", "surv_not_number", "surv_n")}
    ex = []
    for i, (_, pred, raw, rep) in sorted(latest.items()):
        rec = recs[i]
        choices = rec.get("Choice") or list(canonical_open_answer_options(rec["Question"]))
        cnt["n"] += 1
        cnt["repairs"] += bool(rep)
        if not choices:
            cnt["surv_n"] += 1
            cnt["surv_not_number"] += not re.fullmatch(r"\s*\d+(\.\d+)?\s*", pred)
            continue
        issues = []
        if not pred.strip(): issues.append("empty")
        elif pred not in choices: issues.append("not_in_choices")
        if FOREIGN.search(raw): issues.append("foreign_script")
        if REPEAT.search(raw): issues.append("repeat")
        if len(raw) > 700: issues.append("long_raw")
        for k in issues: cnt[k] += 1
        if issues and len(ex) < 4: ex.append((i, issues, pred[:40], raw[:90].replace("\n", " ")))
    bad = sum(1 for i, (_, pred, raw, _) in latest.items()
              if (recs[i].get("Choice") or canonical_open_answer_options(recs[i]["Question"]))
              and (not pred.strip() or pred not in (recs[i].get("Choice") or list(canonical_open_answer_options(recs[i]["Question"])))
                   or FOREIGN.search(raw) or REPEAT.search(raw) or len(raw) > 700))
    print(f"== {arm}: questions {cnt['n']} | any issue (choice/label Qs) {bad} | " +
          " ".join(f"{k}={v}" for k, v in cnt.items() if k != "n"))
    for e in ex: print("   ", e)
