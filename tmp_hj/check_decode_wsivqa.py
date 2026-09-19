import json, glob, re, collections, sys
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from vision_text_mas.prompts import canonical_open_answer_options
K = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/nav4_wsivqa"
recs = json.load(open("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/data_wsivqa/wsivqa.json"))
for arm in ("latplan",):
    st = collections.Counter(); bad = []; ans = collections.Counter(); raw_len = []
    for f in glob.glob(f"{K}/{arm}/wsivqa/gpu*_attempt_*/*/result.json"):
        d = f.rsplit("/", 1)[0]
        try:
            r = json.load(open(f)); c = json.load(open(d + "/answerer_call.json"))
        except Exception:
            continue
        i = int(r["dataset_index"]); rec = recs[i]
        a = r.get("answer", {}); pred = str(a.get("answer", "")) if isinstance(a, dict) else str(a)

        raw = " ".join(c.get("final_outputs") or [])
        raw_len.append(len(raw))
        choices = rec.get("Choice") or list(canonical_open_answer_options(rec["Question"]))
        kind = "mcq" if rec.get("Choice") else ("open" if choices else "surv")
        st[kind] += 1
        st["repairs>0"] += bool(c.get("format_repairs"))
        issues = []
        if not pred.strip(): issues.append("EMPTY")
        if choices and pred not in choices: issues.append("NOT_IN_CHOICES")
        if kind == "surv" and not re.fullmatch(r"\d+(\.\d+)?", pred.strip()): issues.append("SURV_NOT_NUMBER")
        if len(raw) > 700: issues.append(f"LONG_RAW{len(raw)}")
        if re.search(r"(\b\w+\b)(\s+\1){4,}", raw): issues.append("REPEAT")
        if "</think>" in raw or "<think>" in raw: issues.append("THINK_TAG")
        pt = c.get("prompt", "")
        if "CHOICE 1:" in pt or (choices and "Choice: [" not in pt): issues.append("PROMPT_FMT")
        for x in issues: st[x.split("LONG_RAW")[0] or "LONG_RAW"] += 1
        if issues: bad.append((i, kind, issues, pred[:60], raw[:160]))
        ans[(kind, pred[:30])] += 1
    print(f"==== {arm}: n={sum(st[k] for k in ('mcq','open','surv'))} {dict(st)} raw_len max={max(raw_len) if raw_len else None}")
    for b in sorted(bad)[:12]: print("  ", b)
    print("  top answers:", ans.most_common(8))
