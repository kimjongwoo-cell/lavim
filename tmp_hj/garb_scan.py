import json, glob, re, os, collections, sys
root = sys.argv[1]
for arm in ["nat", "can"]:
    for ds in ["gtex", "tcga_expert_vqa", "tcga_slidebench", "panda", "tcga"]:
        files = sorted(glob.glob(f"{root}/{arm}_{ds}/*/result.json"))
        stats = collections.Counter(); ex = []
        rats = collections.Counter(); confs = collections.Counter(); ans = collections.Counter()
        maxlen = 0
        for f in files:
            d = json.load(open(f))
            rc = [r for r in d["role_calls"] if r["role"] == "answerer"][-1]
            out = rc["final_outputs"][-1] if rc["final_outputs"] else ""
            txt = '{"answer":' + out
            maxlen = max(maxlen, len(out))
            stats["n"] += 1
            if int(rc.get("format_repairs", 0)): stats["repair"] += 1
            try:
                j = json.loads(txt.strip())
            except Exception:
                # first complete object
                try:
                    j, end = json.JSONDecoder().raw_decode(txt.strip())
                    if txt.strip()[end:].strip(): stats["trailing_junk"] += 1; ex.append(("trail", out[:200]))
                except Exception:
                    stats["parse_fail"] += 1; ex.append(("parse", out[:200])); j = {}
            a = d["answer"]
            ans[str(a.get("answer"))] += 1
            r = str(a.get("rationale", "")); rats[r] += 1
            confs[str(a.get("confidence"))] += 1
            if re.search(r"[^\x00-\x7F–—≤≥×µ°±]", out): stats["nonascii"] += 1; ex.append(("nonascii", out[:200]))
            toks = re.findall(r"\w+", out)
            if len(toks) > 20 and len(set(toks)) / len(toks) < 0.3: stats["repeat"] += 1; ex.append(("repeat", out[:200]))
            if not r.strip(): stats["empty_rat"] += 1
            if d.get("decision_basis") != "PLANNED_EVIDENCE": stats["basis_" + str(d.get("decision_basis"))] += 1
        print(f"{arm}_{ds}: {dict(stats)} maxlen={maxlen}")
        print("   answers:", dict(ans.most_common(6)))
        print("   conf:", dict(confs.most_common(5)))
        print("   top rationale:", rats.most_common(1)[0][1] if rats else 0, "/", stats["n"], "|", (rats.most_common(1)[0][0][:90] if rats else ""))
        for k, e in ex[:3]: print("   EX", k, repr(e))
