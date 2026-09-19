import json, glob, os, re, collections
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/"
cands = []
for d in sorted(glob.glob(R + "nav3_prompt1_base_*_gpu*_2026091*")):
    for f in sorted(glob.glob(d + "/*/result.json")):
        r = json.load(open(f))
        rc = [x for x in r["role_calls"] if x["role"] == "answerer"][-1]
        out = rc["final_outputs"][-1] if rc["final_outputs"] else ""
        t = ('{"answer":' + out).strip()
        try: json.loads(t); continue
        except Exception: pass
        try: json.JSONDecoder().raw_decode(t); continue
        except Exception: pass
        ans_broken = not re.match(r'\s*"[^"{}\n]*"\s*,', out)
        rep = bool(re.search(r'(\b\w+\b)(\W+\1\b){4,}', out))
        spaces = bool(re.search(r'\s{20,}', out))
        ds = re.match(r"nav3_prompt1_base_(tcga_expert_vqa|tcga_slidebench|gtex|panda|tcga)_", os.path.basename(d)).group(1)
        cands.append((3*ans_broken + 2*rep + spaces, ds, f.replace(R, "sender_relay_exp/runs/"), r["gold_answer"], r["answer"].get("answer"), out))
cands.sort(key=lambda c: -c[0])
picked, per = [], collections.Counter()
for c in cands:
    if per[c[1]] >= 2: continue
    picked.append(c); per[c[1]] += 1
    if len(picked) == 10: break
for i, (s, ds, f, g, a, out) in enumerate(picked, 1):
    o = re.sub(r" {6,}", lambda m: f" ⟨공백{len(m.group())}⟩ ", out)
    o = re.sub(r"\n{2,}", lambda m: f"⟨개행{len(m.group())}⟩", o).replace("\n", "\\n")
    print(f"### {i}. {ds} | gold={g!r} | 채점된 답={str(a)[:60]!r}")
    print(f"{f}")
    print('{"answer":' + o[:420] + ("…" if len(o) > 420 else ""))
    print()
tot = collections.Counter(); br = collections.Counter()
seen = {}
for d in sorted(glob.glob(R + "nav3_prompt1_base_*_gpu*_2026091*")):
    ds = re.match(r"nav3_prompt1_base_(tcga_expert_vqa|tcga_slidebench|gtex|panda|tcga)_", os.path.basename(d)).group(1)
    for f in glob.glob(d + "/*/result.json"):
        r = json.load(open(f)); k = (ds, r["dataset_index"]); m = os.path.getmtime(f)
        if k not in seen or m > seen[k][0]: seen[k] = (m, r)
for (ds, _), (m, r) in seen.items():
    rc = [x for x in r["role_calls"] if x["role"] == "answerer"][-1]
    out = rc["final_outputs"][-1] if rc["final_outputs"] else ""
    t = ('{"answer":' + out).strip(); tot[ds] += 1
    try: json.loads(t); continue
    except Exception: pass
    try: json.JSONDecoder().raw_decode(t); continue
    except Exception: br[ds] += 1
print("COUNTS", {k: f"{br[k]}/{tot[k]}" for k in tot})
