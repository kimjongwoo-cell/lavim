import collections
import json
import sys

sys.path.insert(0, ".")
import scripts.rescore as rs  # noqa: E402

am = rs.am
DATA = "/home/users/whddn12316/datasets/MultiPathQA/ready_wsivqa/full_no_panda"
BAL = {"gtex", "tcga", "panda"}


def cases_from(pats, variant, exclude, ds):
    dirs = rs.run_dirs(rs.resolve_roots(pats), variant, list(exclude))
    out = {}
    for d in dirs:
        for f in d.glob("*/result.json"):
            try:
                p = json.loads(f.read_text())
            except Exception:
                continue
            if rs.dataset_from_slide(str(p["slide_id"])) != ds:
                continue
            a = p.get("answer", {})
            a = a.get("answer") if isinstance(a, dict) else a
            k = int(p["dataset_index"])
            st = f.stat().st_mtime
            if k not in out or st > out[k]["t"]:
                pats_ = p.get("patches", [])
                tf = [pp.get("tissue_fraction") for pp in pats_ if pp.get("tissue_fraction") is not None]
                mags = [pp.get("magnification") for pp in pats_]
                out[k] = {"t": st, "gold": str(p["gold_answer"]), "pred": str(a),
                          "tf": (sum(tf) / len(tf) if tf else None),
                          "n5": mags.count(5), "n20": mags.count(20)}
    return out


SRC = {
    "tcga_expert_vqa": {
        "nav2": (["0912/runs/nav2v2_base_expert_*"], "base", ()),
        "nav3": (["sender_relay_exp/runs/nav3_prompt1_base_tcga_expert_vqa_gpu*_20260914"], None, ()),
        "psar": (["sender_relay_exp/runs/psar_nav3/tcga_expert_vqa"], None, ())},
    "gtex": {
        "nav2": (["0912/runs/*nav2*"], "base", ("prompt2",)),
        "nav3": (["sender_relay_exp/runs/nav3_prompt1_base_gtex_gpu*_20260914"], None, ()),
        "psar": (["sender_relay_exp/runs/psar_nav3/gtex"], None, ())},
    "panda": {
        "nav2": (["0912/runs/*nav2*"], "base", ("prompt2",)),
        "nav3": (["sender_relay_exp/runs/nav3_prompt1_base_panda_gpu*_20260914"], None, ()),
        "psar": (["sender_relay_exp/runs/psar_nav3/panda"], None, ())},
    "tcga_slidebench": {
        "nav2": (["sender_relay_exp/runs/sb_sdpa_base"], None, ()),
        "nav3": (["sender_relay_exp/runs/nav3_prompt1_base_tcga_slidebench_gpu*_20260914"], None, ()),
        "psar": (["sender_relay_exp/runs/psar_nav3/tcga_slidebench"], None, ())},
    "tcga": {
        "nav2": (["0912/runs/*nav2*"], "base", ("prompt2",)),
        "nav3": (["sender_relay_exp/runs/nav3_prompt1_base_tcga_gpu*_20260914"], None, ()),
        "psar": (["sender_relay_exp/runs/psar_nav3/tcga"], None, ())},
}


def ok(C, c, i):
    return am.judge(C[c][i]["pred"], C[c][i]["gold"], None, "exact")[0]


def score(ds, C, c, idx):
    if not idx:
        return None
    oks = {i: ok(C, c, i) for i in idx}
    if ds in BAL:
        by = collections.defaultdict(list)
        for i in idx:
            by[am.normalize(C[c][i]["gold"])].append(oks[i])
        return round(100 * sum(sum(v) / len(v) for v in by.values()) / len(by), 2), sum(oks.values())
    return round(100 * sum(oks.values()) / len(idx), 2), sum(oks.values())


only = [] if (len(sys.argv) > 1 and sys.argv[1] in ("--detail", "--fmt")) else (sys.argv[1:] or list(SRC))
for ds in only:
    src = SRC[ds]
    recs = json.load(open(f"{DATA}/{ds}.json"))
    C = {k: cases_from(*v, ds=ds) for k, v in src.items()}
    print(f"\n===== {ds}: n nav2={len(C['nav2'])} nav3={len(C['nav3'])} psar={len(C['psar'])}")
    common = sorted(set(C["nav2"]) & set(C["psar"]))
    for c in ("nav2", "psar"):
        print(f"  {c:5s} on nav2∩psar ({len(common)}): (score, correct) = {score(ds, C, c, common)}")
    tri = sorted(set(C["nav2"]) & set(C["nav3"]) & set(C["psar"]))
    if tri:
        print(f"  three-way common {len(tri)}: nav2 {score(ds, C, 'nav2', tri)} | nav3 {score(ds, C, 'nav3', tri)} | psar {score(ds, C, 'psar', tri)}")
        g = sum(ok(C, "nav3", i) and not ok(C, "nav2", i) for i in tri)
        l = sum(not ok(C, "nav3", i) and ok(C, "nav2", i) for i in tri)
        g3 = sum(ok(C, "psar", i) and not ok(C, "nav3", i) for i in tri)
        l3 = sum(not ok(C, "psar", i) and ok(C, "nav3", i) for i in tri)
        s3 = sum(C["nav3"][i]["pred"].strip() == C["psar"][i]["pred"].strip() for i in tri)
        s2 = sum(C["nav2"][i]["pred"].strip() == C["nav3"][i]["pred"].strip() for i in tri)
        print(f"  navigator effect nav2->nav3 base: gain {g} loss {l} same_answer {s2}/{len(tri)}")
        print(f"  PSAR effect nav3 base->+PSAR:   gain {g3} loss {l3} same_answer {s3}/{len(tri)}")
    g = [i for i in common if ok(C, "psar", i) and not ok(C, "nav2", i)]
    l = [i for i in common if not ok(C, "psar", i) and ok(C, "nav2", i)]
    print(f"  total nav2->nav3+PSAR: gain {len(g)} loss {len(l)} same_answer "
          f"{sum(C['nav2'][i]['pred'].strip() == C['psar'][i]['pred'].strip() for i in common)}/{len(common)}")
    missing = sorted(set(C["nav2"]) - set(C["psar"]))
    if missing and len(C["psar"]) and ds != "tcga":
        print(f"  in nav2 but not in psar: {len(missing)} {missing[:15]}")
    task = lambda i: str(recs[i].get("Task", "?"))[:44]
    tg = collections.Counter(task(i) for i in g)
    tl = collections.Counter(task(i) for i in l)
    tn = collections.Counter(task(i) for i in common)
    print("  by Task  (n / gain / loss):")
    for t, n in tn.most_common(14):
        print(f"    {t:44s} {n:4d}  +{tg.get(t, 0)}  -{tl.get(t, 0)}")
    for c in ("nav2", "psar"):
        cnt = collections.Counter(C[c][i]["pred"].strip() for i in common)
        top, tc = cnt.most_common(1)[0]
        print(f"  {c} most frequent answer {tc}/{len(common)} ({top[:30]!r}), distinct {len(cnt)}")
    gold = collections.Counter(C["nav2"][i]["gold"].strip() for i in common)
    gt, gc = gold.most_common(1)[0]
    print(f"  gold most frequent {gc}/{len(common)} ({gt[:30]!r})")
    for c in ("nav2", "psar"):
        tfs = [C[c][i]["tf"] for i in common if C[c][i]["tf"] is not None]
        n5 = [C[c][i]["n5"] for i in common]
        n20 = [C[c][i]["n20"] for i in common]
        tfs_s = f"{sum(tfs) / len(tfs):.3f} (cases {len(tfs)})" if tfs else "n/a"
        print(f"  {c} mean patch tissue_fraction {tfs_s}, x5/x20 per case {sum(n5) / len(n5):.1f}/{sum(n20) / len(n20):.1f}")


def detail(ds):
    recs = json.load(open(f"{DATA}/{ds}.json"))
    C = {k: cases_from(*v, ds=ds) for k, v in SRC[ds].items()}
    common = sorted(set(C["nav2"]) & set(C["psar"]))
    rows = []
    for i in common:
        a, b = ok(C, "nav2", i), ok(C, "psar", i)
        if a != b:
            rows.append((("LOSS" if a else "GAIN"), i, recs[i]["Question"][:90].replace("\n", " "),
                         C["nav2"][i]["gold"][:28], C["nav2"][i]["pred"][:28], C["psar"][i]["pred"][:28],
                         C["nav2"][i]["tf"], C["psar"][i]["tf"], i in C["nav3"]))
    for r in sorted(rows):
        print(f"  {r[0]} {r[1]:3d} tf {r[6]:.2f}->{r[7]:.2f} nav3base={'Y' if r[8] else '-'} | gold={r[3]!r} nav2={r[4]!r} psar={r[5]!r} | Q: {r[2]}")
    import statistics as st
    for lab in ("LOSS", "GAIN"):
        d = [r[7] - r[6] for r in rows if r[0] == lab]
        if d:
            print(f"  {lab} mean tissue_fraction change nav2->nav3: {st.mean(d):+.3f} (n {len(d)})")
    same = [C["psar"][i]["tf"] - C["nav2"][i]["tf"] for i in common if ok(C, "nav2", i) == ok(C, "psar", i)]
    print(f"  UNCHANGED mean tissue_fraction change: {st.mean(same):+.3f} (n {len(same)})")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--detail":
    for ds in sys.argv[2:]:
        print(f"\n##### detail {ds}")
        detail(ds)


def fmt(ds):
    C = {k: cases_from(*v, ds=ds) for k, v in SRC[ds].items()}
    bad = lambda s: s.strip().lower() in ("text", "", "short answer phrase") or s.strip().upper().startswith("CHOICE")
    for c in ("nav2", "nav3", "psar"):
        n = len(C[c])
        if not n:
            continue
        b = sorted(i for i in C[c] if bad(C[c][i]["pred"]))
        print(f"  {ds:16s} {c:5s} format-fail answers {len(b)}/{n} {b[:12]}")
    common = sorted(set(C["nav3"]) & set(C["psar"]))
    if common:
        print(f"  {ds:16s} on nav3∩psar ({len(common)}): nav3 fails {sum(bad(C['nav3'][i]['pred']) for i in common)}, psar fails {sum(bad(C['psar'][i]['pred']) for i in common)}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--fmt":
    for ds in sys.argv[2:]:
        fmt(ds)
