#!/usr/bin/env python3
"""Headroom analysis over existing full runs (read-only; input = dump.jsonl pulled from remote)."""
import collections, difflib, json, math, os, re, sys
from hr_lib import *

H = "/home/users/whddn12316/"
S = H + "wsi_latent_0915_decode_hj/sender_relay_exp/runs/"
T = H + "wsi_latent_0915_decode_hj/"
R = H + "wsi_latentmas_0806/results/runs/"
DSN = {"gtex": "gtex", "tcga_expert_vqa": "evqa", "tcga_slidebench": "sb", "panda": "panda", "tcga": "tcga"}

rows = load_dump()
L = merge(rows)
BYNAME = {}
for (n, ds), e in L.items():
    BYNAME[(n, ds)] = e
PI_CACHE = {}


def get(path, ds):
    e = BYNAME.get((path, ds))
    if e is None:
        return None
    if (path, ds) not in PI_CACHE:
        PI_CACHE[(path, ds)] = per_item(e)
    return PI_CACHE[(path, ds)]


def short(p):
    return p.replace(S, "S:").replace(T, "T:").replace(R, "R:")


def pred_class(ds, idx, pe):
    """map expanded prediction to a choice: argmax quick_ratio (first max); None if no pred."""
    if pe is None:
        return None
    ch = DS[ds][idx]["Choice"]
    best, bs = None, -1
    for c in ch:
        r = difflib.SequenceMatcher(None, pe, str(c)).quick_ratio()
        if r > bs:
            best, bs = c, r
    return best


# ---------------------------------------------------------------- groups
CONTAM = {"c2orig_gtex10", "c2orig_evqa10", "c2shuforder_gtex10", "c2shuforder_evqa10"}
CONTROL_PAT = re.compile(r"wrong|shuf|rand|nokey|ident|aavm2|aavmdiag|ver_|geom_audit|kvprobe")
C2_PAT = re.compile(r"c2can|D_cross|E_both|c2_restage|restage")
DIAG_PAT = re.compile(r"smoke|rpath_|kvswap|accessdiag|accterm|latentdecode|latentreadout|failed|partial|inert")


def expq_arms(ds, steps, model="4B"):
    out = []
    for (n, d), e in BYNAME.items():
        if d != ds:
            continue
        if not (n.startswith(S + "expq/") or n.startswith(S + "prune_v/") or n.startswith(S + "generalize/")
                or n.startswith(S + "allsteps_matrix_20260903/")):
            continue
        b = os.path.basename(n)
        if n.startswith(S + "allsteps_matrix_20260903/"):
            if f"/{model.lower()}/" not in n:
                continue
        else:
            if re.search(r"(2b|8b)", b):
                if model == "4B":
                    continue
            elif model != "4B":
                continue
        if b.startswith("_") or DIAG_PAT.search(b) or b in CONTAM:
            continue
        if e["manifest"].get("latent_steps") != steps:
            continue
        out.append(n)
    return sorted(out)


def G1():
    base = {"gtex": R + "multipathqa_gtex_qwen_base_allsteps_gpu5678_20260901/4b/step10",
            "tcga_expert_vqa": R + "multipathqa_remaining_qwen_base_allsteps_gpu5678_20260902/tcga_expert_vqa/4b/step10",
            "tcga_slidebench": S + "expq/sb_base10", "tcga": S + "expq/tcga_base10", "panda": S + "expq/panda_base10"}
    groups = {}
    for ds in DSK:
        core = [S + f"consol_v/{v}_{ds}" for v in ("scale", "wsi", "scale512", "wsi512", "hier25", "v3c512", "c2can")]
        core += [S + f"qpris_full/qt_{ds}", S + f"qpris_full/qn_{ds}", S + f"c2_replication/{ds}/4b/D_cross_step10"]
        core = [c for c in core if (c, ds) in BYNAME]
        ext = [n for n in expq_arms(ds, 10) if n != base[ds]]
        groups[ds] = {"base": base[ds], "core": core, "ext": ext}
    return groups


def G2():
    base = {"gtex": S + "multipath_qwen4b_step5_20260903/gtex/base",
            "tcga_expert_vqa": S + "multipath_qwen4b_step5_20260903/tcga_expert_vqa/base",
            "tcga_slidebench": S + "generalize/slidebench_base", "tcga": S + "generalize/tcga_base"}
    groups = {}
    for ds, b in base.items():
        core = [S + f"consolidation_ablation/{ds}/{a}" for a in ("C_rescue", "D_cross", "E_both")]
        core += [S + f"c2_restage/{ds}/c2_restage", S + f"multipath_qwen4b_step5_20260903/{ds}/pruning_v3"]
        core += [S + f"restage_probe_mp/{ds}/{a}" for a in ("identity_full", "restage", "wrong_restage")]
        core += [S + f"restage_stack/{a}" for a in ("bind_drop_full", "early_drop_full", "early_full", "park_full")] if ds == "gtex" else []
        core += [S + f"consol33/{ds}/4b/consol33_step5", S + f"consol33_v2/{ds}/4b/consol33v2_step5"]
        core += [S + f"budget_curve/{ds}/keep{k}" for k in ("0.25", "0.5", "0.75", "0.5_wrong")]
        if ds == "tcga_expert_vqa":
            core += [S + "restage_followups/wrong_evqa_4b5"]
        core = [c for c in core if (c, ds) in BYNAME]
        ext = [n for n in expq_arms(ds, 5) if n != b]
        ext += [n for (n, d) in BYNAME if d == ds and (n.startswith(S + f"multipath_qwen4b_step5_20260903/{ds}/pr_")
                                                     or n.startswith(S + f"refeed_qwen4b_step5_20260903/{ds}/"))]
        groups[ds] = {"base": b, "core": core, "ext": sorted(set(ext) - set(core))}
    return groups


def G3():
    out = {}
    for ds, rds in (("gtex", "multipathqa_gtex_qwen_base_allsteps_gpu5678_20260901/4b/step20"),
                    ("tcga_expert_vqa", "multipathqa_remaining_qwen_base_allsteps_gpu5678_20260902/tcga_expert_vqa/4b/step20")):
        core = [S + f"c2_replication/{ds}/4b/D_cross_step20"] + ([S + "expq/gtex20_restage"] if ds == "gtex" else [])
        out[ds] = {"base": R + rds, "core": [c for c in core if (c, ds) in BYNAME], "ext": expq_arms(ds, 20)}
        out[ds]["ext"] = [x for x in out[ds]["ext"] if x not in core]
    return out


def G4():
    out = {}
    for ds, rds in (("gtex", "multipathqa_gtex_qwen_base_allsteps_gpu5678_20260901/8b/step20"),
                    ("tcga_expert_vqa", "multipathqa_remaining_qwen_base_allsteps_gpu5678_20260902/tcga_expert_vqa/8b/step20")):
        nm = "gtex" if ds == "gtex" else "evqa"
        core = [S + f"c2_replication/{ds}/8b/D_cross_step20", S + f"restage_followups/restage_8b20_{nm}"]
        if ds == "gtex":
            core.append(S + "expq/gtex8b20_park")
        ext = [n for (n, d) in BYNAME if d == ds and n.startswith(S + f"allsteps_matrix_20260903/{ds}/8b/") and n.endswith("step20")]
        out[ds] = {"base": R + rds, "core": [c for c in core if (c, ds) in BYNAME], "ext": sorted(ext)}
    return out


O = T + "0912/runs/"


def OTHER():
    return {
        "O1 EVQA pb12 SDPA(0912, base_sdpa 계열)": {"tcga_expert_vqa": {"base": O + "base_sdpa", "core": [O + x for x in (
            "pruning_b_sdpa", "relay2_adaptive", "pruning_b_relay2_full128_0912", "pruning_b_relay2_broad_full128_0913",
            "attention_context_full128_0912", "attention_context_read_full128_0912", "contextual_adaptive_gain3_full128_0912",
            "contextual_adaptive_gain4_full128_0912", "contextual_adaptive_gain6_full128_0912", "lvr128_0912",
            "adaptive_consistency_singlepass_128", "relation_contrast_real_full")], "ext": []}},
        "O2 EVQA nav2v2 SDPA(0913)": {"tcga_expert_vqa": {"base": O + "nav2v2_base_expert_0913", "core": [O + x for x in (
            "nav2v2_pruning_b_expert_0913", "nav2v2_relay2_expert_0913", "nav2v2_both_expert_0913",
            "nav2v2_factorial_pruning_relay_dual_expert_0913", "nav2v2_factorial_pruning_relay_nav_expert_0913",
            "nav2v2_factorial_relay_dual_expert_0913", "nav2v2_factorial_relay_nav_expert_0913")], "ext": []}},
        "O3 GTEx pb12 12patch(0913, attn 미확인)": {"gtex": {"base": O + "gtex_base_12patch_0913", "core": [O + x for x in (
            "gtex_latent_kv_relay2_12patch_0913", "gtex_pruning_b_12patch_0913", "gtex_pruning_b_latent_kv_relay2_12patch_0913")], "ext": []}},
        "O4 SB pb12 12patch(0913, attn 미확인)": {"tcga_slidebench": {"base": O + "tcga_slidebench_base_12patch_0913", "core": [
            O + "tcga_slidebench_pruning_b_12patch_0913"], "ext": []}},
        "O5 EVQA pb12 eager(9/7 T:runs)": {"tcga_expert_vqa": {"base": T + "runs/expertvqa_latent_base_x5_4_x20_8_latent10_gpu8_mlen12288", "core": [T + "runs/" + x for x in (
            "expertvqa_pruning_a_x5_4_x20_8_latent10_gpu8_mlen12288_full", "expertvqa_pruning_b_contextual_x5_4_x20_8_latent10_gpu8_mlen12288_full",
            "expertvqa_reallocation_a_direct_relay_x5_4_x20_8_latent10_gpu8_mlen12288_full")], "ext": []}},
        "O6 GTEx/PANDA pb12(9/8 T:runs multipath_remaining_patch12)": {
            "gtex": {"base": T + "runs/multipath_remaining_patch12_4b_latent10_20260908/gtex/latent_base", "core": [
                T + "runs/multipath_remaining_patch12_4b_latent10_20260908/gtex/" + x for x in ("pruning_b", "reallocation_a", "both")], "ext": []},
            "panda": {"base": T + "runs/multipath_remaining_patch12_4b_latent10_20260908/panda/latent_base", "core": [
                T + "runs/multipath_remaining_patch12_4b_latent10_20260908/panda/pruning_b"], "ext": []}},
    }


# ---------------------------------------------------------------- metrics

def entropy(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c) if n else 0.0


def analyze(ds, base, arms, label, cov_min=0.9, verbose_arms=True):
    bpi = get(base, ds)
    if bpi is None:
        return {"error": f"base missing {short(base)}"}
    bidx = set(bpi)
    res = {"ds": ds, "label": label, "base": short(base), "base_n": len(bpi)}
    bs, bn = score_items(ds, bpi)
    res["base_score"] = bs
    arm_rows = []
    used = []
    excluded = []
    for a in arms:
        api = get(a, ds)
        if api is None:
            excluded.append((short(a), "없음"))
            continue
        cov = len(set(api) & bidx) / len(bidx)
        s, n = score_items(ds, api)
        common = sorted(set(api) & bidx)
        w = sum(1 for i in common if not bpi[i][2] and api[i][2])
        l = sum(1 for i in common if bpi[i][2] and not api[i][2])
        # delta on common set (same questions)
        sb_c, _ = score_items(ds, bpi, set(common))
        sa_c, _ = score_items(ds, api, set(common))
        same_ans = sum(1 for i in common if bpi[i][0] == api[i][0])
        row = {"arm": short(a), "n": len(api), "score": s, "delta": s - bs, "delta_common": sa_c - sb_c,
               "wins": w, "losses": l, "cov": cov, "same_ans": same_ans / max(1, len(common)),
               "nonstr": sum(1 for v in api.values() if v[0] is None)}
        arm_rows.append(row)
        if cov >= cov_min:
            used.append(a)
        else:
            excluded.append((short(a), f"coverage {len(api)}/{len(bidx)}"))
    res["arms"] = arm_rows
    res["excluded"] = excluded

    def union_stats(arm_list):
        pis = [bpi] + [get(a, ds) for a in arm_list]
        anyc = {i: int(any(p.get(i, (None, None, 0))[2] for p in pis)) for i in bidx}
        # oracle metric in dataset metric
        pseudo = {i: ("x", bpi[i][1], anyc[i]) for i in bidx}
        osc, _ = score_items(ds, pseudo)
        return sum(anyc.values()), osc, anyc

    k, osc, anyc = union_stats(used)
    res["n_used"] = len(used) + 1
    res["oracle_k"] = k
    res["oracle_score"] = osc
    res["base_correct"] = sum(bpi[i][2] for i in bidx)
    c2 = [a for a in used if C2_PAT.search(os.path.basename(a)) and not re.search(r"wrong|identity", a)]
    k2, osc2, _ = union_stats(c2)
    res["c2_arms"] = [short(a) for a in c2]
    res["c2_oracle_k"], res["c2_oracle_score"] = k2, osc2
    noctl = [a for a in used if not CONTROL_PAT.search(os.path.basename(a))]
    k3, osc3, _ = union_stats(noctl)
    res["noctl_n"], res["noctl_oracle_k"], res["noctl_oracle_score"] = len(noctl) + 1, k3, osc3

    # never-correct
    never = [i for i in bidx if not anyc[i]]
    res["never"] = len(never)
    pis = [bpi] + [get(a, ds) for a in used]
    # per-question: gold choice never output by any run on that question
    gold_never_pred_q = 0
    for i in never:
        ge = bpi[i][1]
        outs = set(pred_class(ds, i, p[i][0]) for p in pis if i in p and p[i][0] is not None)
        if ge not in outs:
            gold_never_pred_q += 1
    res["never_gold_not_output_q"] = gold_never_pred_q
    if ds in BAL:
        allpred = collections.Counter()
        for p in pis:
            for i, v in p.items():
                if i in bidx and v[0] is not None:
                    allpred[pred_class(ds, i, v[0])] += 1
        gold_cnt = collections.Counter(bpi[i][1] for i in bidx)
        never_pred_classes = {c for c in gold_cnt if allpred[c] == 0}
        res["classes_never_predicted_any_run"] = {c: gold_cnt[c] for c in sorted(never_pred_classes)}
        res["never_in_never_pred_class"] = sum(1 for i in never if bpi[i][1] in never_pred_classes)
        # base class analysis
        bpred = collections.Counter(pred_class(ds, i, bpi[i][0]) for i in bidx)
        res["base_pred_entropy_bits"] = entropy(bpred)
        res["n_gold_classes"] = len(gold_cnt)
        res["max_entropy_bits"] = math.log2(len(DS[ds][0]["Choice"])) if ds != "gtex" else math.log2(len(set(c for r in DS[ds] for c in r["Choice"])))
        top = bpred.most_common(3)
        res["base_top_pred"] = [(c, n, n / len(bidx)) for c, n in top]
        res["base_distinct_pred"] = len([c for c in bpred if c is not None])
        rec = {}
        for c in gold_cnt:
            idxs = [i for i in bidx if bpi[i][1] == c]
            rec[c] = (sum(bpi[i][2] for i in idxs), len(idxs))
        res["base_recall"] = rec
        zero_all = {}
        for c in gold_cnt:
            idxs = [i for i in bidx if bpi[i][1] == c]
            if not any(anyc[i] for i in idxs):
                zero_all[c] = len(idxs)
        res["zero_recall_all_arms"] = zero_all
        res["zero_recall_base"] = {c: n for c, (k_, n) in rec.items() if k_ == 0}
        res["gold_dist"] = dict(gold_cnt)
    # invariance
    def inv(arm_list):
        pl = [bpi] + [get(a, ds) for a in arm_list]
        same = 0; tot = 0
        for i in bidx:
            ans = [p[i][0] for p in pl if i in p]
            tot += 1
            if len(set(ans)) == 1:
                same += 1
        return same, tot
    res["inv_all"] = inv(used)
    res["inv_noctl"] = inv(noctl)
    # never-correct restricted to non-control union
    k3set = union_stats(noctl)[2]
    never3 = [i for i in bidx if not k3set[i]]
    res["never_noctl"] = len(never3)
    if ds in BAL:
        res["never_noctl_in_never_pred_class"] = sum(1 for i in never3 if bpi[i][1] in res["classes_never_predicted_any_run"])
        zero3 = {}
        for c in res["gold_dist"]:
            idxs = [i for i in bidx if bpi[i][1] == c]
            if not any(k3set[i] for i in idxs):
                zero3[c] = len(idxs)
        res["zero_recall_noctl"] = zero3
    res["used"] = [short(a) for a in used]
    return res


def fmt(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def run_group(title, groups, out, core_only=False):
    out.append(f"\n## {title}")
    allres = {}
    for ds, g in groups.items():
        arms = g["core"] + ([] if core_only else g["ext"])
        r_all = analyze(ds, g["base"], arms, "all")
        r_core = analyze(ds, g["base"], g["core"], "core")
        allres[ds] = {"all": r_all, "core": r_core}
    return allres


if __name__ == "__main__":
    OUT = {}
    OUT["G1"] = {ds: {"all": analyze(ds, g["base"], g["core"] + g["ext"], "all"), "core": analyze(ds, g["base"], g["core"], "core")}
                 for ds, g in G1().items()}
    OUT["G1_def"] = {ds: {k: ([short(x) for x in v] if isinstance(v, list) else short(v)) for k, v in g.items()} for ds, g in G1().items()}
    OUT["G2"] = {ds: {"all": analyze(ds, g["base"], g["core"] + g["ext"], "all"), "core": analyze(ds, g["base"], g["core"], "core")}
                 for ds, g in G2().items()}
    OUT["G2_def"] = {ds: {k: ([short(x) for x in v] if isinstance(v, list) else short(v)) for k, v in g.items()} for ds, g in G2().items()}
    OUT["G3"] = {ds: {"all": analyze(ds, g["base"], g["core"] + g["ext"], "all")} for ds, g in G3().items()}
    OUT["G4"] = {ds: {"all": analyze(ds, g["base"], g["core"] + g["ext"], "all")} for ds, g in G4().items()}
    OUT["OTHER"] = {t: {ds: {"all": analyze(ds, g["base"], g["core"], "all")} for ds, g in grp.items()} for t, grp in OTHER().items()}
    json.dump(OUT, open(os.path.join(HERE, "headroom_results.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print("written")
