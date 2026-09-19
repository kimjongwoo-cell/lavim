#!/usr/bin/env python3
"""Efficiency metrics for nav3 base / C1 #14 QASC / C1 #14.6 NTRS from the per-case artifacts (no re-run).

Per case (result.json role_calls elapsed_seconds, latent_audits.json added_tokens, answerer_call.json prompt/output):
  * wall time per role (model calls only) and total
  * Reasoner prefill tokens (added_tokens of the reasoner stage) and the context at the Answerer
      c_A = planner + navigator + reasoner added tokens + 3 x 10 latent steps
  * KV cache at the Answerer: c_A x 2 x L x n_kv x head_dim x 2 bytes (bf16) = c_A x 147,456 B
  * analytic LM FLOPs with backbone/efficiency.py: flops(q, c) = 2 P q + 4 L d q (c + q), P = 3.63e9, L = 36, d = 2560,
    summed over planner/navigator/reasoner prefills, 30 latent steps, Answerer prompt prefill and decode steps
    (vision-encoder FLOPs excluded; Answerer prompt/decode token counts from the tokenizer, chat-template overhead ignored)
  * selection overhead: mean `sec=` of [QASC]/[NTRS] kept lines in the run logs
Δ columns are paired on the same dataset_index against nav3 base.
usage: eff_report.py
"""
import glob
import json
import re
import statistics as st
from pathlib import Path

R = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs")
MODEL = "/home/users/whddn12316/models/Qwen3-VL-4B-Thinking"
P, L, D = 3.63e9, 36, 2560
KV_BYTES = 2 * L * 8 * 128 * 2
DS = ["tcga_expert_vqa", "tcga_slidebench", "gtex", "tcga", "panda"]
METHODS = {"nav3 base": "nav3_prompt1_base_{ds}_gpu*", "#14 QASC": "qasc_nav3/{ds}/gpu*_attempt_*",
           "#14.6 NTRS": "ntrs_nav3/{ds}/gpu*_attempt_*", "nav3 base clamp": "nav3clamp_base/{ds}/gpu*_attempt_*", "NTRS+LPVH": "lpvh_ntrs_nav3/{ds}/gpu*_attempt_*", "NTRS+SRVW": "srvw_ntrs_nav3/{ds}/gpu*_attempt_*", "NTRS 50%": "ntrs50_nav3/{ds}/gpu*_attempt_*"}

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)


def flops(q, c):
    return 2 * P * q + 4 * L * D * q * (c + q)


def case_metrics(case_dir: Path):
    try:
        res = json.loads((case_dir / "result.json").read_text())
        aud = json.loads((case_dir / "latent_audits.json").read_text())
        ans = json.loads((case_dir / "answerer_call.json").read_text())
    except Exception:
        return None
    added = {a["stage"]: int(a["added_tokens"]) for a in aud}
    steps = {a["stage"]: int(a["latent_steps"]) for a in aud}
    roles = {}
    for rc in res["role_calls"]:
        roles[rc["role"]] = roles.get(rc["role"], 0.0) + float(rc.get("elapsed_seconds") or 0.0)
    q_ans = len(tok(ans.get("prompt", ""), add_special_tokens=False)["input_ids"])
    out = ans.get("final_outputs") or [""]
    n_dec = len(tok(out[0] if isinstance(out, list) else str(out), add_special_tokens=False)["input_ids"])
    c = 0
    fl = 0.0
    for stage in ("evidence_planner", "navigator", "reasoner"):
        q = added.get(stage, 0)
        fl += flops(q, c)
        c += q
        for _ in range(steps.get(stage, 0)):
            fl += flops(1, c)
            c += 1
    c_answer = c
    fl_ans = flops(q_ans, c)
    c += q_ans
    for _ in range(n_dec):
        fl_ans += flops(1, c)
        c += 1
    meas = {}
    if (case_dir / "efficiency.json").exists():                # VLMAS_EFF_DUMP=1 sidecar (measured)
        try:
            e = json.loads((case_dir / "efficiency.json").read_text())
            meas = {"m_flops_total": e.get("flops_total"), "m_ttft_sec": e.get("ttft_sec"),
                    "m_peak_gpu_gb": (e["peak_gpu_mem_bytes"] / 2 ** 30) if e.get("peak_gpu_mem_bytes") else None,
                    "m_prefill_tokens": e.get("prefill_tokens_total"), "m_decode_steps": e.get("decode_steps_total")}
        except Exception:
            meas = {}
    return {**meas, "idx": int(res["dataset_index"]), "reasoner_tokens": added.get("reasoner", 0), "ctx_answer": c_answer,
            "kv_answer_mb": c_answer * KV_BYTES / 2 ** 20, "flops_total": fl + fl_ans, "flops_pre_answer": fl,
            "decode_tokens": n_dec, "t_total": sum(roles.values()), "t_reasoner": roles.get("reasoner", 0.0),
            "t_answerer": roles.get("answerer", 0.0), "t_nav": roles.get("navigator", 0.0) + roles.get("navigator_detail", 0.0),
            "t_planner": roles.get("evidence_planner", 0.0)}


def collect(pattern):
    latest = {}
    for d in glob.glob(str(R / pattern)):
        for cd in Path(d).glob("*/result.json"):
            m = case_metrics(cd.parent)
            if m is None:
                continue
            mt = cd.stat().st_mtime
            if m["idx"] not in latest or latest[m["idx"]][0] < mt:
                latest[m["idx"]] = (mt, m)
    return {k: v for k, (_, v) in latest.items()}


def sel_sec(pattern, tag):
    vals = []
    for f in glob.glob(str(R / (pattern + ".log"))):
        for line in open(f, errors="ignore"):
            if line.startswith(f"[{tag}] kept"):
                m = re.search(r"sec=([0-9.]+)", line)
                if m:
                    vals.append(float(m.group(1)))
    return st.mean(vals) if vals else None


rows = []
for ds in DS:
    data = {name: collect(pat.format(ds=ds)) for name, pat in METHODS.items()}
    base = data["nav3 base"]
    for name, cases in data.items():
        if not cases:
            continue
        ids = sorted(cases)
        mean = lambda k: st.mean(cases[i][k] for i in ids)
        paired = [i for i in ids if i in base]
        dm = lambda k: (st.mean(cases[i][k] - base[i][k] for i in paired) if paired and name != "nav3 base" else None)
        rel = lambda k: (st.mean(cases[i][k] for i in paired) / st.mean(base[i][k] for i in paired) - 1 if paired and name != "nav3 base" else None)
        tag = {"#14 QASC": "QASC", "#14.6 NTRS": "NTRS"}.get(name)
        pat = METHODS[name].format(ds=ds)
        row = {"ds": ds, "method": name, "n": len(ids), "paired": len(paired),
               "reasoner_tokens": mean("reasoner_tokens"), "d_reasoner_tokens": dm("reasoner_tokens"),
               "ctx_answer": mean("ctx_answer"), "kv_answer_mb": mean("kv_answer_mb"), "rel_kv": rel("kv_answer_mb"),
               "flops_total_e12": mean("flops_total") / 1e12, "rel_flops": rel("flops_total"),
               "flops_pre_answer_e12": mean("flops_pre_answer") / 1e12, "decode_tokens": mean("decode_tokens"),
               "t_total": mean("t_total"), "rel_t_total": rel("t_total"), "t_reasoner": mean("t_reasoner"),
               "rel_t_reasoner": rel("t_reasoner"), "t_answerer": mean("t_answerer"), "t_nav": mean("t_nav"),
               "sel_sec": sel_sec(pat, tag) if tag else None}
        for mk in ("m_flops_total", "m_ttft_sec", "m_peak_gpu_gb", "m_prefill_tokens", "m_decode_steps"):
            vals = [cases[i][mk] for i in ids if cases[i].get(mk) is not None]
            row[mk] = st.mean(vals) if vals else None
            row[mk + "_n"] = len(vals)
        rows.append(row)
        f = lambda v, p=1: "—" if v is None else (f"{v:+.{p}%}" if isinstance(v, float) and abs(v) < 5 and p == "pct" else f"{v:.{p}f}")
        pct = lambda v: "—" if v is None else f"{100 * v:+.1f}%"
        print(f"{ds:16s} {name:11s} n={len(ids):3d} reasonerTok {row['reasoner_tokens']:7.0f} ctxA {row['ctx_answer']:7.0f} "
              f"KV_A {row['kv_answer_mb']:6.0f}MB ({pct(row['rel_kv'])}) FLOPs {row['flops_total_e12']:6.1f}e12 ({pct(row['rel_flops'])}) "
              f"dec {row['decode_tokens']:5.0f} t_tot {row['t_total']:5.1f}s ({pct(row['rel_t_total'])}) t_R {row['t_reasoner']:4.2f}s ({pct(row['rel_t_reasoner'])}) "
              f"t_A {row['t_answerer']:4.2f}s t_nav {row['t_nav']:5.2f}s sel {'—' if row['sel_sec'] is None else f'{row[chr(115)+chr(101)+chr(108)+chr(95)+chr(115)+chr(101)+chr(99)]:.2f}s'}",
              flush=True)
        if row["m_ttft_sec_n"] or row["m_peak_gpu_gb_n"]:
            print(f"{'':16s} {'  measured':11s} n_eff={row['m_peak_gpu_gb_n']} TTFT {row['m_ttft_sec']} s peak {row['m_peak_gpu_gb']} GB "
                  f"FLOPs(meas) {None if row['m_flops_total'] is None else round(row['m_flops_total'] / 1e12, 1)}e12", flush=True)
Path(R / "eff_report_0915.json").write_text(json.dumps(rows, indent=1))
print("-> runs/eff_report_0915.json")
