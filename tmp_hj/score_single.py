"""Score the single_v7 WSI-VQA run the same way as tmp_hj/score_wsivqa.py + wsivqa_fulltable.py:
exact (MCQ 390 / open 345 incl. survival / all 735) and metrics.py (accuracies, BLEU-1..4, METEOR, ROUGE-L, token F1).
Prediction = the adapter's `prediction` field (MCQ answers are snapped to the nearest choice by the adapter;
`raw_prediction` kept to count answers that were not a choice before snapping). Optional id filter (json list).
usage: score_single.py <run_id> [ids.json]"""
import importlib.util, json, subprocess, sys
from pathlib import Path

R = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj")
PY = "/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python"
run = sys.argv[1]
ids = set(json.load(open(sys.argv[2]))) if len(sys.argv) > 2 else None
D = R / f"sender_relay_exp/runs/single_wsivqa/qwen3-vl-4b/{run}"
recs = json.load(open(R / "sender_relay_exp/data_wsivqa/wsivqa.json"))
s = importlib.util.spec_from_file_location("am", "/home/users/whddn12316/wsi_latent_0902_2155_hj/eval/answer_match.py")
am = importlib.util.module_from_spec(s); s.loader.exec_module(am)
preds = {}
for l in open(D / "predictions/qwen3vl_predictions.jsonl"):
    r = json.loads(l)
    preds[int(r.get("sample_index", r["question_id"]))] = r
jl, n = [], {"mcq": [0, 0], "open": [0, 0]}
for i, rec in enumerate(recs):
    if ids is not None and i not in ids:
        continue
    r = preds.get(i)
    if r is None:
        continue
    assert r["slide_id"] == rec["Id"] and r["question"] == rec["Question"], i
    pred = str(r.get("prediction", ""))
    cat = "mcq" if rec.get("Choice") else "open"
    n[cat][0] += am.normalize(pred) == am.normalize(rec["Answer"]); n[cat][1] += 1
    row = {"question_id": str(i), "Id": rec["Id"], "question": rec["Question"], "prediction": pred, "ground_truth": rec["Answer"]}
    if rec.get("Choice"):
        row["Choice"] = rec["Choice"]
    jl.append(row)
tag = "all" if ids is None else "sub"
out_dir = D / f"metrics_py_{tag}"; out_dir.mkdir(exist_ok=True)
(out_dir / "scored.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in jl))
subprocess.run([PY, str(R / "code/eval/metrics.py"), "--input", str(out_dir / "scored.jsonl"), "--output-dir", str(out_dir),
                "--name", "single"], check=True, capture_output=True, text=True)
sm = json.load(open(out_dir / "single_summary.json"))
KEYS = ["mcq_accuracy", "open_substring_accuracy", "total_accuracy", "normalized_exact_match", "token_f1", "open_token_f1",
        "BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "METEOR", "ROUGE_L", "all_BLEU_1", "all_BLEU_4", "all_METEOR", "all_ROUGE_L"]
res = {"n": len(jl), "ex_mcq": 100 * n["mcq"][0] / n["mcq"][1], "ex_open": 100 * n["open"][0] / n["open"][1],
       "ex_all": 100 * (n["mcq"][0] + n["open"][0]) / len(jl), "correct": n["mcq"][0] + n["open"][0], "n_open": n["open"][1],
       **{k: sm.get(k) for k in KEYS}}
print(json.dumps(res))
