"""Run score_wsivqa.py for one arm set, then print exact (MCQ 390 / open 345 incl. survival) and every metrics.py metric."""
import json, subprocess, sys
R = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
PY = "/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python"
arms = sys.argv[1:]
out = subprocess.run([PY, f"{R}/tmp_hj/score_wsivqa.py", *arms], capture_output=True, text=True, cwd=R).stdout
d = json.loads(out)
KEYS = ["mcq_accuracy", "open_substring_accuracy", "total_accuracy", "normalized_exact_match", "token_f1", "open_token_f1",
        "BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "METEOR", "ROUGE_L", "all_BLEU_1", "all_BLEU_4", "all_METEOR", "all_ROUGE_L"]
rows = {}
for a, v in d["arms_detail"].items():
    e = v["exact"]
    n_mcq, n_lab, n_sur = e["n_mcq"], e["n_open"], e["n_surv"]
    c_mcq = round(e["mcq_acc"] * n_mcq / 100); c_lab = round(e["open_acc"] * n_lab / 100); c_sur = round(e["surv_acc"] * n_sur / 100)
    s = json.load(open(f"{R}/sender_relay_exp/runs/nav4_wsivqa/{a}/metrics_py/{a}_summary.json"))
    rows[a] = {"n": d["n_scored"], "ex_mcq": 100 * c_mcq / n_mcq, "ex_open": 100 * (c_lab + c_sur) / (n_lab + n_sur),
               "ex_all": e["overall_acc"], "correct": e["correct"], "n_open": n_lab + n_sur,
               "vs": [x for k, x in v.items() if k.startswith("vs_")], **{k: s.get(k) for k in KEYS}}
print(json.dumps(rows, ensure_ascii=False))
