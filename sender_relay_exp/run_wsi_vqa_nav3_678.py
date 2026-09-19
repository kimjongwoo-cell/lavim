"""Run WSI-VQA Nav3 Base, then Pruning-B + Dual Relay2 on GPUs 6/7/8."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "sender_relay_exp" / "runs"
PYTHON = "/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python"
DATASET = "/home/users/whddn12316/datasets/WSI-VQA/WsiVQA_test.json"
SLIDES = "/home/users/whddn12316/datasets/WSI-VQA/DATA_SVS"
MODEL = "/home/users/whddn12316/models/Qwen3-VL-4B-Thinking"
GPUS = (6, 7, 8)
STAMP = "20260915_678_step10"

COMMON = [
    "-m", "wsi_latentmas.pipeline.latent_mas", "--backbone", "qwen3-vl",
    "--dataset", DATASET, "--slide-root", SLIDES, "--model", MODEL,
    "--device", "cuda:0", "--latent-steps", "10", "--patch-budget", "12",
    "--max-model-len", "12288", "--navigator-control-tokens", "512",
    "--temperature", "0.0", "--top-p", "1.0", "--seed", "42",
    "--deterministic", "--answerer-greedy", "--no-answerer-thinking",
    "--answerer-max-new-tokens", "512", "--answerer-rationale",
    "--answerer-protocol", "structured_json", "--canonical-open-options",
    "--navigator-kv", "--io-pipeline", "--no-save-navigation-pngs",
    "--case-retries", "0",
]


def run_phase(controller_log, condition: str, variant: str) -> None:
    processes: list[tuple[int, subprocess.Popen[bytes]]] = []
    for slot, gpu in enumerate(GPUS):
        ids = [str(index) for index in range(735) if index % 3 == slot]
        output = RUNS / f"nav3_prompt1_{condition}_wsi_vqa_gpu{gpu}_{STAMP}"
        if output.exists():
            raise RuntimeError(f"output root must be new: {output}")
        log_path = Path(f"{output}.log")
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "VLMAS_ATTN_IMPLEMENTATION": "sdpa",
            "VLMAS_PROMPT_SET": "prompt1",
            "VLMAS_NAV2": "1",
            "VLMAS_NAV3_INDEPENDENT": "1",
            "VLMAS_CROSS_SCALE_ROUTER": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(REPO),
        })
        command = [PYTHON, *COMMON, "--variant", variant]
        for index in ids:
            command.extend(("--dataset-index", index))
        command.extend(("--output-root", str(output)))
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        controller_log.write(
            f"START condition={condition} gpu={gpu} pid={process.pid} cases={len(ids)} output={output}\n"
        )
        controller_log.flush()
        processes.append((gpu, process))

    failed = False
    for gpu, process in processes:
        return_code = process.wait()
        controller_log.write(f"END condition={condition} gpu={gpu} rc={return_code}\n")
        controller_log.flush()
        failed = failed or return_code != 0
    if failed:
        controller_log.write(f"WARNING condition={condition} had failures; continuing\n")
        controller_log.flush()


def main() -> None:
    controller_path = RUNS / f"wsi_vqa_nav3_678_controller_{STAMP}.log"
    with controller_path.open("w", encoding="utf-8") as controller_log:
        controller_log.write("controller started\n")
        controller_log.flush()
        run_phase(controller_log, "base", "base")
        run_phase(controller_log, "both", "pruning_b_latent_kv_relay2_dual")
        controller_log.write("controller finished\n")


if __name__ == "__main__":
    main()
