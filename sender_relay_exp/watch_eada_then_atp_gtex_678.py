"""Run the GTEx ATP-SAP arm after the 6/7/8 E-AdaPrune shards finish."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/users/whddn12316/venvs/wsi-latentmas-py312/bin/python")
RUNS = REPO / "sender_relay_exp" / "runs"
WRAPPER = REPO / "0912" / "run_dataset_nav2_prompt1.sh"
EADA_PIDS = (2941549, 2941550, 2941551)
ATP_VARIANT = "pruning_atp_sap_pathology_latent_kv_relay2_dual"
ATP_PREFIX = "nav3_prompt1_atp_sap_pathology_dual_gtex_gpu"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    log_path = RUNS / "eada_then_atp_gtex_678_watch.log"
    with log_path.open("a", encoding="utf-8") as watch_log:
        watch_log.write(f"watcher started pids={EADA_PIDS}\n")
        watch_log.flush()
        while any(alive(pid) for pid in EADA_PIDS):
            time.sleep(30)
        watch_log.write("E-Ada shards exited; launching ATP-SAP shards\n")
        watch_log.flush()

        for gpu in (6, 7, 8):
            ids = ",".join(str(index) for index in range(190) if index % 3 == gpu - 6)
            output = RUNS / f"{ATP_PREFIX}{gpu}_full_20260915_sdpa"
            output_log = Path(f"{output}.log")
            if output.exists():
                watch_log.write(f"skip gpu={gpu}: output exists {output}\n")
                continue
            env = dict(os.environ)
            env.update(
                {
                    "VLMAS_ATTN_IMPLEMENTATION": "sdpa",
                    "VLMAS_NAV3_INDEPENDENT": "1",
                    "VLMAS_NAV2": "1",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONPATH": str(REPO),
                }
            )
            log = output_log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [
                    "bash",
                    str(WRAPPER),
                    str(gpu),
                    "gtex",
                    ATP_VARIANT,
                    ids,
                    str(output),
                ],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log.close()
            watch_log.write(f"launched gpu={gpu} pid={process.pid} cases={len(ids.split(','))}\n")
            watch_log.flush()


if __name__ == "__main__":
    main()
