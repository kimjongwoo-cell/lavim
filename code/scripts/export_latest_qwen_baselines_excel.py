"""Write every current dashboard metric into the single results workbook."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from shutil import copy2
import sys
from typing import TypeAlias

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill


CODE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = CODE_ROOT.parent / "results"
sys.path.insert(0, str(CODE_ROOT))

from dashboard import server


OUTPUT = RESULTS_ROOT / "main_performance.xlsx"
BACKUP_DIR = RESULTS_ROOT / "archive"
HEADERS = (
    "Dataset", "Backbone", "Method", "Parameter", "Latent step",
    "Evaluation metric", "Primary score", "WSI Total", "WSI Closed",
    "WSI Open", "BLEU-1", "BLEU-4", "ROUGE-L", "METEOR",
    "Seconds / question", "Samples", "Status",
)
CellValue: TypeAlias = str | int | float | None
MetricRow: TypeAlias = dict[str, CellValue]
JobValue: TypeAlias = CellValue | list[int]
JobRow: TypeAlias = dict[str, JobValue]


def current_rows() -> tuple[list[MetricRow], list[JobRow]]:
    """Read all live quality metrics and job states from the dashboard source."""
    payload = json.loads(server.status_payload())
    rows: list[MetricRow] = [
        row
        for row in payload["quality_metrics"]
        if isinstance(row.get("samples"), int) and row["samples"] > 0
    ]
    jobs: list[JobRow] = payload["jobs"]
    model_order = {"2B": 0, "4B": 1, "8B": 2}
    def row_order(row: MetricRow) -> tuple[str, str, int, str, int]:
        latent_steps = row.get("latent_steps")
        return (
            str(row.get("dataset", "WSI-VQA")),
            str(row.get("backbone", "")),
            model_order.get(str(row.get("model_size")), 9),
            str(row.get("mode", "")),
            int(latent_steps) if isinstance(latent_steps, (int, str)) else -1,
        )

    ordered_rows = sorted(
        rows,
        key=row_order,
    )
    return ordered_rows, jobs


def main() -> None:
    """Back up and replace the workbook with all current dashboard results."""
    rows, jobs = current_rows()
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"main_performance_precanonical_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    if OUTPUT.is_file():
        copy2(OUTPUT, backup)

    workbook = Workbook()
    sheet = workbook.active
    if sheet is None:
        raise RuntimeError("Workbook did not create an active worksheet")
    sheet.title = "Main Performance"
    sheet.append(HEADERS)
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    for row in rows:
        multipath = str(row.get("dataset", "")).startswith("MultiPathQA/")
        sheet.append((
            row.get("dataset"), row.get("backbone"), row.get("mode"), row.get("model_size"),
            row.get("latent_steps"), row.get("score_label"),
            row.get("total") if multipath else row.get("mcq"),
            row.get("total"), row.get("mcq"), row.get("open"), row.get("bleu_1"),
            row.get("bleu_4"), row.get("rouge_l"), row.get("meteor"),
            row.get("seconds_per_case"), row.get("samples"), row.get("state"),
        ))
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column, width in {
        "A": 28, "B": 14, "C": 30, "D": 12, "E": 12, "F": 18,
        "G": 14, "H": 12, "I": 12, "J": 12, "K": 12, "L": 12,
        "M": 12, "N": 12, "O": 20, "P": 10, "Q": 16,
    }.items():
        sheet.column_dimensions[column].width = width

    jobs_sheet = workbook.create_sheet("Run Status")
    job_headers = ("Dataset", "Job", "State", "GPU", "Processed", "Success", "Failure", "Seconds / question")
    jobs_sheet.append(job_headers)
    for cell in jobs_sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    for job in jobs:
        allocated_gpus = job.get("allocated_gpus")
        gpu_text = ", ".join(
            str(gpu)
            for gpu in allocated_gpus
            if isinstance(gpu, int)
        ) if isinstance(allocated_gpus, list) else ""
        jobs_sheet.append((
            job.get("dataset"), job.get("name"), job.get("state_label"),
            gpu_text,
            job.get("processed"), job.get("success"), job.get("failure"),
            job.get("seconds_per_case"),
        ))
    jobs_sheet.freeze_panes = "A2"
    jobs_sheet.auto_filter.ref = jobs_sheet.dimensions
    for column, width in {"A": 28, "B": 56, "C": 14, "D": 14, "E": 12, "F": 12, "G": 12, "H": 20}.items():
        jobs_sheet.column_dimensions[column].width = width

    readme = workbook.create_sheet("README")
    readme.append(("Updated at", datetime.now().isoformat(timespec="seconds")))
    readme.append(("Source", "Live dashboard metric aggregation"))
    readme.append(("Rows", len(rows)))
    readme.append(("Accuracy", "ExpertVQA and SlideBench"))
    readme.append(("Balanced accuracy", "TCGA, GTEx, and PANDA"))
    readme.append(("Backup", str(backup.relative_to(ROOT)) if OUTPUT.is_file() else "None"))
    workbook.save(OUTPUT)
    print(f"wrote={OUTPUT}")
    print(f"backup={backup}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
