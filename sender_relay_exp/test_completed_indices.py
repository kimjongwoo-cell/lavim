"""Dataset-boundary regression checks for resumable 0912 experiment drivers."""

import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "0912" / "completed_indices.py"
    spec = importlib.util.spec_from_file_location("completed_indices", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tcga_resume_excludes_prefixed_benchmarks(tmp_path: Path) -> None:
    """Given related dataset names, only exact TCGA slide IDs count as complete."""
    run = tmp_path / "nav2_prompt1_pruning_a_dual_both_tcga_mixed"
    for index, slide_id in enumerate(
        ("tcga__case", "tcga_expert_vqa__case", "tcga_slidebench__case")
    ):
        result = run / f"{index:03d}_{slide_id}" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({"dataset_index": index, "slide_id": slide_id}))

    completed = load_module().completed_indices(
        tmp_path,
        run_prefix="nav2_prompt1_pruning_a_dual_both_tcga_",
        dataset="tcga",
    )

    assert completed == (0,)
