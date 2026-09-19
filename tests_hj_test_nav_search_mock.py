"""Mock e2e for VLMAS_NAV_SEARCH wiring (no GPU, fake slide + fake client)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")

from PIL import Image

os.environ["VLMAS_NAV_SEARCH"] = "1"

from vision_text_mas.contracts import EvidenceRound, NavigationAction, RoleCall
from vision_text_mas.navigation_contracts import EvidencePlan
from vision_text_mas.onepass_navigation_roots import prepare_root_grid
from vision_text_mas.onepass_navigation_search import run_search_navigation
from vision_text_mas.qwen_client import ParsedRoleCall


class FakeSlide:
    """Solid pink 'tissue' slide big enough for a 2x3 root grid."""

    def __init__(self) -> None:
        self._width = 24_576
        self._height = 16_384

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return (1.0, 32.0)

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        return Image.new("RGB", size, (200, 120, 160))

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return 1 if downsample >= 32.0 else 0

    def read_region(self, location, level, size) -> Image.Image:
        return Image.new("RGB", size, (200, 120, 160))


class FakeClient:
    """Return a fixed evidence query per role and record the calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        role = kwargs["role"]
        adapter = kwargs["output_adapter"]
        labels = kwargs.get("image_labels", ())
        if role == "navigator":
            payload = {"evidence_query": "sheets of uniform pink epithelium at low power"}
        elif os.environ.get("VLMAS_NAV_EVENTS") == "1":
            parents = [lbl.replace("x5-", "") for lbl in labels]
            payload = {
                "fine_queries": [
                    {"parent_id": p, "evidence_query": f"nuclear detail in {p}"}
                    for p in parents
                ]
            }
        else:
            payload = {
                "evidence_query": "nuclear detail and mitotic figures in the densest epithelium"
            }
        value = adapter.validate_python(payload)
        query = payload.get("evidence_query", "fine-array")
        record = RoleCall(
            role=role,
            prompt=kwargs["user_prompt"],
            image_labels=kwargs["image_labels"],
            reasoning="",
            final_outputs=(json.dumps({"evidence_query": query}),),
            format_repairs=0,
            physical_calls=1,
            elapsed_seconds=0.01,
        )
        return ParsedRoleCall(value=value, record=record)


PLAN = EvidencePlan.model_validate(
    {
        "question_focus": "identify the tissue of origin",
        "needed_visual_information": ["glandular architecture"],
        "thumbnail_observations": ["pink stained section"],
        "search_instruction": "look for epithelium",
        "success_criteria": ["origin identified"],
        "detail_trigger": "cellular detail needed",
        "scale_plan": {
            "overview_target": "overall architecture and staining pattern",
            "overview_patch_count": 3,
            "detail_target": "nuclear morphology and mitoses",
            "detail_patch_count": 5,
            "detail_anchor_count": 3,
            "scale_success_criteria": ["both scales covered"],
            "patch_budget": 8,
        },
    }
)


def main() -> None:
    artifact_root = Path(
        "/home/users/whddn12316/wsi_latent_0915_decode_hj/tmp_hj/"
        "scratchpad/nav_search_mock_artifacts"
    )
    slide = FakeSlide()
    thumbnail = slide.get_thumbnail((1024, 683))
    root_grid = prepare_root_grid(slide, thumbnail=thumbnail)
    client = FakeClient()

    patches, records, trace = run_search_navigation(
        slide,
        client=client,
        root_grid=root_grid,
        evidence_plan=PLAN,
        artifact_root=artifact_root,
        round_number=1,
    )

    # 1. two query calls with the right roles and image counts
    assert [c["role"] for c in client.calls] == ["navigator", "navigator_detail"], (
        client.calls
    )
    assert len(client.calls[0]["images"]) == 0
    assert len(client.calls[1]["images"]) == 3, len(client.calls[1]["images"])

    # 2. bank composition: 3 x5 + 3*2 x20 = 9, sequential IDs, valid round
    assert len(patches) == 9, len(patches)
    scales = [int(p.magnification) for p in patches]
    assert scales == [5, 5, 5] + [20] * 6, scales
    evidence_round = EvidenceRound(
        round_number=1, action=NavigationAction.INITIAL, patches=patches
    )
    assert len(evidence_round.patches) == 9

    # 3. provenance: every x20 parent is one of the x5 patch ids
    x5_ids = {p.patch_id for p in patches[:3]}
    for patch in patches[3:]:
        assert patch.source_x5_anchor in x5_ids, patch
        assert patch.source_x5_anchor != patch.patch_id

    # 4. x20 boxes nested inside their parent x5 box
    by_id = {p.patch_id: p for p in patches}
    for patch in patches[3:]:
        parent = by_id[patch.source_x5_anchor]
        assert patch.box.x >= parent.box.x and patch.box.y >= parent.box.y
        assert patch.box.x + patch.box.width <= parent.box.x + parent.box.width
        assert patch.box.y + patch.box.height <= parent.box.y + parent.box.height

    # 5. provenance artifact with both queries
    prov = json.loads(
        (artifact_root / "round_1_search_provenance.json").read_text()
    )
    assert prov["coarse_query"].startswith("sheets of uniform")
    assert prov["fine_query"].startswith("nuclear detail")
    assert len(prov["crops"]) == 9

    # 6. records + trace shape
    assert len(records) == 2
    assert trace.label == "round-1-search-coarse"

    # 7. fine prompt saw the detail target; coarse the overview target
    assert "nuclear morphology and mitoses" in client.calls[1]["user_prompt"]
    assert "overall architecture" in client.calls[0]["user_prompt"]

    print("MOCK NAV_SEARCH PASS: 9 patches (3x5+6x20), provenance + queries OK")


def main_events() -> None:
    os.environ["VLMAS_NAV_EVENTS"] = "1"
    artifact_root = Path(
        "/home/users/whddn12316/wsi_latent_0915_decode_hj/tmp_hj/"
        "scratchpad/nav_events_mock_artifacts"
    )
    slide = FakeSlide()
    thumbnail = slide.get_thumbnail((1024, 683))
    root_grid = prepare_root_grid(slide, thumbnail=thumbnail)
    client = FakeClient()
    patches, records, trace = run_search_navigation(
        slide, client=client, root_grid=root_grid, evidence_plan=PLAN,
        artifact_root=artifact_root, round_number=1,
    )
    assert [c["role"] for c in client.calls] == ["navigator", "navigator_detail"]
    assert len(patches) == 9, len(patches)
    prov = json.loads(
        (artifact_root / "round_1_acquisition_events.json").read_text()
    )
    assert prov["events_mode"] is True
    # x20 crops under different parents must carry DIFFERENT event queries
    peq = prov["patch_event_query"]
    x20_queries = {peq[p.patch_id] for p in patches if int(p.magnification) == 20}
    assert len(x20_queries) >= 2, f"anchor-specific addresses collapsed: {x20_queries}"
    # x5 crops all carry the coarse event query
    x5_queries = {peq[p.patch_id] for p in patches if int(p.magnification) == 5}
    assert x5_queries == {prov["coarse_query"]}, x5_queries
    print(f"MOCK NAV_EVENTS PASS: {len(x20_queries)} distinct anchor addresses, "
          f"x5 share coarse event")


if __name__ == "__main__":
    main()
    main_events()
