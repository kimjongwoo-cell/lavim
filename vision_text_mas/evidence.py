"""Append-only evidence memory and deterministic board rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vision_text_mas.contracts import EvidenceRound, PatchId, PatchMetadata
from vision_text_mas.errors import FailureCode, PipelineFailure


BOARD_TILE_SIZE = 256
BOARD_LABEL_HEIGHT = 32
BOARD_ROW_HEIGHT = BOARD_TILE_SIZE + BOARD_LABEL_HEIGHT


@dataclass(frozen=True, slots=True)
class EvidenceMemory:
    """Immutable ordered evidence rounds; append returns a new memory."""

    rounds: tuple[EvidenceRound, ...]

    @classmethod
    def empty(cls) -> EvidenceMemory:
        return cls(rounds=())

    @property
    def patches(self) -> tuple[PatchMetadata, ...]:
        return tuple(patch for evidence_round in self.rounds for patch in evidence_round.patches)

    def append(self, evidence_round: EvidenceRound) -> EvidenceMemory:
        """Append the next sequential five-through-twenty-five-patch round."""
        if len(self.rounds) >= 3:
            raise PipelineFailure(
                code=FailureCode.SELECTION_CONTRACT,
                stage="evidence",
                detail="evidence memory cannot exceed three rounds",
            )
        expected_round = len(self.rounds) + 1
        if evidence_round.round_number != expected_round:
            raise PipelineFailure(
                code=FailureCode.SELECTION_CONTRACT,
                stage="evidence",
                detail=f"expected round {expected_round}, got {evidence_round.round_number}",
            )
        known_ids = {patch.patch_id for patch in self.patches}
        if any(patch.patch_id in known_ids for patch in evidence_round.patches):
            raise PipelineFailure(
                code=FailureCode.SELECTION_CONTRACT,
                stage="evidence",
                detail="patch IDs must remain unique across rounds",
            )
        return EvidenceMemory(rounds=(*self.rounds, evidence_round))

    def find_patch(self, patch_id: PatchId) -> PatchMetadata:
        """Resolve one patch ID or fail the selection contract."""
        for patch in self.patches:
            if patch.patch_id == patch_id:
                return patch
        raise PipelineFailure(
            code=FailureCode.SELECTION_CONTRACT,
            stage="evidence",
            detail=f"unknown patch ID: {patch_id}",
        )


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def render_evidence_board(memory: EvidenceMemory, path: Path) -> Path:
    """Render five labeled patch columns and one row per evidence round."""
    canvas = Image.new(
        "RGB",
        (
            max(len(evidence_round.patches) for evidence_round in memory.rounds)
            * BOARD_TILE_SIZE,
            len(memory.rounds) * BOARD_ROW_HEIGHT,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for row, evidence_round in enumerate(memory.rounds):
        for col, patch in enumerate(evidence_round.patches):
            with Image.open(patch.image_path) as source:
                tile = source.convert("RGB")
                tile.thumbnail((BOARD_TILE_SIZE, BOARD_TILE_SIZE), Image.Resampling.LANCZOS)
            x = col * BOARD_TILE_SIZE + (BOARD_TILE_SIZE - tile.width) // 2
            y0 = row * BOARD_ROW_HEIGHT
            y = y0 + BOARD_LABEL_HEIGHT + (BOARD_TILE_SIZE - tile.height) // 2
            canvas.paste(tile, (x, y))
            label = f"{patch.patch_id} · x{patch.magnification.value}"
            draw.text((col * BOARD_TILE_SIZE + 6, y0 + 6), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path
