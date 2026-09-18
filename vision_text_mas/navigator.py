"""Public composition of root/x5 and zoom navigation responsibilities."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from vision_text_mas.contracts import (
    EvidenceRound,
    NavigationAction,
    NewRegionDecision,
    RoleCall,
    ZoomDetailDecision,
)
from vision_text_mas.evidence import EvidenceMemory
from vision_text_mas.navigation_render import SlideLike
from vision_text_mas.navigation_contracts import EvidencePlan, SelectionTrace
from vision_text_mas.navigation_session import DiscreteSelector, SelectionSession
from vision_text_mas.navigator_regions import RegionNavigator
from vision_text_mas.navigator_zoom import ZoomNavigator
from vision_text_mas.navigator_multiscale import MultiscaleComposer


class Navigator:
    """Retrieve exactly five patches without diversity or fallback rules."""

    def __init__(self, *, selector: DiscreteSelector, artifact_root: Path) -> None:
        session = SelectionSession(selector)
        self._session = session
        self._regions = RegionNavigator(session=session, artifact_root=artifact_root)
        self._zoom = ZoomNavigator(session=session, artifact_root=artifact_root)
        self._multiscale = MultiscaleComposer(self._zoom)
        self._visited_roots: set[int] = set()

    @property
    def records(self) -> tuple[RoleCall, ...]:
        """Return every validated Navigator call in execution order."""
        return self._session.records

    @property
    def traces(self) -> tuple[SelectionTrace, ...]:
        """Return controller filtering traces in execution order."""
        return self._session.traces

    def initial(
        self,
        slide: SlideLike,
        *,
        thumbnail: Image.Image,
        evidence_plan: EvidencePlan,
        round_number: int,
    ) -> EvidenceRound:
        """Retrieve five exact-x5 patches from unrestricted eligible roots."""
        scout = self._regions.run(
            slide,
            thumbnail=thumbnail,
            evidence_plan=evidence_plan,
            round_number=round_number,
            action=NavigationAction.INITIAL,
            excluded_roots=frozenset(),
        )
        self._visited_roots.update(int(patch.root_id) for patch in scout.patches)
        return self._multiscale.compose(
            slide,
            scout=scout,
            evidence_plan=evidence_plan,
        )

    def new_region(
        self,
        slide: SlideLike,
        *,
        evidence_plan: EvidencePlan,
        request: NewRegionDecision,
        round_number: int,
        evidence: EvidenceMemory,
    ) -> EvidenceRound:
        """Retrieve five x5 patches from roots never visited in prior rounds."""
        visited = frozenset(
            (*self._visited_roots, *(int(patch.root_id) for patch in evidence.patches))
        )
        scout = self._regions.run(
            slide,
            thumbnail=None,
            evidence_plan=evidence_plan,
            round_number=round_number,
            action=NavigationAction.NEW_REGION,
            excluded_roots=visited,
        )
        self._visited_roots.update(int(patch.root_id) for patch in scout.patches)
        return self._multiscale.compose(
            slide,
            scout=scout,
            evidence_plan=evidence_plan,
        )

    def zoom(
        self,
        slide: SlideLike,
        *,
        evidence_plan: EvidencePlan,
        request: ZoomDetailDecision,
        round_number: int,
        evidence: EvidenceMemory,
    ) -> EvidenceRound:
        """Retrieve five x20/x40 patches inside one original x5 anchor."""
        return self._zoom.run(
            slide,
            evidence_plan=evidence_plan,
            request=request,
            round_number=round_number,
            evidence=evidence,
        )
