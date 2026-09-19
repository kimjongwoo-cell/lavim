from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _dashboard_module():
    dashboard_path = Path(__file__).parents[2] / "0912" / "live_dashboard.py"
    spec = spec_from_file_location("live_dashboard", dashboard_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collects_independent_x5_and_x20_origins() -> None:
    # Given: the completed corrected Nav3 visual-selection smoke artifact.
    dashboard = _dashboard_module()

    # When: the dashboard builds its patch-origin payload.
    cases = dashboard.collect_nav3_selection_cases()

    # Then: every case exposes four overview and eight detail boxes.
    assert len(cases) == 3
    assert all(len(case.x5_patches) == 4 for case in cases)
    assert all(len(case.x20_patches) == 8 for case in cases)
    assert all(case.thumbnail_url.startswith("/nav3-assets/") for case in cases)
