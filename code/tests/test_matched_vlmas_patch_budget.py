from vision_text_mas.controller_onepass import _onepass_patch_budget


def test_accepts_twelve_patch_matched_vlmas_budget(monkeypatch) -> None:
    # Given: the text-communication baseline is matched to the latent 12-patch run.
    monkeypatch.setenv("MATCHED_VLMAS_PATCH_BUDGET", "12")

    # When: the controller parses its patch budget.
    budget = _onepass_patch_budget()

    # Then: the matched comparison retains all twelve patches.
    assert budget == 12
