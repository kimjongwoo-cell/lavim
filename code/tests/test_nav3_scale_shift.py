from vision_text_mas.geometry import Box, Magnification
from vision_text_mas.onepass_navigation_scale_shift import detail_boxes


def test_builds_x20_grid_inside_x5_patch() -> None:
    # Given: one 4096-pixel x5 field.
    anchor = Box(x=100, y=200, width=4096, height=4096)

    # When: the field is shifted to x20 candidates.
    boxes = detail_boxes(anchor, Magnification.X20)

    # Then: it contains sixteen 1024-pixel fields.
    assert len(boxes) == 16
    assert {box.width for box in boxes} == {1024}
    assert all(anchor.contains(box) for box in boxes)


def test_builds_x40_grid_inside_x20_patch() -> None:
    # Given: one 1024-pixel x20 field.
    anchor = Box(x=100, y=200, width=1024, height=1024)

    # When: the field is shifted to x40 candidates.
    boxes = detail_boxes(anchor, Magnification.X40)

    # Then: it contains four 512-pixel fields.
    assert len(boxes) == 4
    assert {box.width for box in boxes} == {512}
    assert all(anchor.contains(box) for box in boxes)
