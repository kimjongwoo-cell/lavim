"""Wide coarse pool: PLIP must be able to pick the position inside a root."""
import os, sys
sys.path.insert(0, "/home/users/whddn12316/wsi_latent_0915_decode_hj")
from PIL import Image
from vision_text_mas.wsi_search_tool import PlipSearchTool, TissueSearchTool
from vision_text_mas.onepass_navigation_roots import prepare_root_grid


class FakeSlide:
    def __init__(self): self._w, self._h = 24_576, 16_384
    @property
    def dimensions(self): return (self._w, self._h)
    @property
    def level_downsamples(self): return (1.0, 32.0)
    def get_thumbnail(self, size): return Image.new("RGB", size, (200, 120, 160))
    def get_best_level_for_downsample(self, d): return 1 if d >= 32.0 else 0
    def read_region(self, loc, level, size): return Image.new("RGB", size, (200, 120, 160))


class StubPlip(PlipSearchTool):
    """Avoid loading real weights; rank by a caller-supplied order."""
    def __init__(self, *, root_grid, order_fn):
        TissueSearchTool.__init__(self, root_grid=root_grid)
        self._order_fn = order_fn
    def _plip_rank(self, query, images): return self._order_fn(len(images))


slide = FakeSlide()
grid = prepare_root_grid(slide, thumbnail=slide.get_thumbnail((1024, 683)))

os.environ["VLMAS_NAV_PLIP_PER_ROOT"] = "1"
narrow = StubPlip(root_grid=grid, order_fn=lambda n: list(range(n)))
pool_narrow = narrow._wide_coarse_pool(slide, per_root=1)
got_narrow = narrow.search_coarse(slide, query="q", top_k=3)

os.environ["VLMAS_NAV_PLIP_PER_ROOT"] = "4"
wide = StubPlip(root_grid=grid, order_fn=lambda n: list(reversed(range(n))))
pool_wide = wide._wide_coarse_pool(slide, per_root=4)
got_wide = wide.search_coarse(slide, query="q", top_k=3)

assert len(pool_wide) > len(pool_narrow), (len(pool_wide), len(pool_narrow))
roots_wide = {o.root_id for o in pool_wide}
per_root = {r: sum(1 for o in pool_wide if o.root_id == r) for r in roots_wide}
assert max(per_root.values()) > 1, per_root
assert len(got_wide) == 3 and len({o.box for o in got_wide}) == 3
# reversed ranking must land on different crops than the tissue-argmax order
assert {o.crop_id for o in got_wide} != {o.crop_id for o in got_narrow}
# same root can now contribute more than one position
multi = any(v > 1 for v in per_root.values())
print(f"WIDE POOL PASS: pool {len(pool_narrow)} -> {len(pool_wide)}, "
      f"max/root={max(per_root.values())}, within-root choice={multi}, "
      f"picks differ from tissue-argmax")
