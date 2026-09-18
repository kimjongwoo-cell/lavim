"""C2 12.4 Observation-Complete Latent Deliberation (OCLD). Notion 3dcd771b-c9e2-8129.

Env-gated. Off (VLMAS_OCLD unset) = no hooks, byte-identical.

VLMAS_OCLD=1        OCLD. For a Reasoner call over G retained physical observations (C1META
                    `_visual_meta.obs_id`), the Reasoner runs M = G + 1 latent steps:
                      steps 0..G-1  local read z_g   (support-isolated)
                        * input embedding = the Reasoner seed e_R (the step-0 input, page §2.1)
                        * attention may see: every non-visual column, the visual columns of
                          G_g only, and itself; the other supports' visual columns and every
                          earlier local read z_h (h < g) are masked (page §2.2)
                      step G        global integration z_int
                        * input = e_R, full attention (all V, all z_g, H^NV) (page §3)
                    Every z_g and z_int stays in the single append-only cache; nothing is moved,
                    re-RoPE'd or re-weighted, so the Answerer reads raw V + Z + z_int (page §4).
VLMAS_OCLD=native   matched compute (page §5): the Reasoner runs the same M = G + 1 standard
                    all-bank latent steps (native realign chain, no mask, no seed reuse).

VLMAS_OCLD_ORDER=acq (default)  local reads in Navigator acquisition order (obs id ascending).
VLMAS_OCLD_ORDER=shuffle         same supports, seeded permutation of the processing order.

Support ids = C1META obs_id (page §10). Visual columns that C1 dropped simply do not exist; a
support whose every token was dropped still gets its (visual-free) local read so M stays G + 1.
"""
from __future__ import annotations

import contextlib
import os
import random

import torch


def mode() -> str:
    v = os.environ.get("VLMAS_OCLD", "").strip().lower()
    return v if v in ("1", "native") else ""


def steps_for(n_observations: int) -> int:
    """M = G + 1 (page §5)."""
    return int(n_observations) + 1


def order(n_groups: int, seed: int) -> list[int]:
    idx = list(range(int(n_groups)))
    if os.environ.get("VLMAS_OCLD_ORDER", "acq").strip().lower() == "shuffle":
        random.Random(int(seed)).shuffle(idx)
    return idx


def groups_from_meta(meta, n_expected: int) -> list[torch.Tensor]:
    """Absolute visual cache columns per physical observation, indexed by obs id 0..G-1."""
    cols = meta["abs_cols"].detach().cpu().long()
    obs = meta["obs_id"].detach().cpu().long()
    n = max(int(n_expected), int(obs.max()) + 1 if obs.numel() else 0)
    return [cols[obs == g] for g in range(n)]


def local_mask(total_len: int, vis_all: torch.Tensor, vis_g: torch.Tensor,
               latent_start: int, t: int, device=None) -> torch.Tensor:
    """[1, total_len] 0/1 mask for the query row at column total_len-1 (the local read z_t).

    allowed = everything, minus visual columns outside G_g, minus earlier local reads
    latent_start .. latent_start+t-1. The row itself (last column) is always allowed.
    """
    allowed = torch.ones(int(total_len), dtype=torch.long)
    if vis_all.numel():
        allowed[vis_all] = 0
    if vis_g.numel():
        allowed[vis_g] = 1
    if t > 0:
        allowed[int(latent_start):int(latent_start) + int(t)] = 0
    allowed[int(total_len) - 1] = 1
    return allowed.unsqueeze(0).to(device) if device is not None else allowed.unsqueeze(0)


def _cache_len(kv) -> int | None:
    layers = getattr(kv, "layers", None)
    if layers:
        return int(layers[0].keys.shape[-2])
    get = getattr(kv, "get_seq_length", None)
    return int(get()) if callable(get) else None


class ReasonerOCLD:
    """forward-pre-hook on bb.lm: rewrites the first M single-row cached calls of one Reasoner call."""

    def __init__(self, bb, m: int, n_obs: int, seed: int):
        self.bb, self.m, self.n_obs, self.seed = bb, int(m), int(n_obs), int(seed)
        self.t = 0
        self.e_R = None
        self.L0 = None
        self.groups = None
        self.perm = None
        self.vis_all = None
        self.note = ""
        self.skip = ""
        self.rows = []  # per-step bookkeeping for the summary

    def _setup(self):
        meta = getattr(self.bb, "_visual_meta", None)
        if meta is None:
            return f"no visual provenance ({getattr(self.bb, '_visual_meta_note', '')})"
        self.groups = groups_from_meta(meta, self.n_obs)
        if len(self.groups) != self.n_obs:
            return f"obs groups {len(self.groups)} != images {self.n_obs}"
        self.vis_all = meta["abs_cols"].detach().cpu().long()
        if self.vis_all.numel() and int(self.vis_all.max()) >= int(self.L0):
            return "visual columns not inside the pre-latent cache"
        self.perm = order(len(self.groups), self.seed)
        return ""

    def pre(self, module, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        kv = kwargs.get("past_key_values")
        if self.skip or self.t >= self.m or emb is None or emb.dim() != 3 or int(emb.shape[1]) != 1:
            return None
        if kv is None:
            return None
        cur = _cache_len(kv)
        if cur is None:
            self.skip = "cache length unavailable"
            return None
        t = self.t
        self.t += 1
        if t == 0:
            self.e_R = emb.detach().clone()
            self.L0 = cur
            self.skip = self._setup()
            if self.skip:
                return None
        elif cur != self.L0 + t:
            self.skip = f"cache length {cur} != {self.L0 + t} at step {t} (columns moved/evicted)"
            return None
        if kwargs.get("attention_mask") is not None:
            self.skip = "caller already passes an attention_mask"
            return None
        G = len(self.groups)
        new = dict(kwargs)
        new["inputs_embeds"] = self.e_R.to(dtype=emb.dtype, device=emb.device)
        if t < G:
            g = self.perm[t]
            vis_g = self.groups[g]
            new["attention_mask"] = local_mask(cur + 1, self.vis_all, vis_g, self.L0, t, device=emb.device)
            self.rows.append(("local", g, int(vis_g.numel()), int(self.vis_all.numel() - vis_g.numel()), t))
        else:
            self.rows.append(("integrate", -1, int(self.vis_all.numel()), 0, 0))
        return args, new

    def summary(self) -> str:
        if self.skip:
            return f"SKIP ({self.skip}) at step {self.t}"
        loc = [r for r in self.rows if r[0] == "local"]
        integ = [r for r in self.rows if r[0] == "integrate"]
        return (f"G={len(self.groups or [])} M={self.m} steps_seen={self.t} local={len(loc)} "
                f"integrate={len(integ)} L0={self.L0} n_vis={0 if self.vis_all is None else int(self.vis_all.numel())} "
                f"order={self.perm} sizes={[r[2] for r in loc]} seed_reused=1")


@contextlib.contextmanager
def reasoner_ocld(bb, *, m: int, n_obs: int, stage: str, seed: int = 0):
    if mode() != "1" or stage != "reasoner" or int(m) <= 0 or int(n_obs) <= 0:
        yield None
        return
    ctl = ReasonerOCLD(bb, m=m, n_obs=n_obs, seed=seed)
    h = bb.lm.register_forward_pre_hook(ctl.pre, with_kwargs=True)
    try:
        yield ctl
    finally:
        h.remove()
        bb._ocld_summary = ctl.summary()
        print(f"[OCLD] {bb._ocld_summary}", flush=True)
