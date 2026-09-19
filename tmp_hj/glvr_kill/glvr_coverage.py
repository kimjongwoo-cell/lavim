"""Coverage-driven Reasoner latent steps (Order 88 §9.4 candidate 1).

Phase A showed that the latent trajectory collapses whatever the grounding target is
(F0 ~= F1 ~= F2): grounding moves the latent INPUT but not WHERE each latent step looks.
This module acts on the attention itself. While the Reasoner writes its m latent steps,
every decoder layer adds a per-head bias to the visual columns,

    s_{k,j} <- s_{k,j} - beta * c_{k-1,j} / max_j' p_{1,j'},      c_{k,j} = sum_{t<=k} p_{t,j},

where p_{t,j} is the (biased) attention step t actually paid to visual column j in that
layer and head. Raw per-token attention on ~10^3 visual columns is ~10^-3, far below logit
scale, so the coverage is divided by the first step's per-head peak: beta is in logit
units, and a state read at that peak on two steps loses 2*beta -- the scale is fixed, so
the penalty keeps growing instead of cycling between the two most-read states. A visual state that earlier steps already read becomes less attractive to
the next step, so the steps have to spread over the evidence -- a coverage penalty.
beta = VLMAS_GLVR_COV; 0 / unset leaves the Reasoner untouched.

The same hooks record, per layer and head, what each step read from the visual columns:
rho_{V,k} (visual mass), u_{V,k} (the normalised visual read) and the visual route. These
are written to bb._glvr in the format glvr_diag expects, so the Answerer-side diagnostic and
the relay run unchanged -- with the trajectory taken from the generation itself, not from a
replay (a replay would recompute the latent rows without the coverage bias).

Only rows == 1 calls inside LatentQwenEngine.append(stage="reasoner") are touched, i.e. the
Reasoner's latent steps; its image/prompt prefill and every other role run natively.
"""
from __future__ import annotations

import contextlib
import os

import torch


def beta() -> float:
    try:
        return float(os.environ.get("VLMAS_GLVR_COV", "0") or 0.0)
    except ValueError:
        return 0.0


def _norm(c: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Cumulative coverage in units of the first step's per-head peak attention."""
    return c / scale.to(c.device).clamp_min(1e-12)


class CoverageState:
    def __init__(self, bb, vis: torch.Tensor, beta_: float):
        self.bb = bb
        self.vis = vis
        self.beta = float(beta_)
        self.cov = {}          # li -> [H, Nv] cumulative attention on the visual columns
        self.scale = {}        # li -> [H, 1] first step's peak visual attention
        self.rho = {}          # li -> list of [H]
        self.u = {}            # li -> list of [H, D]
        self.route = {}        # li -> list of [Nv] (head mean)
        self.lat_cols = []     # absolute column of each latent step (layer 0 decides)
        self.note = None

    # ---- bias ------------------------------------------------------------------------
    def pre(self, li: int):
        def hook(module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            cache = kwargs.get("past_key_values")
            if hs is None or cache is None or self.vis is None or int(hs.shape[1]) != 1 or self.beta == 0.0:
                return None
            c = self.cov.get(li)
            if c is None:
                return None                                   # first step: nothing read yet
            try:
                L = int(cache.layers[li].keys.shape[2]) + 1   # keys after this step's update
                H = int(c.shape[0])
                bias = torch.zeros(1, H, 1, L, device=hs.device, dtype=hs.dtype)
                bias[0, :, 0, self.vis.to(hs.device)] = (-self.beta * _norm(c, self.scale[li])).to(hs.dtype)
                mask = kwargs.get("attention_mask")
                if mask is not None:
                    if mask.dtype == torch.bool:
                        mask = torch.zeros(mask.shape, device=hs.device, dtype=hs.dtype).masked_fill(
                            ~mask, torch.finfo(hs.dtype).min)
                    bias = bias + mask[..., :L].to(hs.dtype)
                kwargs = dict(kwargs)
                kwargs["attention_mask"] = bias
                return args, kwargs
            except Exception as exc:  # noqa: BLE001 - never break the Reasoner
                self.note = f"pre {type(exc).__name__}: {exc}"
                return None
        return hook

    # ---- what this step actually read ---------------------------------------------------
    def post(self, li: int):
        def hook(module, args, kwargs, output):
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
            hs = kwargs.get("hidden_states", args[0] if args else None)
            cache = kwargs.get("past_key_values")
            pe = kwargs.get("position_embeddings")
            if hs is None or cache is None or pe is None or self.vis is None or int(hs.shape[1]) != 1:
                return output
            try:
                with torch.no_grad():
                    keys = cache.layers[li].keys
                    vals = cache.layers[li].values
                    K = (keys[0] if keys.dim() == 4 else keys).float()      # [Hkv, L, D]
                    V = (vals[0] if vals.dim() == 4 else vals).float()
                    L = int(K.shape[1])
                    if li == 0:
                        self.lat_cols.append(L - 1)
                    Hd = module.head_dim
                    G = module.num_key_value_groups
                    q = module.q_norm(module.q_proj(hs).view(1, 1, -1, Hd)).transpose(1, 2)
                    cos, sin = pe
                    q, _ = apply_rotary_pos_emb(q, torch.zeros_like(q), cos, sin)
                    H = int(q.shape[1])
                    Hkv = H // G
                    qg = q[0, :, 0, :].float().reshape(Hkv, G, Hd)               # [Hkv, G, D]
                    s = torch.einsum("kgd,kld->kgl", qg, K).reshape(H, L) * module.scaling
                    vis = self.vis.to(s.device)
                    c = self.cov.get(li)
                    if c is not None and self.beta != 0.0:
                        s[:, vis] = s[:, vis] - self.beta * _norm(c.to(s.device), self.scale[li])
                    p = torch.softmax(s, dim=-1)                                  # [H, L]
                    pv = p[:, vis]                                                # [H, Nv]
                    rho = pv.sum(-1)                                              # [H]
                    Vv = V[:, vis, :].repeat_interleave(G, dim=0)                 # [H, Nv, D]
                    u = torch.einsum("hn,hnd->hd", pv, Vv) / rho.clamp_min(1e-12).unsqueeze(-1)
                    if c is None:
                        self.scale[li] = pv.amax(-1, keepdim=True).detach()
                    self.cov[li] = pv if c is None else c.to(pv.device) + pv
                    self.rho.setdefault(li, []).append(rho.cpu())
                    self.u.setdefault(li, []).append(u.cpu())
                    self.route.setdefault(li, []).append(
                        (pv / rho.clamp_min(1e-12).unsqueeze(-1)).mean(0).cpu())
            except Exception as exc:  # noqa: BLE001
                self.note = f"post {type(exc).__name__}: {exc}"
            return output
        return hook

    def to_glvr(self, case_index: int) -> dict | None:
        """The record glvr_diag reads from bb._glvr (same keys as the replay capture)."""
        layers = sorted(self.u)
        if not layers or not self.lat_cols:
            return None
        m = len(self.lat_cols)
        if any(len(self.u[li]) != m for li in layers):
            self.note = (self.note or "") + f" uneven steps {[len(self.u[li]) for li in layers]}"
            return None
        U = [torch.stack(self.u[li], dim=1) for li in layers]                  # [H, m, D]
        rho = torch.stack([torch.stack(self.rho[li], dim=1).mean(0) for li in layers])   # [L, m]
        route = torch.stack([torch.stack(self.route[li], dim=0) for li in layers])       # [L, m, Nv]
        lat = torch.tensor(self.lat_cols, dtype=torch.long)
        return {"case": int(case_index), "m": m, "n_vis": int(self.vis.numel()),
                "vis": self.vis.cpu(), "lat": lat, "pre_len": int(lat.min()),
                "U": U, "rho": rho, "route": route, "n_layers": len(layers),
                "note": self.note, "source": f"coverage beta={self.beta}"}


@contextlib.contextmanager
def install(bb, vis: torch.Tensor, beta_: float):
    st = CoverageState(bb, vis, beta_)
    handles = []
    for li, layer in enumerate(bb.lm.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(st.pre(li), with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(st.post(li), with_kwargs=True))
    try:
        yield st
    finally:
        for h in handles:
            h.remove()
