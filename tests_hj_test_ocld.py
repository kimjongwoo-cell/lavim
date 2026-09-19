"""OCLD unit tests (CPU, tiny random Qwen3). python3 tests_hj_test_ocld.py

Reference = independent full-sequence recomputation with an explicit 4D boolean mask built from the
page's rules (no cache, no hook). Candidate = the backbone-style cached latent loop with the OCLD hook.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory import ocld  # noqa: E402

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL", name, extra)


def tiny_model(attn="sdpa"):
    from transformers import Qwen3Config, Qwen3Model
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=48, intermediate_size=96, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=12,
                      max_position_embeddings=256, attn_implementation=attn)
    m = Qwen3Model(cfg).eval().to(torch.float64)
    return m


class BB:
    def __init__(self, lm, meta):
        self.lm = lm
        self._visual_meta = meta


def cached_loop(bb, prefix, m):
    """backbone-style: prefill, then m single-row latent steps with le = last hidden (realign=identity)."""
    o = bb.lm(inputs_embeds=prefix, use_cache=True, output_hidden_states=True)
    kv, last = o.past_key_values, o.last_hidden_state[:, -1:, :]
    P = prefix.shape[1]
    rows = []
    for step in range(m):
        o = bb.lm(inputs_embeds=last, position_ids=torch.tensor([[P + step]]), past_key_values=kv,
                  use_cache=True, output_hidden_states=True)
        kv, last = o.past_key_values, o.last_hidden_state[:, -1:, :]
        rows.append(last[0, 0].clone())
    return torch.stack(rows), kv, o


def reference(lm, prefix, groups, perm, integrate=True):
    """Full recompute. Latent input for every OCLD step = e_R = last prefill hidden."""
    P, d = prefix.shape[1], prefix.shape[2]
    o = lm(inputs_embeds=prefix)
    e_R = o.last_hidden_state[:, -1:, :]
    G = len(groups)
    M = G + (1 if integrate else 0)
    x = torch.cat([prefix] + [e_R] * M, dim=1)
    T = P + M
    allow = torch.tril(torch.ones(T, T, dtype=torch.bool))
    vis_all = torch.cat(groups) if groups else torch.empty(0, dtype=torch.long)
    for t in range(G):
        r = P + t
        allow[r, vis_all] = False
        allow[r, groups[perm[t]]] = True
        allow[r, P:P + t] = False
        allow[r, r] = True
    mask = torch.zeros(1, 1, T, T, dtype=torch.float64)
    mask[0, 0][~allow] = torch.finfo(torch.float64).min
    o2 = lm(inputs_embeds=x, position_ids=torch.arange(T).unsqueeze(0), attention_mask=mask)
    return o2.last_hidden_state[0, P:P + M]


def run_case(attn, P=20, G=3, sizes=(4, 3, 5), vis_start=2, shuffle=False, drop_group=None):
    lm = tiny_model(attn)
    torch.manual_seed(1)
    prefix = torch.randn(1, P, 48, dtype=torch.float64)
    cols, obs = [], []
    c = vis_start
    for g, s in enumerate(sizes):
        for _ in range(s):
            if drop_group is None or g != drop_group:
                cols.append(c)
                obs.append(g)
            c += 1
    meta = {"abs_cols": torch.tensor(cols), "obs_id": torch.tensor(obs)}
    bb = BB(lm, meta)
    os.environ["VLMAS_OCLD"] = "1"
    os.environ["VLMAS_OCLD_ORDER"] = "shuffle" if shuffle else "acq"
    M = ocld.steps_for(G)
    with ocld.reasoner_ocld(bb, m=M, n_obs=G, stage="reasoner", seed=7) as ctl:
        rows, kv, _ = cached_loop(bb, prefix, M)
    groups = ocld.groups_from_meta(meta, G)
    ref = reference(lm, prefix, groups, ctl.perm)
    return rows, ref, ctl, kv


# ---------------------------------------------------------------- pure helpers
check("steps_for", ocld.steps_for(12) == 13)
mk = ocld.local_mask(10, torch.tensor([1, 2, 3, 4]), torch.tensor([3, 4]), 6, 2)
check("mask shape", tuple(mk.shape) == (1, 10))
check("mask values", mk[0].tolist() == [1, 0, 0, 1, 1, 1, 0, 0, 1, 1], mk.tolist())
mk0 = ocld.local_mask(8, torch.tensor([1, 2]), torch.tensor([1, 2]), 6, 0)
check("mask t0 own group", mk0[0].tolist() == [1] * 8)
os.environ["VLMAS_OCLD_ORDER"] = "acq"
check("order acq", ocld.order(5, 3) == [0, 1, 2, 3, 4])
os.environ["VLMAS_OCLD_ORDER"] = "shuffle"
p1, p2 = ocld.order(6, 3), ocld.order(6, 3)
check("order shuffle deterministic perm", p1 == p2 and sorted(p1) == list(range(6)))
os.environ.pop("VLMAS_OCLD", None)
check("mode off", ocld.mode() == "")
os.environ["VLMAS_OCLD"] = "native"
check("mode native", ocld.mode() == "native")
os.environ["VLMAS_OCLD"] = "bogus"
check("mode bogus off", ocld.mode() == "")

# ---------------------------------------------------------------- off / native = no hook
os.environ.pop("VLMAS_OCLD", None)
lm = tiny_model()
bb = BB(lm, {"abs_cols": torch.arange(2, 8), "obs_id": torch.tensor([0, 0, 1, 1, 2, 2])})
prefix = torch.randn(1, 12, 48, dtype=torch.float64)
r_plain, _, _ = cached_loop(bb, prefix, 4)
with ocld.reasoner_ocld(bb, m=4, n_obs=3, stage="reasoner") as ctl:
    r_off, _, _ = cached_loop(bb, prefix, 4)
check("off yields None", ctl is None)
check("off identical", torch.equal(r_plain, r_off))
os.environ["VLMAS_OCLD"] = "native"
with ocld.reasoner_ocld(bb, m=4, n_obs=3, stage="reasoner") as ctl:
    r_nat, _, _ = cached_loop(bb, prefix, 4)
check("native no hook", ctl is None and torch.equal(r_plain, r_nat))
os.environ["VLMAS_OCLD"] = "1"
with ocld.reasoner_ocld(bb, m=4, n_obs=3, stage="navigator") as ctl:
    r_nav, _, _ = cached_loop(bb, prefix, 4)
check("other stage no hook", ctl is None and torch.equal(r_plain, r_nav))

# ---------------------------------------------------------------- OCLD vs independent reference
for attn in ("sdpa", "eager"):
    for kw in ({}, {"shuffle": True}, {"G": 4, "sizes": (2, 6, 1, 3), "vis_start": 5, "P": 24},
               {"drop_group": 1}):
        G = kw.get("G", 3)
        rows, ref, ctl, kv = run_case(attn, **kw)
        tag = f"{attn} {kw}"
        check(f"no skip {tag}", ctl.skip == "", ctl.skip)
        check(f"steps {tag}", ctl.t == G + 1 and len(ctl.rows) == G + 1)
        err = (rows - ref).abs().max().item()
        check(f"cached==reference {tag}", err < 1e-9, f"maxerr {err}")
        check(f"cache len {tag}", ocld._cache_len(kv) == kw.get("P", 20) + G + 1)
        check(f"order perm {tag}", sorted(ctl.perm) == list(range(G)))

# sanity: the mask actually matters (OCLD differs from plain seed-reuse without masks)
lm = tiny_model()
torch.manual_seed(1)
prefix = torch.randn(1, 20, 48, dtype=torch.float64)
groups = [torch.tensor([2, 3, 4, 5]), torch.tensor([6, 7, 8]), torch.tensor([9, 10, 11, 12, 13])]
ref_mask = reference(lm, prefix, groups, [0, 1, 2])
T = 24
x = torch.cat([prefix] + [lm(inputs_embeds=prefix).last_hidden_state[:, -1:, :]] * 4, dim=1)
nomask = lm(inputs_embeds=x, position_ids=torch.arange(T).unsqueeze(0)).last_hidden_state[0, 20:24]
check("mask changes local rows", (ref_mask[:3] - nomask[:3]).abs().max().item() > 1e-4)
check("integration row sees z_g (differs only via cache)", (ref_mask[3] - nomask[3]).abs().max().item() > 1e-6)

# ---------------------------------------------------------------- skip paths
os.environ["VLMAS_OCLD"] = "1"
bbn = BB(lm, None)
with ocld.reasoner_ocld(bbn, m=4, n_obs=3, stage="reasoner") as ctl:
    r_skip, _, _ = cached_loop(bbn, prefix, 4)
check("no meta -> skip native", "no visual provenance" in ctl.skip)
bb_plain = BB(lm, None)
r_ref_plain, _, _ = cached_loop(bb_plain, prefix, 4)
check("no meta -> identical to native", torch.equal(r_skip, r_ref_plain))
bbm = BB(lm, {"abs_cols": torch.tensor([2, 3]), "obs_id": torch.tensor([0, 1])})
with ocld.reasoner_ocld(bbm, m=4, n_obs=3, stage="reasoner") as ctl:
    cached_loop(bbm, prefix, 4)
check("group count mismatch still G groups (empty support)", ctl.skip == "" and len(ctl.groups) == 3
      and ctl.groups[2].numel() == 0, ctl.skip)
bbx = BB(lm, {"abs_cols": torch.tensor([2, 30]), "obs_id": torch.tensor([0, 1])})
with ocld.reasoner_ocld(bbx, m=3, n_obs=2, stage="reasoner") as ctl:
    cached_loop(bbx, prefix, 3)
check("visual col outside prefix -> skip", "not inside" in ctl.skip)
check("summary text", "SKIP" in ctl.summary())

print(f"OCLD tests {PASS}/{PASS + FAIL}")
sys.exit(1 if FAIL else 0)
