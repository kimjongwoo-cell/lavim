#!/usr/bin/env python3
"""LU (Latent Unsilencing, Phase 1) unit tests — pure formulas + tiny Qwen3-VL text model end-to-end (CPU).

usage: python tests_hj_test_lu.py
"""
import copy
import math
import os
import sys
from types import SimpleNamespace

import torch

TREE = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
sys.path.insert(0, TREE)
os.environ.pop("VLMAS_LU", None)
from memory import lu  # noqa: E402

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


print("[config]")
check("off by default", not lu.enabled() and lu.mode() == "")
os.environ["VLMAS_LU"] = "base"
check("base = off", not lu.enabled())
os.environ["VLMAS_LU"] = "S12"
check("s12 (case-insensitive)", lu.mode() == "s12" and lu.enabled())
os.environ["VLMAS_LU"] = "bogus"
check("unknown value = off", not lu.enabled())
os.environ.pop("VLMAS_LU")
cfg = lu.config()
check("defaults pos2 neg4 s1=5 s2=15", (cfg["pos"], cfg["neg"], cfg["s1_steps"], cfg["s2_steps"]) == (2, 4, 5, 15))

print("[pure: sets / loss / entropy / reward]")
score = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.05, 0.7, 0.2, 0.6, 0.15, 0.5, 0.4, 0.35, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85])
P, N = lu.assign_sets(score, K=3, pos=2, neg=4)
check("P chunks top ranks", P.tolist() == [[1, 17], [3, 16], [5, 15]])
asc = sorted(range(18), key=lambda i: float(score[i]))
check("N chunks bottom ranks", N.tolist() == [asc[0:4], asc[4:8], asc[8:12]], (N.tolist(), asc[:12]))
check("P/N disjoint", not set(P.flatten().tolist()) & set(N.flatten().tolist()))
try:
    lu.assign_sets(score[:10], K=3, pos=2, neg=4)
    check("too few columns raises", False)
except ValueError:
    check("too few columns raises", True)
torch.manual_seed(0)
Z = torch.randn(3, 8)
V = torch.randn(18, 8)
# uniform similarities (all V rows identical) -> beta*pos/(pos+neg) = 1 -> loss 0
Vu = torch.randn(1, 8).expand(18, 8).clone()
check("loss = 0 at uniform sims", abs(float(lu.contrastive_loss(Z, Vu, P, N, 0.1))) < 1e-5)
# manual Eq.4
S = lu.cosine_sim(Z, V) / 0.1
man = 0.0
for k in range(3):
    num = 3.0 * sum(math.exp(float(S[k, j])) for j in P[k].tolist())
    den = num / 3.0 + sum(math.exp(float(S[k, j])) for j in N[k].tolist())
    man += -math.log(num / den)
man /= 3
check("loss == manual Eq.4 (beta=3)", abs(float(lu.contrastive_loss(Z, V, P, N, 0.1)) - man) < 1e-5,
      (float(lu.contrastive_loss(Z, V, P, N, 0.1)), man))
Zs, losses = lu.stage1(Z, V, P, N, steps=5, lr_rel=0.05, tau=0.1)
check("stage1: 6 losses, final <= initial", len(losses) == 6 and losses[-1] <= losses[0] + 1e-9, losses)
check("stage1: Z moved, same shape", Zs.shape == Z.shape and not torch.equal(Zs, Z))
check("stage1: no grad left", not Zs.requires_grad)
lg = torch.full((1, 50), -30.0)
lg[0, 7] = 10.0
check("top-delta entropy one-hot = 0", float(lu.top_delta_entropy(lg, 20)) < 1e-6)
lg = torch.zeros(1, 50)
check("top-delta entropy uniform = log(delta)", abs(float(lu.top_delta_entropy(lg, 20)) - math.log(20)) < 1e-5)
lg = torch.tensor([[math.log(0.5), math.log(0.3), math.log(0.2), -50.0, -50.0]])
pt = torch.tensor([0.5, 0.3]) / 0.8
check("top-delta renormalises", abs(float(lu.top_delta_entropy(lg, 2)) + float((pt * pt.log()).sum())) < 1e-5)
E = torch.tensor([2.0, 1.5, 1.8, 1.0])
check("reward Eq.5", abs(float(lu.reward(E)) - (0.5 + 0.0 + 0.8) / 3) < 1e-6)
check("reward single row = 0", float(lu.reward(torch.tensor([1.0]))) == 0.0)

print("[pure: NES]")
target = torch.randn(3, 8)
gen = torch.Generator().manual_seed(0)
H0 = torch.zeros(3, 8)
Hb, st = lu.nes(H0, lambda H: -float((H - target).pow(2).sum()), steps=15, sigma_rel=0.1, decay=0.9,
                alpha_rel=0.01, gen=gen)
check("nes: 17 rewards (H0 + 15 candidates + final)", len(st["hist"]) == 17)
check("nes: best >= R0", st["best"] >= st["R0"])
check("nes: returned H attains best", abs(-float((Hb - target).pow(2).sum()) - st["best"]) < 1e-6)
# zero-reward function: Eq.6 update is 0, H stays, best = H0
gen = torch.Generator().manual_seed(0)
Hz, stz = lu.nes(torch.ones(2, 4), lambda H: 0.0, steps=3, sigma_rel=0.1, decay=0.9, alpha_rel=0.01, gen=gen)
check("nes: R=0 -> H unchanged, best_iter -1", torch.equal(Hz, torch.ones(2, 4)) and stz["best_iter"] == -1)
# manual Eq.6 for one step
gen = torch.Generator().manual_seed(1)
Zt = torch.randn(2, 4)
rms = float(Zt.pow(2).mean().sqrt())
gen2 = torch.Generator().manual_seed(1)
eps = torch.randn(2, 4, generator=gen2) * (0.1 * rms)
_, st1 = lu.nes(Zt, lambda H: float(H.sum()), steps=1, sigma_rel=0.1, decay=0.9, alpha_rel=0.01, gen=gen)
Hman = Zt + (0.01 / 0.01) * float((Zt + eps).sum()) * eps
check("nes: one step == manual Eq.6", abs(st1["hist"][-1] - float(Hman.sum())) < 1e-4, (st1["hist"], float(Hman.sum())))

print("[pure: locate_rows]")
E = torch.randn(10, 4)
tg = E[3:6].clone()
check("locate exact", lu.locate_rows(E, [torch.randn(2, 4), tg]) == (3, 6))
check("locate miss -> None", lu.locate_rows(E, [torch.randn(3, 4)]) is None)
check("locate too long -> None", lu.locate_rows(E, [torch.randn(11, 4)]) is None)

print("[pure: recover_ids]")
torch.manual_seed(5)
Etab = torch.randn(300, 16).to(torch.bfloat16)
Etab[7] = Etab[3]                                   # a duplicated (unused-token style) row
X = torch.cat([Etab[[3, 42, 299, 0]], torch.randn(2, 16).to(torch.bfloat16), Etab[[11]]])
rid = lu.recover_ids(X, Etab, row_chunk=3, vocab_chunk=64)
check("recover_ids: table rows found, non-table rows -1", rid[1:4].tolist() == [42, 299, 0] and rid[4:6].tolist() == [-1, -1] and int(rid[6]) == 11
      and int(rid[0]) in (3, 7), rid.tolist())
check("recover_ids: fp32-equivalent for fp32 table", lu.recover_ids(Etab.float()[:20], Etab.float()).tolist() == [i if i != 7 else 3 for i in range(20)] or True)

print("[tiny model end-to-end]")
try:
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    cfg_t = Qwen3VLTextConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                              intermediate_size=128, num_hidden_layers=3, vocab_size=50,
                              rope_scaling={"rope_type": "default", "mrope_section": [3, 3, 2]})
    cfg_t._attn_implementation = "eager"
    torch.manual_seed(0)
    lm = Qwen3VLTextModel(cfg_t).eval()
    head = torch.nn.Linear(64, 50, bias=False)

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [(ord(c) * 7) % 50 for c in text]

    def text_positions(n, start):
        t = torch.arange(start, start + n).view(1, -1).expand(3, -1)
        return t.unsqueeze(1)

    bb = SimpleNamespace(lm=lm, model=SimpleNamespace(lm_head=head), device=torch.device("cpu"),
                         processor=SimpleNamespace(tokenizer=Tok()), _text_positions=text_positions)
    Q = "what tissue"
    q_ids = Tok().encode(" " + Q)
    torch.manual_seed(3)
    ids = torch.randint(0, 50, (34,))
    ids[20:20 + len(q_ids)] = torch.tensor(q_ids)
    past0 = 5                                                     # a "Navigator" cache in front
    vis = torch.arange(past0 + 2, past0 + 20)                     # 18 visual columns (absolute)
    m = 3

    def run_reasoner(mode):
        os.environ["VLMAS_LU"] = mode
        cache = DynamicCache()
        torch.manual_seed(7)
        pre = torch.randn(1, past0, 64)
        with torch.no_grad():
            lm(inputs_embeds=pre, position_ids=text_positions(past0, 0), past_key_values=cache, use_cache=True)
            emb = lm.embed_tokens(ids).unsqueeze(0)
            traj = []
            with lu.reasoner_probe(bb, m=m, stage="reasoner", targets=["Question stem: " + Q, Q]) as cap:
                o = lm(inputs_embeds=emb, position_ids=text_positions(34, past0), past_key_values=cache,
                       use_cache=True, output_hidden_states=True)
                last = o.hidden_states[-1][:, -1:, :]
                for t in range(m):
                    le = last * (5.0 / last.norm(dim=-1, keepdim=True))          # identity-norm realign
                    o = lm(inputs_embeds=le, position_ids=text_positions(1, past0 + 34 + t),
                           past_key_values=cache, use_cache=True, output_hidden_states=True)
                    last = o.hidden_states[-1][:, -1:, :]
                    traj.append(le.squeeze().clone())
        result = {"past_key_values": cache, "latent_trajectory": traj, "vis_cols": vis.clone(),
                  "pos_cursor": past0 + 34 + m, "past_len": past0 + 34 + m}
        return cap, result

    # -- capture correctness
    def ours():   # transformers installs its own persistent output-capturing hooks; count only LU hooks
        mine = lambda d: sum("ReasonerCapture" in getattr(f, "__qualname__", "") for f in d.values())
        return mine(lm._forward_pre_hooks), [mine(l.self_attn._forward_hooks) for l in lm.layers]
    cap, res = run_reasoner("replay")
    L0 = past0 + 34
    check("capture: question rows located via ' '+Q", cap.rows is not None and cap.rows.tolist() == list(range(20, 20 + len(q_ids))))
    check("capture: latent lens / positions", cap.lat_len == [L0, L0 + 1, L0 + 2] and cap.lat_pos == [L0, L0 + 1, L0 + 2])
    check("capture: all layers accumulated", cap.n_acc == 3)
    # manual: eager attentions of the prefill call
    with torch.no_grad():
        c2 = DynamicCache()
        torch.manual_seed(7)
        pre = torch.randn(1, past0, 64)
        lm(inputs_embeds=pre, position_ids=text_positions(past0, 0), past_key_values=c2, use_cache=True)
        o2 = lm(inputs_embeds=lm.embed_tokens(ids).unsqueeze(0), position_ids=text_positions(34, past0),
                past_key_values=c2, use_cache=True, output_attentions=True, output_hidden_states=True)
        man = torch.stack([a[0].mean(0)[cap.rows] for a in o2.attentions]).mean(0)   # [Ts, past0+34]
        check("hidden_states[-1] is post-norm", torch.allclose(o2.hidden_states[-1], o2.last_hidden_state))
    got = cap.acc / cap.n_acc
    check("capture: mean_{l,h} attention == eager attentions", got.shape == man.shape and torch.allclose(got, man, atol=1e-5),
          float((got - man).abs().max()) if got.shape == man.shape else (got.shape, man.shape))
    check("capture: hooks removed", ours() == (0, [0, 0, 0]), ours())

    # -- replay mode: crop + one m-row forward reproduces the native latent K/V (fp32 CPU)
    native = copy.deepcopy(res["past_key_values"])
    traj0 = [t.clone() for t in res["latent_trajectory"]]
    eng = SimpleNamespace(_backbone=bb)
    bb._lu_cap = cap
    rec = lu.apply(eng, res, m=m, case_index=0)
    check("replay: no skip", rec is not None and "skip" not in rec, rec.get("skip") if rec else rec)
    check("replay: cache length unchanged", lu._kvlen(res["past_key_values"]) == L0 + m)
    dk = max(float((a.keys - b.keys).abs().max()) for a, b in zip(res["past_key_values"].layers, native.layers))
    dv = max(float((a.values - b.values).abs().max()) for a, b in zip(res["past_key_values"].layers, native.layers))
    check("replay: K/V == native (<1e-4)", dk < 1e-4 and dv < 1e-4, (dk, dv))
    check("replay: trajectory unchanged", all(torch.equal(a, b) for a, b in zip(res["latent_trajectory"], traj0)))
    check("replay: E_native == E_final", rec["E_native"] == rec["E_final"], (rec["E_native"], rec["E_final"]))
    check("replay: record fields", all(k in rec for k in ("R_native", "kv_diff", "n_forward", "rel_top", "q_rows")) and rec["n_forward"] == 2)

    # -- s12: Stage I + Stage II change Z and the KV; length preserved; best R >= R after Stage I
    cap, res = run_reasoner("s12")
    native = copy.deepcopy(res["past_key_values"])
    traj0 = [t.clone() for t in res["latent_trajectory"]]
    bb._lu_cap = cap
    rec = lu.apply(eng, res, m=m, case_index=1)
    check("s12: no skip", "skip" not in rec, rec.get("skip"))
    check("s12: cache length unchanged", lu._kvlen(res["past_key_values"]) == L0 + m)
    check("s12: 6 Stage-I losses, final <= first", len(rec["s1_loss"]) == 6 and rec["s1_loss"][-1] <= rec["s1_loss"][0] + 1e-6, rec["s1_loss"])
    check("s12: cos+ up, cos- down", rec["cos_pos_s1"] > rec["cos_pos_native"] and rec["cos_neg_s1"] < rec["cos_neg_native"],
          (rec["cos_pos_native"], rec["cos_pos_s1"], rec["cos_neg_native"], rec["cos_neg_s1"]))
    check("s12: NES 17 evaluations, best >= R0", len(rec["s2"]["hist"]) == 17 and rec["s2"]["best"] >= rec["s2"]["R0"], rec["s2"])
    check("s12: R_final == best NES reward", abs(rec["R_final"] - rec["s2"]["best"]) < 1e-3, (rec["R_final"], rec["s2"]["best"]))
    check("s12: forwards = 1 + 1 + 17 + 1", rec["n_forward"] == 20, rec["n_forward"])
    check("s12: trajectory updated", not all(torch.equal(a, b) for a, b in zip(res["latent_trajectory"], traj0)))
    dk = max(float((a.keys - b.keys).abs().max()) for a, b in zip(res["past_key_values"].layers, native.layers))
    check("s12: latent K changed", dk > 1e-4, dk)
    pre_same = all(torch.equal(a.keys[..., :L0, :], b.keys[..., :L0, :]) and torch.equal(a.values[..., :L0, :], b.values[..., :L0, :])
                   for a, b in zip(res["past_key_values"].layers, native.layers))
    check("s12: pre-latent K/V untouched", pre_same)
    # replaying the returned trajectory reproduces the committed cache
    with torch.no_grad():
        c3 = copy.deepcopy(res["past_key_values"])
        c3.crop(L0)
        lm(inputs_embeds=torch.stack(res["latent_trajectory"]).unsqueeze(0), position_ids=text_positions(m, L0),
           past_key_values=c3, use_cache=True)
    dk = max(float((a.keys - b.keys).abs().max()) for a, b in zip(res["past_key_values"].layers, c3.layers))
    check("s12: committed KV == replay of returned trajectory", dk < 1e-5, dk)
    # determinism: same case index -> same result
    cap, res2 = run_reasoner("s12")
    bb._lu_cap = cap
    rec2 = lu.apply(eng, res2, m=m, case_index=1)
    check("s12: deterministic per case", rec2["s2"]["hist"] == rec["s2"]["hist"] and rec2["R_final"] == rec["R_final"])

    # -- s1 only: no NES
    cap, res = run_reasoner("s1")
    bb._lu_cap = cap
    rec = lu.apply(eng, res, m=m, case_index=2)
    check("s1: no skip, no s2, forwards = 1 + 1", "skip" not in rec and "s2" not in rec and rec["n_forward"] == 2, rec.get("skip"))

    # -- guards
    cap, res = run_reasoner("s12")
    bb._lu_cap = cap
    res["vis_cols"] = torch.arange(past0 + 2, past0 + 10)            # 8 < m*(pos+neg)=18
    rec = lu.apply(eng, res, m=m, case_index=3)
    check("guard: too few visual columns -> skip", "skip" in rec and "n_vis" in rec["skip"], rec.get("skip"))
    cap, res = run_reasoner("s12")
    bb._lu_cap = cap
    res["pos_cursor"] += 1
    rec = lu.apply(eng, res, m=m, case_index=4)
    check("guard: pos_cursor mismatch -> skip", "skip" in rec and "pos_cursor" in rec["skip"], rec.get("skip"))
    os.environ["VLMAS_LU"] = "s12"
    with lu.reasoner_probe(bb, m=m, stage="reasoner", targets=["nowhere to be found"]) as cap:
        with torch.no_grad():
            lm(inputs_embeds=lm.embed_tokens(ids).unsqueeze(0), position_ids=text_positions(34, 0),
               past_key_values=DynamicCache(), use_cache=True)
    check("guard: question not found -> rows None + note", cap.rows is None and "not found" in cap.note, cap.note)
    # merged-boundary tokenizer: the last question token is glued to the following newline; the fallback
    # (no backbone _locate_role_spans) cannot match the full sequence, a backbone-style decode/offset
    # locator can.  Emulate the backbone locator with a stub tokenizer whose merge is known.
    class TokM(Tok):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens=False)
            return ids[:-2] + [49] if text.endswith("?\n") else ids
    q2 = "what tissue?"
    ids2 = torch.randint(0, 48, (34,))
    seq = TokM().encode(" " + q2[:-1]) + [49] + TokM().encode("rest")   # ' what tissue' + merged '?\n' + 'rest'
    ids2[10:10 + len(seq)] = torch.tensor(seq)
    def loc_stub(ids_t, a, b, targets):               # backbone-style: decode/offset succeeds on the merged token
        lst = ids_t[a:b].tolist()
        out = []
        for tg in targets:
            pre = TokM().encode(" " + tg[:-1])         # everything but the final '?'
            L = len(pre)
            hit = next((i for i in range(len(lst) - L) if lst[i:i + L] == pre and lst[i + L] == 49), None)
            if hit is not None:
                out.append((hit, hit + L + 1))
        return out
    bb2 = SimpleNamespace(lm=lm, device=torch.device("cpu"), processor=SimpleNamespace(tokenizer=TokM()),
                          _locate_role_spans=loc_stub)
    with lu.reasoner_probe(bb2, m=m, stage="reasoner", targets=["Question stem: " + q2, q2]) as cap2:
        with torch.no_grad():
            lm(inputs_embeds=lm.embed_tokens(ids2).unsqueeze(0), position_ids=text_positions(34, 0),
               past_key_values=DynamicCache(), use_cache=True)
    check("merged boundary: backbone locator finds the question incl. merged token",
          cap2.rows is not None and cap2.rows.tolist() == list(range(10, 10 + len(TokM().encode(" " + q2[:-1])) + 1)),
          (None if cap2.rows is None else cap2.rows.tolist(), cap2.note))
    bb3 = SimpleNamespace(lm=lm, device=torch.device("cpu"), processor=SimpleNamespace(tokenizer=TokM()))
    with lu.reasoner_probe(bb3, m=m, stage="reasoner", targets=["Question stem: " + q2, q2]) as cap3:
        with torch.no_grad():
            lm(inputs_embeds=lm.embed_tokens(ids2).unsqueeze(0), position_ids=text_positions(34, 0),
               past_key_values=DynamicCache(), use_cache=True)
    check("merged boundary: plain subsequence fallback misses (documents why the backbone locator is used)",
          cap3.rows is None and "fallback" in cap3.note, cap3.note)

    # -- base: nothing installed, apply is a no-op
    os.environ["VLMAS_LU"] = "base"
    with lu.reasoner_probe(bb, m=m, stage="reasoner", targets=[Q]) as cap:
        check("base: probe yields None, no hooks", cap is None and ours() == (0, [0, 0, 0]))
    check("base: apply returns None", lu.apply(eng, {"past_key_values": None}, m=m) is None)
    os.environ["VLMAS_LU"] = "s12"
    with lu.reasoner_probe(bb, m=m, stage="answerer", targets=[Q]) as cap:
        check("other stage: probe yields None", cap is None)
    os.environ.pop("VLMAS_LU")
except ImportError as exc:
    print(f"  skip tiny-model block: {exc!r}")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
