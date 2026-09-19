#!/usr/bin/env python3
"""Text-cut unit tests — bookkeeping, drop / filler / diag on a tiny Qwen3-VL text model (CPU), mass hook.

usage: python tests_hj_test_textcut.py
"""
import copy
import os
import sys
from types import SimpleNamespace

import torch

TREE = "/home/users/whddn12316/wsi_latent_0915_decode_hj"
sys.path.insert(0, TREE)
os.environ.pop("VLMAS_TEXTCUT", None)
from memory import textcut as tc  # noqa: E402

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


print("[pure]")
check("off by default", not tc.enabled())
check("runs_of", tc.runs_of([5, 6, 7, 10, 12, 13]) == [(5, 8), (10, 11), (12, 14)])
check("remap_after_drop", tc.remap_after_drop(torch.tensor([0, 4, 9]), torch.tensor([1, 2, 6])).tolist() == [0, 2, 6])
check("filler shuffle keeps multiset", sorted(tc.filler_ids([3, 1, 2, 2], "shuffle", 1, 9)) == [1, 2, 2, 3])
check("filler space", tc.filler_ids([3, 1], "space", 1, 9) == [9, 9])
os.environ["VLMAS_TEXTCUT"] = "drop"
check("roles default", tc.config()["roles"] == ["evidence_planner", "navigator"])
os.environ.pop("VLMAS_TEXTCUT")

print("[tiny model]")
try:
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    cfg = Qwen3VLTextConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            intermediate_size=128, num_hidden_layers=3, vocab_size=50,
                            rope_scaling={"rope_type": "default", "mrope_section": [3, 3, 2]})
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    lm = Qwen3VLTextModel(cfg).eval()

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [(ord(c) * 7) % 50 for c in text]

    def text_positions(n, start):
        return torch.arange(start, start + n).view(1, -1).expand(3, -1).unsqueeze(1)

    bb = SimpleNamespace(lm=lm, device=torch.device("cpu"), processor=SimpleNamespace(tokenizer=Tok()))
    eng = SimpleNamespace(_backbone=bb)
    m = 2
    close_n = len(Tok().encode("<|im_end|>\n"))

    def stage(cache, name, n_text, n_vis, cursor, start):
        """prefill = [text n_text | visual n_vis | text 3], then m latent rows, then the close tokens."""
        ids = torch.randint(0, 50, (n_text + 3,))
        emb_text = lm.embed_tokens(ids)
        vis_emb = torch.randn(n_vis, 64) * 3                        # not table rows -> recovered as -1
        emb = torch.cat([emb_text[:n_text], vis_emb, emb_text[n_text:]]).unsqueeze(0)
        T = emb.shape[1]
        pos = text_positions(T, cursor)
        with torch.no_grad(), tc.stage_probe(bb, name):
            o = lm(inputs_embeds=emb, position_ids=pos, past_key_values=cache, use_cache=True, output_hidden_states=True)
            last = o.hidden_states[-1][:, -1:, :]
            for t in range(m):
                le = last * (5.0 / last.norm(dim=-1, keepdim=True))
                o = lm(inputs_embeds=le, position_ids=text_positions(1, cursor + T + t), past_key_values=cache,
                       use_cache=True, output_hidden_states=True)
                last = o.hidden_states[-1][:, -1:, :]
        past_len = start + T + m
        pc = cursor + T + m
        with torch.no_grad():
            lm(inputs_embeds=lm.embed_tokens(torch.tensor(Tok().encode("<|im_end|>\n"))).unsqueeze(0),
               position_ids=text_positions(close_n, pc), past_key_values=cache, use_cache=True)
        closed = past_len + close_n
        vis_cols = torch.arange(start + n_text, start + n_text + n_vis)
        tc.record_stage(eng, stage=name, start=start, past_len=past_len, latent_steps=m, closed_length=closed,
                        vis_cols=vis_cols, close_thinking=False, pos_cursor=pc)
        return closed, pc + close_n, ids

    def build(mode):
        os.environ["VLMAS_TEXTCUT"] = mode
        torch.manual_seed(5)
        cache = DynamicCache()
        tc.reset(eng)
        L1, c1, ids1 = stage(cache, "evidence_planner", 6, 4, 0, 0)
        L2, c2, ids2 = stage(cache, "navigator", 5, 3, c1, L1)
        L3, c3, ids3 = stage(cache, "reasoner", 7, 5, c2, L2)
        return cache, (L1, L2, L3), c3, (ids1, ids2, ids3)

    cache, (L1, L2, L3), cur, ids_all = build("diag")
    blocks = bb._tc_blocks
    check("bookkeeping: 3 stages", [b["stage"] for b in blocks] == ["evidence_planner", "navigator", "reasoner"])
    b0 = blocks[0]
    check("planner: text = 6 head + 3 tail, visual 4, latent 2, close", b0["text"] == list(range(0, 6)) + list(range(10, 13))
          and b0["visual"] == list(range(6, 10)) and b0["latent"] == [13, 14] and b0["close"] == list(range(15, 15 + close_n)), b0)
    check("planner: text ids recovered", [b0["text_ids"][c] for c in b0["text"]] == ids_all[0].tolist())
    check("planner: close ids / positions", b0["close_ids"] == Tok().encode("<|im_end|>\n") and b0["close_pos"][15] == [15, 15, 15])
    check("navigator block starts at planner end", blocks[1]["start"] == L1 and blocks[1]["end"] == L2)

    def answer(cache, mode):
        os.environ["VLMAS_TEXTCUT"] = mode
        with tc.terminal(eng, cache, cur, case_index=0) as st:
            with torch.no_grad():
                torch.manual_seed(9)
                lm(inputs_embeds=torch.randn(1, 4, 64), position_ids=text_positions(4, cur), past_key_values=cache, use_cache=True)
                lm(inputs_embeds=torch.randn(1, 1, 64), position_ids=text_positions(1, cur + 4), past_key_values=cache, use_cache=True)
        return st

    # diag
    native = copy.deepcopy(cache)
    st = answer(cache, "diag")
    check("diag: cache untouched, n_cut counted", all(torch.equal(a.keys[..., :L3, :], b.keys) for a, b in zip(cache.layers, native.layers))
          and st["n_cut"] == (9 + close_n) + (8 + close_n) and not st["applied"], st)
    ms = st["mass"]
    tot = sum(v for k, v in ms.items() if k != "sink")
    check("diag: masses cover the cache (sum == 1)", abs(tot - 1.0) < 1e-4, tot)
    check("diag: groups present", all(k in ms for k in ("evidence_planner.text", "navigator.visual", "reasoner.latent", "answerer_prompt")))

    # drop
    cache, _, cur, _ = build("drop")
    native = copy.deepcopy(cache)
    st = answer(cache, "drop")
    cut = sorted(set(blocks[0]["text"] + blocks[0]["close"] + blocks[1]["text"] + blocks[1]["close"]))
    keep = [c for c in range(L3) if c not in cut]
    ok = all(torch.equal(a.keys[..., :len(keep), :], b.keys[..., keep, :]) and torch.equal(a.values[..., :len(keep), :], b.values[..., keep, :])
             for a, b in zip(cache.layers, native.layers))
    check("drop: surviving columns == native kept columns in order", ok and st["len_after"] == L3 - len(cut), (st.get("len_after"), L3 - len(cut)))
    check("drop: answerer rows appended after the shortened cache", tc._kvlen(cache) == L3 - len(cut) + 5)
    ms = st["mass"]
    check("drop: cut groups have zero mass, others cover 1", ms["evidence_planner.text"] == 0 and ms["navigator.text"] == 0
          and abs(sum(v for k, v in ms.items() if k != "sink") - 1.0) < 1e-4, ms)

    # filler (shuffle)
    cache, _, cur, _ = build("filler")
    native = copy.deepcopy(cache)
    os.environ["VLMAS_TEXTCUT_FILLER"] = "shuffle"
    st = answer(cache, "filler")
    same_len = tc._kvlen(cache) == L3 + 5
    others_same = all(torch.equal(a.keys[..., keep, :], b.keys[..., keep, :]) and torch.equal(a.values[..., keep, :], b.values[..., keep, :])
                      for a, b in zip(cache.layers, native.layers))
    changed = all(not torch.equal(a.keys[..., cut, :], b.keys[..., cut, :]) for a, b in zip(cache.layers, native.layers))
    check("filler: same length, kept columns untouched, cut columns replaced", same_len and others_same and changed and st["applied"], st.get("filler_runs"))
    # manual recomputation of the first run with the same filler ids on the native prefix
    runs = tc.runs_of(cut)
    s, e = runs[0]
    b = blocks[0]
    ids = [b["text_ids"][c] for c in range(s, e)]
    fids = tc.filler_ids(ids, "shuffle", seed=42 + 0 * 1000 + s, space_id=Tok().encode(" ")[0])
    with torch.no_grad():
        tmp = DynamicCache()
        for li, layer in enumerate(native.layers):
            tmp.update(layer.keys[..., :s, :], layer.values[..., :s, :], li)
        pos = torch.tensor([b["text_pos"][c] for c in range(s, e)]).T.unsqueeze(1)
        out = lm(inputs_embeds=lm.embed_tokens(torch.tensor(fids)).unsqueeze(0), position_ids=pos, past_key_values=tmp, use_cache=True)
    ok = all(torch.allclose(cache.layers[li].keys[..., s:e, :], out.past_key_values.layers[li].keys[..., s:e, :], atol=1e-5)
             for li in range(len(cache.layers)))
    check("filler: first run == manual forward of the shuffled ids at original positions", ok)
    check("filler: runs = contiguous cut runs", st["filler_runs"] == len(runs), (st["filler_runs"], len(runs)))
    # filler (space)
    cache, _, cur, _ = build("filler")
    os.environ["VLMAS_TEXTCUT_FILLER"] = "space"
    st = answer(cache, "filler")
    check("filler space: applied", st["applied"] and st["filler"] == "space")
    os.environ.pop("VLMAS_TEXTCUT_FILLER")

    # roles override
    cache, _, cur, _ = build("drop")
    os.environ["VLMAS_TEXTCUT_ROLES"] = "navigator"
    st = answer(cache, "drop")
    check("roles override: only navigator cut", st["cut_by_role"] == {"navigator": 8 + close_n})
    os.environ.pop("VLMAS_TEXTCUT_ROLES")

    # no bookkeeping -> skip; base -> nothing
    bb._tc_blocks = []
    os.environ["VLMAS_TEXTCUT"] = "drop"
    with tc.terminal(eng, DynamicCache(), 10, case_index=0) as st:
        pass
    check("no blocks -> skip", "skip" in st)
    os.environ.pop("VLMAS_TEXTCUT")
    n_pre = len(lm._forward_pre_hooks)
    with tc.stage_probe(bb, "reasoner") as cap:
        check("base: probe None, no hooks", cap is None and len(lm._forward_pre_hooks) == n_pre)
except ImportError as exc:
    print(f"  skip tiny-model block: {exc!r}")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
