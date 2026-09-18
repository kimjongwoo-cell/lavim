"""Decode the latent steps an agent produces, straight out of the latent loop (opt-in).

Nothing here runs unless VLMAS_LATDEC=<jsonl>. Measurement only: no extra forward, no
decode change — it reads `latent_trajectory`, which `continue_with_latent_steps` already
returns (one soft embedding per latent step, the thing fed back in as the next step's
input embedding).

Qwen3-VL ties the input embedding and the unembedding, so a latent embedding lives in the
same space as token embeddings and can be read directly:

    logits_k = E z_k          top-k tokens of latent step k
    p_k      = softmax(E z_k) how peaked that step is
    cos_k    = cos(z_k, E[a]) how close the step is to its own nearest token

With `--realign-method wa` the step is a realigned combination of embedding rows, so the
top-1 sequence is the closest thing to "what the agent said to itself" that exists — it is
NOT a generated sentence and should never be quoted as the model's reasoning.

Env
  VLMAS_LATDEC=<path.jsonl>   output file (append; one row per stage per case). Unset = no-op.
  VLMAS_LATDEC_TOPK=<int>     tokens kept per step (default 8)
  VLMAS_LATDEC_STAGES=a,b     which stages to dump (default: reasoner)
"""
from __future__ import annotations

import json
import os

import torch


def decode_steps(traj, emb, tokenizer, topk: int = 8):
    """[{step, top:[(tok,score,p)], norm, entropy}] for every latent step."""
    E = emb.detach().float()
    out = []
    for k, z in enumerate(traj):
        v = torch.as_tensor(z).detach().float().reshape(-1)
        if v.numel() != E.shape[1]:
            continue
        logits = E @ v.to(E.device)
        p = torch.softmax(logits.double(), dim=-1)
        sc, ix = torch.topk(logits, k=min(int(topk), int(logits.numel())))
        ent = float(-(p * (p + 1e-30).log()).sum())
        out.append({
            "step": k,
            "norm": round(float(v.norm()), 4),
            "entropy": round(ent, 4),
            "top": [{"id": int(i), "tok": tokenizer.decode([int(i)]),
                     "score": round(float(s), 4), "p": round(float(p[int(i)]), 6)}
                    for s, i in zip(sc.tolist(), ix.tolist())],
        })
    return out


def as_text(steps) -> str:
    """The top-1 token of every step, joined — a reading aid, not a generation."""
    return "".join(s["top"][0]["tok"] for s in steps if s["top"])


@torch.no_grad()
def continue_from_handoff(backbone, cache, pos_cursor, seed_embed, n_tokens: int = 96):
    """Let the state that is handed to the Answerer speak for itself.

    Starts from the post-Reasoner cache (exactly what the Answerer inherits) and feeds the
    LAST latent embedding as the next input — the same thing the latent loop would have fed
    — then decodes greedily. No Answerer prompt, no instruction, nothing injected: whatever
    comes out is what the handoff state continues into on its own.
    """
    from memory.vcca_diag import _unembedding

    lm = backbone.lm
    unemb = _unembedding(backbone)
    norm = lm.norm
    dev = backbone.device
    dtype = next(lm.parameters()).dtype
    # The last latent row is already written into the cache, so its hidden state — the one
    # that decides the FIRST spoken token — is gone. Drop that row and re-feed the same
    # latent embedding: the forward reproduces the row exactly and hands back its hidden
    # state. Without this the decode starts one latent step late.
    kv = int(backbone._kv_len(cache))
    cache.crop(kv - 1)
    cursor = int(pos_cursor) - 1
    x = torch.as_tensor(seed_embed).reshape(1, 1, -1).to(device=dev, dtype=dtype)
    tk = backbone.processor.tokenizer
    stop = {i for i in (getattr(tk, "eos_token_id", None),
                        tk.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0}
    ids: list[int] = []
    for _ in range(int(n_tokens)):
        pos = (backbone._text_positions(1, start=cursor)
               if getattr(backbone, "mrope_pos", False) else
               torch.tensor([[cursor]], device=dev))
        out = lm(inputs_embeds=x, past_key_values=cache, position_ids=pos, use_cache=True)
        h = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])[:, -1]
        z = unemb.to(h.dtype).to(h.device) @ norm(h)[0]
        t = int(z.argmax())
        ids.append(t)
        cursor += 1
        if t in stop:
            break
        emb = getattr(lm, "embed_tokens", None)
        if emb is None:
            break
        x = emb(torch.tensor([[t]], device=dev)).to(dtype)
    return ids


ASK_DEFAULT = (
    "The information above is provided in latent KV representation format. "
    "Output the original contents in text, copied as faithfully as you can. "
    "Do not add analysis and do not answer any question.\n"
    "Original contents:"
)


@torch.no_grad()
def ask_readout(backbone, cache, pos_cursor, n_tokens: int, ask: str) -> str:
    """The paper's own readout: ask the consuming agent to restate the latent contents.

    LatentMAS Appendix E hands each consumer a prompt saying the upstream material is "in
    latent KV representation format" and asks it to output "(1) original plan contents",
    and Appendix D's case study shows the downstream agent verbalising the upstream latent
    plan. So the way to read a latent handoff is to PROMPT for it, not to run the latent
    rows through the unembedding.
    """
    from vision_text_mas.latent_terminal import generate_terminal_json

    return generate_terminal_json(
        backbone=backbone, cache=cache, position_cursor=pos_cursor,
        system_prompt="", user_prompt=ask, json_prefix=None,
        max_new_tokens=int(n_tokens), temperature=0.0, top_p=1.0, do_sample=False)


def run(engine, *, stage: str, result: dict, case_index: int) -> None:
    out_path = os.environ.get("VLMAS_LATDEC", "").strip()
    if not out_path:
        return
    want = [s.strip() for s in
            (os.environ.get("VLMAS_LATDEC_STAGES", "reasoner") or "reasoner").split(",")
            if s.strip()]
    if stage not in want:
        return
    traj = result.get("latent_trajectory") or []
    if not traj:
        print(f"[LatDec] case {case_index} {stage}: SKIP (no latent trajectory)", flush=True)
        return
    from memory.vcca_diag import _unembedding

    bb = engine._backbone
    emb = _unembedding(bb)
    if emb is None:
        print(f"[LatDec] case {case_index} {stage}: SKIP (no unembedding)", flush=True)
        return
    topk = int(os.environ.get("VLMAS_LATDEC_TOPK", "8") or 8)
    steps = decode_steps(traj, emb.cpu(), bb.processor.tokenizer, topk)
    rec = {"case": int(case_index), "stage": stage, "n_steps": len(steps),
           "top1_text": as_text(steps), "steps": steps}

    n_cont = int(os.environ.get("VLMAS_LATDEC_CONT", "0") or 0)
    if n_cont > 0 and result.get("past_key_values") is not None:
        from copy import deepcopy
        cursor = result.get("pos_cursor") or result["past_len"]
        c = deepcopy(result["past_key_values"])
        try:
            ids = continue_from_handoff(bb, c, cursor, traj[-1], n_cont)
            rec["handoff_ids"] = ids
            rec["handoff_text"] = bb.processor.tokenizer.decode(ids)
        except Exception as exc:                      # never break the pipeline for a probe
            rec["handoff_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            del c
        # CONTROL: the same decode from the cache cropped to just BEFORE the latent steps,
        # i.e. the agent's prompt with zero latent rows. If this speaks and the handoff does
        # not, the latent block is what silences it; if this collapses too, the collapse is
        # a property of prompt-free continuation and says nothing about the handoff.
        if os.environ.get("VLMAS_LATDEC_CTRL", "").strip() == "1":
            c = deepcopy(result["past_key_values"])
            try:
                # Matched to the handoff arm: SAME seed (the last latent embedding), same
                # crop-and-refeed. The only difference is how many latent rows remain in
                # the cache — 10 there, 1 here. No text token is injected on either side.
                kv = int(bb._kv_len(c))
                c.crop(kv - len(traj) + 1)
                ids0 = continue_from_handoff(bb, c, cursor - len(traj) + 1, traj[-1], n_cont)
                rec["prelatent_ids"] = ids0
                rec["prelatent_text"] = bb.processor.tokenizer.decode(ids0)
            except Exception as exc:
                rec["prelatent_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                del c

    n_ask = int(os.environ.get("VLMAS_LATDEC_ASK", "0") or 0)
    if n_ask > 0 and result.get("past_key_values") is not None:
        from copy import deepcopy
        ask = os.environ.get("VLMAS_LATDEC_ASK_TEXT", "").strip() or ASK_DEFAULT
        rec["ask_prompt"] = ask
        cur = result.get("pos_cursor") or result["past_len"]
        c = deepcopy(result["past_key_values"])
        try:
            rec["ask_text"] = ask_readout(bb, c, cur, n_ask, ask)
        except Exception as exc:
            rec["ask_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            del c
        # CONTROL: the same question asked of the cache with the latent rows removed.
        # If the answer is the same, the readout is reciting the prompt, not the latents.
        if os.environ.get("VLMAS_LATDEC_CTRL", "").strip() == "1":
            c = deepcopy(result["past_key_values"])
            try:
                kv = int(bb._kv_len(c))
                c.crop(kv - len(traj))
                rec["ask_text_nolatent"] = ask_readout(
                    bb, c, cur - len(traj), n_ask, ask)
            except Exception as exc:
                rec["ask_nolatent_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                del c

    with open(out_path, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tail = (f" handoff={rec.get('handoff_text')!r}" if "handoff_text" in rec
            else (f" handoff_error={rec.get('handoff_error')}" if "handoff_error" in rec else ""))
    print(f"[LatDec] case {case_index} {stage} steps={len(steps)} "
          f"top1={as_text(steps)!r}{tail}", flush=True)


__all__ = ["decode_steps", "as_text", "run"]
