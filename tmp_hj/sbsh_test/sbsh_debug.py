import sys
sys.path.insert(0, ".")
import torch
from copy import deepcopy
from transformers import DynamicCache
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from memory import sbsh
torch.manual_seed(0)
cfg = Qwen3VLTextConfig(vocab_size=97, hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=12, max_position_embeddings=512, rope_scaling={"rope_type": "default", "mrope_section": [2, 2, 2], "mrope_interleaved": True})
cfg._attn_implementation = "sdpa"
lm = Qwen3VLTextModel(cfg).double().eval()
for p in lm.parameters(): p.requires_grad_(False)
W = torch.randn(48, 48, dtype=torch.float64) / 48 ** 0.5
class BB:
    lm = None
    def _apply_realign(self, h): return torch.tanh(h @ W) * 2.0
    def _text_positions(self, n, s): return torch.arange(s, s + n).view(1, -1).expand(3, -1).unsqueeze(1)
bb = BB(); bb.lm = lm
PRE = 40
emb = torch.randn(1, PRE, 48, dtype=torch.float64)
with torch.no_grad():
    o = lm(inputs_embeds=emb, position_ids=bb._text_positions(PRE, 0), past_key_values=DynamicCache(), use_cache=True, output_hidden_states=True)
base = o.past_key_values; h0 = o.hidden_states[-1][:, -1:, :]
grp = [torch.arange(6, 14), torch.arange(14, 22), torch.arange(22, 30)]
r = torch.randn(48, dtype=torch.float64)
def f(uv, M, grad):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        z = sbsh.replay_latents(bb, base, h0, PRE, M, grp, uv)
        return (r * z).sum()          # no normalisation, to isolate
for M in (1, 4):
    uu = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(f(uu, M, True), uu)
    print("M", M, "grad", [round(x, 6) for x in g.tolist()])
    for eps in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-5):
        fd = []
        for i in range(3):
            e = torch.zeros(3, dtype=torch.float64); e[i] = eps
            fd.append(float((f(e, M, False) - f(-e, M, False)) / (2 * eps)))
        print("   eps", eps, "fd", [round(x, 6) for x in fd])
# grad wrt h0 path check (no gate): d(r.z)/d(scale of h0)
def fh(s, M, grad):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        c = deepcopy(base); h = h0 * s
        for step in range(M):
            oo = lm(inputs_embeds=bb._apply_realign(h), position_ids=bb._text_positions(1, PRE + step), past_key_values=c, use_cache=True, output_hidden_states=True)
            c = oo.past_key_values; h = oo.hidden_states[-1][:, -1:, :]
        return (r * h[0, 0]).sum(), oo
s = torch.ones((), dtype=torch.float64, requires_grad=True)
val, oo = fh(s, 2, True)
(gs,) = torch.autograd.grad(val, s)
fds = float((fh(torch.tensor(1 + 1e-5, dtype=torch.float64), 2, False)[0] - fh(torch.tensor(1 - 1e-5, dtype=torch.float64), 2, False)[0]) / 2e-5)
print("h0 scale grad", float(gs), "fd", fds)
import inspect
from transformers.models.qwen3_vl import modeling_qwen3_vl as mq
print([l for l in inspect.getsource(mq.Qwen3VLTextRMSNorm.forward).splitlines() if "float32" in l])
print("hidden_states len", len(oo.hidden_states), "last is last_hidden_state:", torch.equal(oo.hidden_states[-1], oo.last_hidden_state))
