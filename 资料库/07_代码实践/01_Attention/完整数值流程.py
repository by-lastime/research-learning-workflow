"""Complete miniature post-LN Transformer; stdout is a reproducible JSON trace.
Run from project root: .venv/bin/python 资料库/07_代码实践/01_Attention/完整数值流程.py
No downloaded weights, no forced intermediate outputs, no filesystem writes.
"""
import json
import math
import torch
import torch.nn.functional as F

torch.manual_seed(20260908)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
P = {}
def param(name, shape, kind="random"):
    x = (torch.ones(shape) if kind == "ones" else
         torch.zeros(shape) if kind == "zeros" else torch.randn(shape) * 0.25)
    P[name] = x.requires_grad_()
    return P[name]

param("source_embedding", (3, 4))
param("target_embedding", (6, 4))
for group in ["encoder.self", "decoder.self", "decoder.cross"]:
    for suffix in ["WQ", "WK", "WV", "WO"]:
        param(group + "." + suffix, (4, 4))
for group in ["encoder.ffn", "decoder.ffn"]:
    for suffix, shape in [("W1", (4, 8)), ("b1", (8,)), ("W2", (8, 4)), ("b2", (4,))]:
        param(group + "." + suffix, shape, "zeros" if suffix.startswith("b") else "random")
for group in ["encoder.norm1", "encoder.norm2", "decoder.norm1", "decoder.norm2", "decoder.norm3"]:
    param(group + ".gamma", (4,), "ones")
    param(group + ".beta", (4,), "zeros")
# Target output shares the target embedding transpose; source vocab is separate.
param("output.bias", (6,), "zeros")

def save(t, key, x):
    if t is not None:
        t[key] = x.detach().tolist()

def position(n):
    pos = torch.arange(n)[:, None]
    rate = torch.tensor([1., 0.01])[None, :]
    angles = pos * rate
    out = torch.empty(n, 4)
    out[:, 0::2], out[:, 1::2] = torch.sin(angles), torch.cos(angles)
    return out

def embed(ids, group, t):
    rows = P[group + "_embedding"][ids]
    pe = position(len(ids))
    save(t, group + ".lookup", rows)
    save(t, group + ".scaled_embedding", rows * 2)
    save(t, group + ".position", pe)
    out = rows * 2 + pe
    save(t, group + ".input", out)
    return out

def mha(qin, kvin, name, trace, causal=False):
    q = qin @ P[name + ".WQ"]
    k = kvin @ P[name + ".WK"]
    v = kvin @ P[name + ".WV"]
    for key, val in [("Q", q), ("K", k), ("V", v)]:
        save(trace, name + "." + key, val)
    heads = []
    for h in range(2):
        sl = slice(h * 2, (h + 1) * 2)
        raw = q[:, sl] @ k[:, sl].T
        scaled = raw / math.sqrt(2)
        allow = torch.ones_like(scaled, dtype=torch.bool)
        if causal:
            allow = torch.ones_like(scaled, dtype=torch.bool).tril()
        masked = scaled.masked_fill(~allow, -torch.inf)
        a = masked.softmax(dim=-1)
        out = a @ v[:, sl]
        for key, val in [("raw", raw), ("scaled", scaled), ("allowed", allow),
                         ("masked", masked), ("weights", a), ("output", out)]:
            save(trace, f"{name}.head{h}." + key, val)
        assert torch.allclose(a.sum(-1), torch.ones(len(qin)))
        assert torch.all(a[~allow] == 0)
        heads.append(out)
    joined = torch.cat(heads, dim=-1)
    out = joined @ P[name + ".WO"]
    save(trace, name + ".concat", joined)
    save(trace, name + ".projected", out)
    return out

def add_norm(x, update, name, t):
    r = x + update
    mean = r.mean(-1, keepdim=True)
    var = ((r - mean) ** 2).mean(-1, keepdim=True)
    centered = r - mean
    normalized = centered / torch.sqrt(var + 1e-5)
    out = normalized * P[name + ".gamma"] + P[name + ".beta"]
    for key, val in [("residual", r), ("mean", mean), ("variance", var),
                     ("centered", centered), ("normalized", normalized), ("output", out)]:
        save(t, name + "." + key, val)
    assert torch.allclose(out, F.layer_norm(r, (4,), P[name+".gamma"],
                                          P[name+".beta"], 1e-5))
    return out

def ffn(x, name, t):
    hidden = x @ P[name + ".W1"] + P[name + ".b1"]
    active = hidden.relu()
    out = active @ P[name + ".W2"] + P[name + ".b2"]
    for key, val in [("linear1", hidden), ("relu", active), ("linear2", out)]:
        save(t, name + "." + key, val)
    return out

def encode(t=None, ids=None):
    if ids is None: ids = torch.tensor([0, 1, 2])
    x = embed(ids, "source", t)
    e = add_norm(x, mha(x, x, "encoder.self", t), "encoder.norm1", t)
    return add_norm(e, ffn(e, "encoder.ffn", t), "encoder.norm2", t)

def decode(ids, H, t=None):
    u = embed(ids, "target", t)
    d1 = add_norm(u, mha(u, u, "decoder.self", t, True), "decoder.norm1", t)
    d2 = add_norm(d1, mha(d1, H, "decoder.cross", t), "decoder.norm2", t)
    d3 = add_norm(d2, ffn(d2, "decoder.ffn", t), "decoder.norm3", t)
    logits = d3 @ P["target_embedding"].T + P["output.bias"]
    prob = logits.softmax(-1)
    save(t, "output.logits", logits)
    save(t, "output.probabilities", prob)
    return logits, d3

trace = {}
H = encode(trace)
u, labels = torch.tensor([0,1,2,3]), torch.tensor([1,2,3,5])
logits, D = decode(u, H, trace)
loss_each = -logits.log_softmax(-1)[torch.arange(4), labels]
loss = loss_each.mean()
save(trace, "loss.per_position", loss_each)
save(trace, "loss.mean", loss)
assert torch.allclose(loss, F.cross_entropy(logits, labels))

# Verify no future leakage in target self-attention.
with torch.no_grad():
    _, Dchanged = decode(torch.tensor([0,1,4,4]), H)
    assert torch.allclose(D[:2], Dchanged[:2], atol=1e-12)
    Hchanged = encode(ids=torch.tensor([0,1,0]))
    altered, _ = decode(u, Hchanged)
    assert not torch.allclose(logits, altered)
initial = {k: v.detach().tolist() for k, v in P.items()}

# Greedy inference uses the SAME initial parameters, not teacher labels.
generation = []
prefix = [0]
with torch.no_grad():
    for step in range(4):
        gt = {}
        gl, _ = decode(torch.tensor(prefix), H, gt)
        probs = gl[-1].softmax(-1)
        selected = int(probs.argmax())
        generation.append({"prefix": prefix.copy(), "selected": selected, "trace": gt})
        prefix.append(selected)
        if selected == 5:
            break
reason = "EOS" if prefix[-1] == 5 else "max_new_tokens=4"

# One real optimization step, separate from inference above.
loss.backward()
assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in P.values())
gradients = {k: v.grad.detach().tolist() for k,v in P.items()}
lr = 0.01
with torch.no_grad():
    for value in P.values():
        value -= lr * value.grad
    after_logits, _ = decode(u, encode())
    after_loss = F.cross_entropy(after_logits, labels)
after = {k: v.detach().tolist() for k,v in P.items()}
report = {"torch_version": torch.__version__, "seed": 20260908,
          "parameters": initial, "forward": trace,
          "generation": generation, "stop_reason": reason,
          "learning_rate": lr, "gradients": gradients, "updated_parameters": after,
          "after_loss": after_loss.item(),
          "checks": ["attention rows sum to one", "masked entries are zero",
                     "LayerNorm matches torch", "loss matches cross_entropy",
                     "future tokens do not change earlier outputs",
                     "source change changes decoder output", "all gradients finite"]}
# Encode nonfinite mask sentinels explicitly for strict JSON.
def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return "-inf" if x < 0 else "inf"
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, dict): return {k: clean(v) for k,v in x.items()}
    return x
print(json.dumps(clean(report), ensure_ascii=False, allow_nan=False))
