"""Local mechanism checks, not a trained Transformer or a paper reproduction.
Run once to create the record; later runs compare it without overwriting.
"""
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def pack(x):
    return x.detach().tolist() if isinstance(x, torch.Tensor) else x


def scaling_demo():
    q = torch.ones(1, 64)
    k = torch.stack([torch.full((64,), .125), torch.zeros(64),
                     torch.full((64,), -.125)])
    v = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    raw = q @ k.T
    scaled = raw / math.sqrt(q.shape[-1])
    p_raw, p_scaled = raw.softmax(-1), scaled.softmax(-1)
    out_raw, out_scaled = p_raw @ v, p_scaled @ v
    diag_raw = p_raw * (1 - p_raw)
    diag_scaled = p_scaled * (1 - p_scaled)
    assert torch.equal(raw, torch.tensor([[8., 0., -8.]]))
    assert torch.equal(scaled, torch.tensor([[1., 0., -1.]]))
    return {k: pack(v) for k, v in locals().items() if k != "k"} | {"k": pack(k)}


def attention(q, k, v):
    scores = q @ k.T / math.sqrt(q.shape[-1])
    a = scores.softmax(-1)
    return scores, a, a @ v


def heads_demo():
    t = torch.tensor([[1., 0., 1., 0.]])
    h = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 1.]])
    w0, w1 = torch.eye(4)[:, :2], torch.eye(4)[:, 2:]
    wo = torch.eye(4)
    q0, k0, v0 = t @ w0, h @ w0, h @ w0
    q1, k1, v1 = t @ w1, h @ w1, h @ w1
    s0, a0, o0 = attention(q0, k0, v0)
    s1, a1, o1 = attention(q1, k1, v1)
    joined = torch.cat([o0, o1], -1)
    multi = joined @ wo
    s_single, a_single, single = attention(t, h, h)
    assert not torch.allclose(a0, a1)
    assert multi.shape == single.shape == (1, 4)
    return {k: pack(v) for k, v in locals().items()}


def position_demo():
    e = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
    order = torch.tensor([2, 0, 1])
    pos = torch.arange(3)[:, None]
    freq = torch.tensor([1., .01])[None, :]
    angles = pos * freq
    pe = torch.empty(3, 4)
    pe[:, 0::2], pe[:, 1::2] = angles.sin(), angles.cos()
    base = attention(e, e, e)[2]
    permuted = attention(e[order], e[order], e[order])[2]
    positioned = attention(e + pe, e + pe, e + pe)[2]
    swapped_content = attention(e[order] + pe, e[order] + pe, e[order] + pe)[2]
    no_pos_error = (permuted - base[order]).abs().max().item()
    with_pos_difference = (swapped_content - positioned[order]).abs().max().item()
    assert no_pos_error < 1e-12 and with_pos_difference > .01
    return {k: pack(v) for k, v in locals().items()}


def ffn_demo():
    x = torch.tensor([[-1., 2.], [2., -1.]])
    w1 = torch.tensor([[1., -1., 0., 0.], [0., 0., 1., -1.]])
    w2 = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    hidden = x @ w1
    active = hidden.relu()
    out = active @ w2
    linear_only = hidden @ w2
    edited = x.clone()
    edited[0, 0] = -3.
    edited_out = (edited @ w1).relu() @ w2
    assert torch.equal(out, x.abs())
    assert torch.equal(linear_only, torch.zeros_like(out))
    assert torch.equal(out[1], edited_out[1])
    return {k: pack(v) for k, v in locals().items()}


def resnorm_demo():
    x = torch.tensor([[1., 2., 3., 4.]])
    update = torch.tensor([[.5, -.5, .5, -.5]])
    r = x + update
    mu = r.mean(-1, keepdim=True)
    var = ((r - mu) ** 2).mean(-1, keepdim=True)
    z = (r - mu) / torch.sqrt(var + 1e-5)
    gamma = torch.tensor([1., 2., 1., 2.])
    beta = torch.ones(4)
    y = z * gamma + beta
    assert torch.allclose(y, F.layer_norm(r, (4,), gamma, beta, 1e-5))
    assert torch.allclose(y.mean(-1), torch.ones(1))
    # Residual gradient is checked only at the addition, before LayerNorm.
    a = torch.tensor(2., requires_grad=True)
    branch = .01 * a
    residual = a + branch
    branch_grad = torch.autograd.grad(branch, a, retain_graph=True)[0]
    residual_grad = torch.autograd.grad(residual, a)[0]
    assert math.isclose(branch_grad.item(), .01)
    assert math.isclose(residual_grad.item(), 1.01)
    return {k: pack(v) for k, v in locals().items()}


def dependency_demo():
    full = torch.tensor([0, 1, 3, 5])
    inputs, labels = full[:-1], full[1:]
    allow = torch.ones(3, 3, dtype=torch.bool).tril()
    cells = {str(n): n*n for n in (128, 256, 512)}
    assert allow[1].tolist() == [True, True, False]
    assert cells["256"] == 4*cells["128"]
    return dict(full=pack(full), inputs=pack(inputs), labels=pack(labels),
                allow=pack(allow), attention_cells=cells)


def main():
    data = dict(torch_version=torch.__version__, scope="independent deterministic mechanism checks",
                scaling=scaling_demo(), heads=heads_demo(), position=position_demo(),
                ffn=ffn_demo(), resnorm=resnorm_demo(), dependency=dependency_demo())
    file = Path(__file__).with_name("04_设计原理验证_运行记录.json")
    if file.exists():
        assert json.loads(file.read_text()) == data, "Existing record differs; not overwritten."
        print("Existing record matches; not overwritten.")
    else:
        file.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print("Created", file)
    print(json.dumps({
        "scaling_p_raw":data["scaling"]["p_raw"], "scaling_p_scaled":data["scaling"]["p_scaled"],
        "scaling_out_raw":data["scaling"]["out_raw"], "scaling_out_scaled":data["scaling"]["out_scaled"],
        "heads_a0":data["heads"]["a0"],"heads_a1":data["heads"]["a1"],
        "multi":data["heads"]["multi"],"single":data["heads"]["single"],
        "position_errors":[data["position"]["no_pos_error"],data["position"]["with_pos_difference"]],
        "ffn_out":data["ffn"]["out"],"norm_y":data["resnorm"]["y"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
