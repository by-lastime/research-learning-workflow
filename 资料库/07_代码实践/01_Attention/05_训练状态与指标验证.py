"""Deterministic local teaching checks, not Transformer training or BLEU reproduction."""
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

def state_demo():
    w = torch.tensor(2., requires_grad=True)
    rows = []
    def snap(name, loss=None):
        rows.append(dict(step=name, w=w.item(),
                         grad=None if w.grad is None else w.grad.item(),
                         loss=None if loss is None else loss.item()))
    loss1 = (w * 3 - 9) ** 2
    snap("forward1", loss1)
    loss1.backward()
    snap("backward1", loss1)
    with torch.no_grad():
        w -= .01 * w.grad
    snap("update1_old_loss", loss1)
    assert w.grad.item() == -18 and loss1.item() == 9
    w.grad = None
    snap("clear")
    loss2 = (w * 2 - 5) ** 2
    snap("forward2", loss2)
    loss2.backward()
    snap("backward2", loss2)
    fresh = w.grad.item()
    assert math.isclose(fresh, -2.56, abs_tol=1e-12)
    # A separate leaf represents the same new parameter, but retaining an old gradient.
    stale = torch.tensor(w.item(), requires_grad=True)
    stale.grad = torch.tensor(-18.)
    ((stale * 2 - 5) ** 2).backward()
    assert math.isclose(stale.grad.item(), -18 + fresh, abs_tol=1e-12)
    return dict(rows=rows, skipped_clear_grad=stale.grad.item())

def objective_demo():
    logits = torch.tensor([[2.,1.,0.,-1.], [0.,2.,1.,-1.],
                           [0.,1.,2.,-1.], [-1.,0.,1.,2.]])
    labels = torch.tensor([0,1,2,3])
    logp = logits.log_softmax(-1)
    nll = -logp[torch.arange(4), labels]
    q = .9 * F.one_hot(labels, 4).to(logits.dtype) + .1 / 4
    smooth_per_token = -(q * logp).sum(-1)
    assert torch.allclose(smooth_per_token.mean(),
                          F.cross_entropy(logits, labels, label_smoothing=.1))
    return dict(logits=logits.tolist(), labels=labels.tolist(),
                logp=logp.tolist(), p=logp.exp().tolist(), q=q.tolist(),
                nll=nll.tolist(), mean_nll=nll.mean().item(),
                smooth_per_token=smooth_per_token.tolist(),
                smooth_loss=smooth_per_token.mean().item(),
                ppl=nll.mean().exp().item())

def lr_demo():
    def rate(step, d_model=512, warmup=4000):
        return d_model ** -.5 * min(step ** -.5, step * warmup ** -1.5)
    values = {str(s):rate(s) for s in [1,1000,4000,16000]}
    assert values["1000"] < values["4000"]
    assert math.isclose(values["16000"], values["4000"]/2)
    return values

def main():
    data = dict(torch_version=torch.__version__, dtype="float64",
                scope="local state, supplied-logit objective, paper learning-rate formula",
                state=state_demo(), objective=objective_demo(), learning_rate=lr_demo())
    target = Path(__file__).with_name("05_训练状态与指标验证_运行记录.json")
    if target.exists():
        assert json.loads(target.read_text()) == data, "Record differs; not overwritten"
    else:
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

