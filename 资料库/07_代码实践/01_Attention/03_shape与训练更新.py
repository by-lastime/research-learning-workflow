"""Lesson 03 checks: reuse the original model definitions, not its main experiment.
Run: .venv/bin/python 资料库/07_代码实践/01_Attention/03_shape与训练更新.py
Writes a new lesson-specific JSON record; never edits historical results.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "完整数值流程.py"


def load_model():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    # Retain imports, configuration, parameters and function definitions only.
    stop = next(i for i, node in enumerate(tree.body)
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "trace"
                        for t in node.targets))
    tree.body = tree.body[:stop]
    env = {"__name__": "lesson03_model_definitions"}
    exec(compile(tree, str(SOURCE), "exec"), env)
    return env


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def run():
    env = load_model()
    P, encode, decode = (env[k] for k in ("P", "encode", "decode"))
    # Deliberate length test, not a meaningful source/target translation pair.
    source_ids = torch.tensor([0, 1, 2, 1, 2])
    full = torch.tensor([0, 1, 4, 5])  # BOS 我 狗 EOS
    target_ids, labels = full[:-1], full[1:]
    trace = {}
    initial = {k: v.detach().clone() for k, v in P.items()}
    H = encode(trace, ids=source_ids)
    logits, D = decode(target_ids, H, trace)
    shapes = {
        "H": list(H.shape), "D": list(D.shape),
        "logits": list(logits.shape),
        "logits[-1]": list(logits[-1].shape),
        "logits[-1:]": list(logits[-1:].shape),
    }
    for prefix in ("decoder.self", "decoder.cross"):
        for suffix in ("Q", "K", "V", "head0.weights", "head0.output", "concat"):
            key = prefix + "." + suffix
            shapes[key] = list(torch.tensor(trace[key]).shape)
    assert shapes["decoder.self.head0.weights"] == [3, 3]
    assert shapes["decoder.cross.head0.weights"] == [3, 5]
    assert shapes["decoder.cross.head0.output"] == [3, 2]
    assert shapes["decoder.cross.concat"] == [3, 4]
    assert shapes["logits"] == [3, 6]

    probs = logits.softmax(-1)
    loss_each = -logits.log_softmax(-1)[torch.arange(len(labels)), labels]
    loss = loss_each.mean()
    assert all(v.grad is None for v in P.values())
    before_backward = {k: v.detach().clone() for k, v in P.items()}
    loss.backward()
    assert all(torch.equal(v, before_backward[k]) for k, v in P.items())
    assert all(v.grad is not None and v.grad.shape == v.shape for v in P.values())
    gradients = {k: v.grad.detach().clone() for k, v in P.items()}
    lr = 0.01
    with torch.no_grad():
        for value in P.values():
            value -= lr * value.grad
    assert all(torch.equal(v, before_backward[k] - lr * gradients[k]) for k, v in P.items())
    assert any(not torch.equal(v, before_backward[k]) for k, v in P.items())
    updated = {k: v.detach().clone() for k, v in P.items()}
    # Changing parameters does not magically refresh earlier outputs or H.
    with torch.no_grad():
        H_new = encode(ids=source_ids)
        logits_new, _ = decode(target_ids, H_new)
        loss_new = torch.nn.functional.cross_entropy(logits_new, labels)

    # A separate, fully specified one-parameter model exposes every scalar.
    w = torch.tensor(2.0, requires_grad=True)
    x, target, scalar_lr = 3.0, 9.0, 0.01
    prediction = w * x
    scalar_loss = (prediction - target) ** 2
    scalar_loss.backward()
    scalar = {"x": x, "target": target, "lr": scalar_lr,
              "w_before": w.item(), "prediction_before": prediction.item(),
              "loss_before": scalar_loss.item(), "grad": w.grad.item()}
    assert w.item() == 2.0 and w.grad.item() == -18.0
    with torch.no_grad():
        w -= scalar_lr * w.grad
        new_prediction = w * x
        new_loss = (new_prediction - target) ** 2
    scalar.update(w_after=w.item(), prediction_after=new_prediction.item(),
                  loss_after=new_loss.item(), grad_after_update=w.grad.item(),
                  old_loss_still=scalar_loss.item())
    assert math.isclose(w.item(), 2.18)
    assert math.isclose(new_loss.item(), 6.0516)
    # Recompute, then backward without clearing: gradients accumulate.
    ((w * x - target) ** 2).backward()
    scalar["grad_after_second_backward_without_clear"] = w.grad.item()
    assert math.isclose(w.grad.item(), -32.76)
    w.grad = None
    scalar["grad_after_clear"] = w.grad

    prefix, selected = [0, 1], 4  # Given list-operation example, not model selection.
    prefix.append(selected)
    assert prefix == [0, 1, 4] and len(prefix) == 3
    record = {
        "torch_version": torch.__version__, "seed": 20260908,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "scope": "Full miniature forward and one SGD step; separate scalar update and list examples. No generation experiment.",
        "source_ids": source_ids.tolist(), "target_ids": target_ids.tolist(),
        "labels": labels.tolist(), "shapes": shapes,
        "initial_parameters": {k: v.tolist() for k, v in initial.items()},
        "forward_trace": trace, "probs": probs.detach().tolist(),
        "loss_each": loss_each.detach().tolist(), "loss": loss.item(),
        "gradients": {k: v.tolist() for k, v in gradients.items()},
        "lr": lr, "updated_parameters": {k: v.tolist() for k, v in updated.items()},
        "loss_after_recomputed_forward": loss_new.item(),
        "scalar": scalar, "given_prefix_after_append": prefix,
        "checks": "all assertions passed",
    }
    output = ROOT / "03_shape与训练更新_运行记录.json"
    output.write_text(json.dumps(clean(record), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"shapes": shapes, "scalar": scalar,
                      "loss": loss.item(), "loss_after": loss_new.item(),
                      "checks": record["checks"], "record": str(output)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
