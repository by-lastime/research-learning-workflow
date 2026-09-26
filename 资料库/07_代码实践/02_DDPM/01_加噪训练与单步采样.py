"""DDPM 第一讲的独立局部教学程序；不是 U-Net 或完整生成模型。

运行：.venv/bin/python <本文件> [--output 新的记录路径]
默认只打印；指定输出时用 x 模式，拒绝覆盖任何已有运行记录。
固定输入用于逐数追踪，不把手设噪声称作本次随机抽样结果。
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch


def make_schedule(beta):
    alpha = 1 - beta
    return alpha, torch.cumprod(alpha, dim=0)


def extract(values, t):
    """t 使用论文的 1..T 编号；返回每样本一个广播系数。"""
    return values[t - 1].view(-1, 1, 1, 1)


def q_sample(x0, t, eps, alpha_bar):
    a_bar = extract(alpha_bar, t)
    signal = a_bar.sqrt() * x0
    noise = (1 - a_bar).sqrt() * eps
    return signal, noise, signal + noise


class TinyNoisePredictor(torch.nn.Module):
    """人为定义的一参数函数；时间偏置仅为演示，不是论文时间嵌入。"""

    def __init__(self, total_steps):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
        self.total_steps = total_steps

    def forward(self, xt, t):
        time_bias = 0.1 * t.to(xt.dtype).view(-1, 1, 1, 1) / self.total_steps
        return self.w * xt + time_bias


def reverse_step(model, xt, t, z, beta, alpha, alpha_bar):
    """采用 sigma_t²=beta_t；t=1 时按 Algorithm 2 不加随机项。"""
    a = extract(alpha, t)
    b = extract(beta, t)
    ab = extract(alpha_bar, t)
    eps_hat = model(xt, t)
    correction = b / (1 - ab).sqrt() * eps_hat
    mean = (xt - correction) / a.sqrt()
    active_z = torch.where((t > 1).view(-1, 1, 1, 1), z, torch.zeros_like(z))
    random_term = b.sqrt() * active_z
    return eps_hat, correction, mean, random_term, mean + random_term


def values(x):
    return x.detach().tolist()


def run():
    dtype = torch.float64
    beta = torch.tensor([0.1, 0.2, 0.3], dtype=dtype)
    alpha, alpha_bar = make_schedule(beta)
    x0 = torch.tensor([[[[-1.0, -0.5], [0.5, 1.0]]]], dtype=dtype)
    eps = torch.tensor([[[[0.2, -1.0], [0.5, 1.5]]]], dtype=dtype)
    z = torch.tensor([[[[-0.3, 0.4], [1.0, -0.2]]]], dtype=dtype)
    t = torch.tensor([2], dtype=torch.long)
    model = TinyNoisePredictor(total_steps=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    signal, noise, xt = q_sample(x0, t, eps, alpha_bar)
    optimizer.zero_grad(set_to_none=True)
    w_before = model.w.detach().clone()
    eps_hat = model(xt, t)
    error = eps_hat - eps
    squared_error = error.square()
    loss = squared_error.mean()
    assert torch.equal(model.w.detach(), w_before)  # 前向不改参数
    loss.backward()
    grad = model.w.grad.detach().clone()
    assert torch.equal(model.w.detach(), w_before)  # backward 仍不改参数
    expected_grad = (2 * error.detach() * xt).mean()
    torch.testing.assert_close(grad, expected_grad)
    optimizer.step()
    torch.testing.assert_close(model.w, w_before - 0.05 * grad)
    assert torch.equal(model.w.grad, grad)  # step 不自动清理旧梯度
    with torch.no_grad():
        prediction_after = model(xt, t)
        loss_after = (prediction_after - eps).square().mean()
        reverse = reverse_step(model, xt, t, z, beta, alpha, alpha_bar)
    assert not reverse[-1].requires_grad
    assert torch.equal(model.w.grad, grad)  # no_grad 也不清旧梯度
    grad_before_clear = values(model.w.grad)
    optimizer.zero_grad(set_to_none=True)
    assert model.w.grad is None

    # 非对称 batch 变式核验 t-1 索引及广播；不是模型训练实验。
    xb = x0.expand(2, 1, 2, 2).clone()
    tb = torch.tensor([1, 3])
    eb = eps.expand_as(xb).clone()
    batch_xt = q_sample(xb, tb, eb, alpha_bar)[-1]
    for i, ti in enumerate(tb.tolist()):
        expected = alpha_bar[ti - 1].sqrt() * xb[i] + (1 - alpha_bar[ti - 1]).sqrt() * eb[i]
        torch.testing.assert_close(batch_xt[i], expected)
    # t=1 边界：即使传入非零 z，也应不添加随机项。
    with torch.no_grad():
        edge = reverse_step(model, xt, torch.tensor([1]), z, beta, alpha, alpha_bar)
    assert torch.count_nonzero(edge[3]).item() == 0
    torch.testing.assert_close(edge[-1], edge[2])

    # 验证正文中的真实抽样写法及 shape；与固定数值主例分开。
    generator = torch.Generator().manual_seed(20260921)
    sampled_t = torch.randint(1, 4, (2,), generator=generator)
    sampled_eps = torch.randn(xb.shape, generator=generator, dtype=dtype)
    sampled_xt = q_sample(xb, sampled_t, sampled_eps, alpha_bar)[-1]
    assert sampled_xt.shape == xb.shape
    assert torch.isfinite(sampled_xt).all()

    return {
        "scope": "固定输入→闭式加噪→一参数函数真实预测/损失/梯度/SGD更新→同一x2的一次x2到x1转移。非完整DDPM。",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "dtype": "float64"},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_pdf_sha256": hashlib.sha256((Path(__file__).resolve().parents[2] / "01_论文原文/04_ddpm_2006_11239.pdf").read_bytes()).hexdigest(),
        "schedule": {"T": 3, "beta": values(beta), "alpha": values(alpha), "alpha_bar": values(alpha_bar), "sigma_squared_choice": "beta"},
        "forward": {"x0": values(x0), "t": values(t), "eps_fixed": values(eps), "sqrt_alpha_bar_t": values(extract(alpha_bar, t).sqrt()), "sqrt_one_minus_alpha_bar_t": values((1-extract(alpha_bar,t)).sqrt()), "signal": values(signal), "noise": values(noise), "xt": values(xt)},
        "training": {"time_bias": 0.1*2/3, "w_before": values(w_before), "eps_hat": values(eps_hat), "error": values(error), "squared_error": values(squared_error), "loss_mean": values(loss), "loss_sum": values(squared_error.sum()), "grad": values(grad), "lr": 0.05, "w_after": values(model.w), "eps_hat_after": values(prediction_after), "same_sample_loss_after": values(loss_after), "grad_after_step_and_no_grad": grad_before_clear, "grad_after_zero_grad": model.w.grad},
        "reverse_one_step": dict(zip(["eps_hat", "correction", "mean", "random_term", "x_prev"], [values(v) for v in reverse])) | {"input_x2": values(xt), "z_fixed": values(z), "sigma_t": values(extract(beta,t).sqrt()), "uses_updated_w": True},
        "checks": {"shape": list(xt.shape), "forward_does_not_change_w": True, "backward_does_not_change_w": True, "analytic_gradient_matches_autograd": True, "sgd_matches_formula": True, "no_grad_does_not_clear_grad": True, "batch_time_index_and_broadcast": True, "t1_zero_random_term": True, "rng_example_seed": 20260921, "rng_example_t": values(sampled_t), "random_draw_shape": list(sampled_eps.shape)}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = run()
    payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(payload)
        print(f"验证通过；新记录：{args.output}")
    else:
        print(payload, end="")
