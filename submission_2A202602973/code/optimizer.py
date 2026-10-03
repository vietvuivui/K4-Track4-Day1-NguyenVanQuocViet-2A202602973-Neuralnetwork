"""optimizer.py — chọn bộ tối ưu, bộ lập lịch lr và cắt gradient.

Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    weight_decay của Adam là L2 trộn vào gradient (bị chia bởi sqrt(v_hat));
    của AdamW là suy giảm tách riêng, không đi qua bộ chuẩn hoá thích nghi.
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận {name!r}")
    if lr is None:
        raise ValueError("lr chưa được đặt")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """Bộ lập lịch lr theo BƯỚC (gọi scheduler.step() sau mỗi optimizer.step()).

    name:
        None          : không dùng
        "cosine"      : CosineAnnealingLR từ lr ban đầu về eta_min trong total_steps bước
        "warmup_cosine": khởi động tuyến tính `warmup_steps` bước rồi cosine về 0
    """
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=kwargs.get("eta_min", 0.0))
    if name == "warmup_cosine":
        warmup = max(1, int(kwargs.get("warmup_steps", 0.05 * total_steps)))

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            t = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"scheduler không hỗ trợ: {name!r}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    max_norm=None: chỉ đo chuẩn (clip_grad_norm_ với max_norm=inf không cắt gì).
    Khi dùng FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này,
    nếu không chuẩn đo được là chuẩn của gradient đã bị nhân hệ số s.
    """
    params = [p for p in params if p.grad is not None]
    if max_norm is None:
        total_norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(total_norm)
