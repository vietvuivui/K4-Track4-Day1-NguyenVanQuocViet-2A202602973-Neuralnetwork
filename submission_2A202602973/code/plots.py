"""plots.py — ảnh biểu đồ là sản phẩm nộp (README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.

Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

METRIC_LABELS = {
    "train_loss": "train loss (eval mode)", "val_loss": "val loss", "val_acc": "val accuracy",
    "val_macro_f1": "val macro-F1", "grad_norm": "grad_norm (trước clip, TB epoch)",
    "grad_norm_max": "grad_norm max / epoch", "epoch_time_s": "thời gian / epoch (s)",
}


def cfg_label(cfg: dict) -> str:
    """Chuỗi cấu hình chính để in trong tiêu đề ảnh."""
    hidden = "-".join(str(h) for h in cfg.get("hidden", ()))
    parts = [f"loss={cfg.get('loss')}", f"opt={cfg.get('optimizer')}", f"lr={cfg.get('lr')}",
             f"wd={cfg.get('weight_decay')}", f"batch={cfg.get('batch')}", f"hidden={hidden}",
             f"drop={cfg.get('dropout')}", f"clip={cfg.get('clip_norm')}", f"prec={cfg.get('precision')}",
             f"init={cfg.get('init')}", f"seed={cfg.get('seed')}"]
    if cfg.get("scheduler"):
        parts.append(f"sched={cfg['scheduler']}")
    return ", ".join(parts)


def plot_run(result: dict, path: str) -> None:
    """Một thí nghiệm -> một ảnh PNG 3 ô: (1) train/val loss, (2) val acc + macro-F1, (3) grad_norm trước clip.
    Đường đứt nét đánh dấu best_epoch (val_loss thấp nhất)."""
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    ax = axes[0]
    ax.plot(ep, h["train_loss"], "o-", ms=3, label="train loss (eval mode)")
    ax.plot(ep, h["val_loss"], "o-", ms=3, label="val loss")
    ax.axhline(s["step0_loss"], color="gray", ls=":", lw=1, label=f"step-0 val loss = {s['step0_loss']:.3f}")
    ax.set_title("Loss"); ax.set_xlabel("epoch"); ax.set_ylabel(f"loss ({cfg.get('loss')})")

    ax = axes[1]
    ax.plot(ep, h["val_acc"], "o-", ms=3, label="val accuracy")
    ax.plot(ep, h["val_macro_f1"], "o-", ms=3, label="val macro-F1")
    ax.axhline(0.4876, color="gray", ls=":", lw=1, label="đoán đa số (acc 0.4876)")
    ax.set_title("Val accuracy / macro-F1"); ax.set_xlabel("epoch"); ax.set_ylabel("score")

    ax = axes[2]
    ax.plot(ep, h["grad_norm"], "o-", ms=3, label="grad_norm TB / epoch")
    if "grad_norm_max" in h:
        ax.plot(ep, h["grad_norm_max"], "--", lw=1, alpha=0.7, label="grad_norm max / epoch")
    if cfg.get("clip_norm"):
        ax.axhline(cfg["clip_norm"], color="red", ls=":", lw=1, label=f"clip c = {cfg['clip_norm']}")
    ax.set_title("Chuẩn gradient (trước clip)"); ax.set_xlabel("epoch"); ax.set_ylabel("‖g‖₂")
    ax.set_yscale("log")

    for ax in axes:
        if s.get("best_epoch"):
            ax.axvline(s["best_epoch"], color="green", ls="--", lw=1, alpha=0.6, label=f"best epoch = {s['best_epoch']}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    title = f"{cfg['exp_id']}  —  {cfg.get('description', '')}"
    if s.get("diverged"):
        title += "  [DIVERGED]"
    fig.suptitle(f"{title}\n{cfg_label(cfg)}", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "") -> None:
    """Vẽ chồng một (hoặc nhiều) chỉ số của nhiều thí nghiệm, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    metric: tên một chỉ số ("val_loss") hoặc list chỉ số (["val_loss", "val_macro_f1", "grad_norm"]) -> mỗi chỉ số một ô.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.8 * len(metrics), 4.4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for r in results:
            h = r["history"]
            if h.get(m):
                ax.plot(h["epoch"], h[m], "o-", ms=2.5, label=r["cfg"]["exp_id"])
        ax.set_title(METRIC_LABELS.get(m, m)); ax.set_xlabel("epoch"); ax.set_ylabel(m)
        if m.startswith("grad_norm"):
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title or f"So sánh: {', '.join(metrics)}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
