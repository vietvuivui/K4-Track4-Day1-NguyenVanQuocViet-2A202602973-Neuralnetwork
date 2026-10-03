"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7
TRAIN_EVAL_SUBSET = 50_000  # train_loss đo trên một tập con CỐ ĐỊNH của train (eval mode) cho nhanh

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val trong notebook (Part 2) rồi ghi đè vào đây.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val trong notebook, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" | "warmup_cosine" (ghi vào notes nếu dùng)
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, k: int = N_CLASSES) -> np.ndarray:
    """Ma trận nhầm lẫn k x k (hàng = nhãn thật, cột = dự đoán), tính trên device rồi trả về numpy."""
    idx = y_true.long() * k + y_pred.long()
    return torch.bincount(idx, minlength=k * k).reshape(k, k).cpu().numpy()


def per_class_scores(cm: np.ndarray):
    """precision, recall, F1 từng lớp — cùng công thức với scripts/evaluate.py."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0."""
    return float(per_class_scores(cm)[2].mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Nhãn dự đoán int64 (N,) = argmax của logits, ở chế độ eval."""
    model.eval()
    preds = [model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)]
    return torch.cat(preds)


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : F.cross_entropy trên logit thô và nhãn int64.
       "mse" : MSE giữa logit và one-hot của y, như nn.MSELoss: không có hệ số 1/2,
               reduction="mean" lấy trung bình trên MỌI phần tử (B*7); "sum" cộng dồn mọi phần tử.
    Loss luôn tính ở FP32 (ép logits.float()) để ổn định khi dùng autocast.
    """
    logits = logits.float()
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        target = F.one_hot(y, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, target, reduction=reduction)
    raise ValueError(f"loss không hỗ trợ: {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """dict(loss, acc, macro_f1, cm) ở chế độ eval() (dropout tắt) và no_grad.

    Loss = tổng loss (reduction="sum") / N; với MSE chia thêm cho 7 để khớp định nghĩa "mean" khi huấn luyện.
    """
    model.eval()
    total, preds = 0.0, []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        total += compute_loss(logits, y[i:i + batch_size], loss_name, reduction="sum").item()
        preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds)
    n = len(X) * (N_CLASSES if loss_name == "mse" else 1)
    cm = confusion_matrix(y, pred)
    return dict(loss=total / n, acc=float((pred == y).float().mean().item()),
                macro_f1=macro_f1_from_confusion(cm), cm=cm)


def _sync(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()


def build_model(cfg: dict) -> MLP:
    """Tạo MLP theo cfg và kiểm tra số tham số đúng quy định."""
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"])
    assert count_params(model) == EXPECTED_PARAMS[hidden], \
        f"số tham số {count_params(model)} != {EXPECTED_PARAMS[hidden]} cho hidden={hidden}"
    return model


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về {"cfg", "history", "summary", "best_state"}.

    - step0_loss: loss trên val TRƯỚC bước cập nhật đầu tiên (CE kỳ vọng ≈ ln 7).
    - train_loss: đo ở eval mode trên tập con cố định TRAIN_EVAL_SUBSET mẫu của train.
    - grad_norm : chuẩn L2 toàn cục TRƯỚC clip, trung bình theo epoch (kèm max để thấy "gai").
    - best_epoch: epoch có val_loss thấp nhất; summary báo cáo val_acc/val_macro_f1 tại epoch đó.
    - diverged  : True nếu loss NaN/inf -> dừng sớm.
    Chỉ dùng val; X_eval không được dùng trong hàm này.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    dev_type = device.type

    set_seed(cfg["seed"])
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), steps_per_epoch * cfg["epochs"])

    precision = cfg["precision"]
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"precision không hỗ trợ: {precision!r}")
    if precision == "fp16" and dev_type != "cuda":
        raise RuntimeError("FP16 autocast + GradScaler cần GPU CUDA")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None

    gen = torch.Generator(device=device)  # xáo dữ liệu tái lập được, độc lập với seed khởi tạo
    gen.manual_seed(cfg["seed"])
    n_sub = min(TRAIN_EVAL_SUBSET, len(X_tr))
    X_sub, y_sub = X_tr[:n_sub], y_tr[:n_sub]  # tập con cố định (split đã xáo sẵn)

    if dev_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step0 = evaluate(model, X_val, y_val, cfg["loss"])
    step0_loss = step0["loss"]

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                            "grad_norm", "grad_norm_max", "clip_frac", "lr", "epoch_time_s")}
    best_val, best_epoch = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    diverged = False

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        _sync(dev_type)
        t0 = time.perf_counter()
        gn_list = []
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=gen):
            with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
            loss = compute_loss(logits, yb, cfg["loss"])
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # đưa gradient về thang thật TRƯỚC khi đo/cắt
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])  # chuẩn TRƯỚC khi cắt
            if scaler is not None:
                scaler.step(optimizer)  # tự bỏ qua bước nếu gradient có inf/NaN (do tràn FP16)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_v = loss.item()
            if not math.isfinite(loss_v) or (scaler is None and not math.isfinite(gn)):
                diverged = True
                break
            if math.isfinite(gn):
                gn_list.append(gn)
        _sync(dev_type)
        epoch_time = time.perf_counter() - t0

        if diverged:
            if verbose:
                print(f"[{cfg['exp_id']}] epoch {epoch}: loss NaN/inf -> diverged, dừng sớm")
            break

        tr = evaluate(model, X_sub, y_sub, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])
        if not math.isfinite(va["loss"]):
            diverged = True
            if verbose:
                print(f"[{cfg['exp_id']}] epoch {epoch}: val_loss NaN/inf -> diverged, dừng sớm")
            break

        gn_arr = np.array(gn_list) if gn_list else np.array([float("nan")])
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr["loss"])
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"])
        hist["val_macro_f1"].append(va["macro_f1"])
        hist["grad_norm"].append(float(gn_arr.mean()))
        hist["grad_norm_max"].append(float(gn_arr.max()))
        hist["clip_frac"].append(float((gn_arr > cfg["clip_norm"]).mean()) if cfg["clip_norm"] else 0.0)
        hist["lr"].append(optimizer.param_groups[0]["lr"])
        hist["epoch_time_s"].append(epoch_time)

        if va["loss"] < best_val:
            best_val, best_epoch = va["loss"], epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(f"[{cfg['exp_id']}] ep {epoch:2d} | train {tr['loss']:.4f} | val {va['loss']:.4f} "
                  f"| acc {va['acc']:.4f} | F1 {va['macro_f1']:.4f} | gn {gn_arr.mean():.3f} "
                  f"| {epoch_time:.1f}s")

    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if dev_type == "cuda" else None
    if best_epoch > 0:
        i = best_epoch - 1
        summary = dict(
            step0_loss=step0_loss, best_val_loss=best_val, best_epoch=best_epoch,
            final_train_loss=hist["train_loss"][-1], final_val_loss=hist["val_loss"][-1],
            val_acc=hist["val_acc"][i], val_macro_f1=hist["val_macro_f1"][i],
            time_per_epoch_s=float(np.mean(hist["epoch_time_s"])),
            peak_mem_MB=peak_mem, diverged=diverged,
        )
    else:  # phân kỳ ngay epoch đầu: chỉ có số đo bước 0
        summary = dict(
            step0_loss=step0_loss, best_val_loss=None, best_epoch=0,
            final_train_loss=None, final_val_loss=None,
            val_acc=step0["acc"], val_macro_f1=step0["macro_f1"],
            time_per_epoch_s=None, peak_mem_MB=peak_mem, diverged=diverged,
        )
    return {"cfg": cfg, "history": hist, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp CSV `row_id,pred` (pred 0..6), đủ mọi dòng eval, mỗi row_id đúng một lần."""
    row_id = np.asarray(row_id).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    assert row_id.shape == preds.shape, (row_id.shape, preds.shape)
    assert len(np.unique(row_id)) == len(row_id), "row_id bị lặp"
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame({"row_id": row_id, "pred": preds}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str,
               repo_root: str | None = None, out_json: str | None = None) -> dict | None:
    """Dùng cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Nếu có repo_root: chạy luôn scripts/evaluate.py (đúng script giảng viên chấm) và trả về JSON kết quả.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    model = build_model(cfg).to(data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])  # fp32, eval mode
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print("đã ghi", pred_path)

    if repo_root is None:
        return None
    out_json = out_json or os.path.splitext(pred_path)[0] + "_result.json"
    cmd = [sys.executable, "scripts/evaluate.py", "--pred", os.path.abspath(pred_path),
           "--out", os.path.abspath(out_json)]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, encoding="utf-8")
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("scripts/evaluate.py báo lỗi")
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)
