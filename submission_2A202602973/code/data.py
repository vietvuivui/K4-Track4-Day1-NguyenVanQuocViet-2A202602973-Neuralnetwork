"""data.py — nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_train, y_train = tr["X"], tr["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]

    for X, y in ((X_train, y_train), (X_eval, y_eval)):
        assert X.ndim == 2 and X.shape[1] == N_FEATURES and X.dtype == np.float32, (X.shape, X.dtype)
        assert y.shape == (X.shape[0],) and y.dtype == np.int64, (y.shape, y.dtype)
        assert y.min() >= 0 and y.max() <= N_CLASSES - 1
    assert eval_row_id.shape == (X_eval.shape[0],)
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval), phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Mọi thí nghiệm dùng cùng seed và val_fraction => cùng một phép tách.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """mean và std của N_NUMERIC cột đầu, CHỈ tính trên phần train (sau khi tách val).

    Tính trên val/eval là rò rỉ thông tin: thống kê của dữ liệu dùng để đánh giá lọt vào tiền xử lý.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    std[std == 0] = 1.0  # tránh chia cho 0 nếu có cột hằng
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Bản sao của X, 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    Xs = X.copy()
    Xs[:, :N_NUMERIC] = (Xs[:, :N_NUMERIC] - mean) / std
    return Xs


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed", verbose: bool = True) -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict: X_tr, y_tr, X_val, y_val, X_eval, y_eval (tensor trên device), eval_row_id (numpy),
    cùng mean/std (numpy) để kiểm tra.
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    X_tr, X_val, X_eval = (apply_standardizer(X, mean, std) for X in (X_tr, X_val, X_eval))

    def T(a, dtype):
        return torch.tensor(a, dtype=dtype, device=device)

    data = dict(
        X_tr=T(X_tr, torch.float32), y_tr=T(y_tr, torch.int64),
        X_val=T(X_val, torch.float32), y_val=T(y_val, torch.int64),
        X_eval=T(X_eval, torch.float32), y_eval=T(y_eval, torch.int64),
        eval_row_id=eval_row_id, mean=mean, std=std,
    )

    if verbose:
        print(f"train (sau tách val): {X_tr.shape}  val: {X_val.shape}  eval: {X_eval.shape}")
        maj = np.bincount(y_tr, minlength=N_CLASSES).argmax()  # lớp đa số xác định trên train
        print(f"lớp đa số (theo train) = {maj}; accuracy 'luôn đoán lớp đa số' trên val = {(y_val == maj).mean():.4f}")
        print("tỉ lệ lớp (%)  train / val / eval:")
        for name, yy in (("train", y_tr), ("val", y_val), ("eval", y_eval)):
            p = 100 * np.bincount(yy, minlength=N_CLASSES) / len(yy)
            print(f"  {name:5s}", " ".join(f"{v:6.2f}" for v in p))
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Batch cuối nhỏ hơn batch_size vẫn được giữ lại (không drop_last) để mọi mẫu đều được dùng mỗi epoch.
    `generator` phải nằm trên cùng device với X (torch.randperm yêu cầu vậy).
    """
    N = len(X)
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)
    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
