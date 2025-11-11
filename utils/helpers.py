import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 1.0

def moving_median(x: np.ndarray, k: int) -> np.ndarray:

    k = int(k) if int(k) % 2 == 1 else int(k) + 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x, dtype=float)

    for i in range(len(x)):
        w = xp[i:i+k]
        out[i] = np.nanmedian(w)

    return out

def linear_fill_short_gaps(x, max_len=3):

    y = x.copy().astype(float)
    n = len(y)
    i = 0

    while i < n:
        if np.isnan(y[i]):
            j = i
            while j < n and np.isnan(y[j]): j += 1
            run = j - i
            if run <= max_len:
                x0 = y[i-1] if i > 0 and np.isfinite(y[i-1]) else y[j]
                x1 = y[j]   if j < n and np.isfinite(y[j])   else y[i-1]
                y[i:j] = np.linspace(x0, x1, run+2)[1:-1]
            i = j
        else:
            i += 1

    return y

def sliding_windows(n, w=120, s=30):

    start = 0

    while start + w <= n:
        yield start, start + w
        start += s

def _safe_fill_numeric(x: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    if not np.isfinite(med):
        med = fallback
    return np.nan_to_num(x, nan=med, posinf=med, neginf=med)

def butter_filter(

    x: np.ndarray,
    cutoff_hz,
    btype: str,
    fs: float = FS,
    order: int = 4,
    fallback: str = "return_filled"

) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    x_filled = _safe_fill_numeric(x, fallback=0.0)

    nyq = fs * 0.5
    if isinstance(cutoff_hz, (tuple, list, np.ndarray)):
        if len(cutoff_hz) != 2:
            raise ValueError("Band cutoff must be (low, high).")
        low, high = float(cutoff_hz[0]), float(cutoff_hz[1])
        if not (0 < low < high < nyq):
            raise ValueError(f"Band edges must satisfy 0<low<high<{nyq} Hz. Got {cutoff_hz}.")
        Wn = (low, high)
    else:
        c = float(cutoff_hz)
        if not (0 < c < nyq):
            raise ValueError(f"Cutoff must be 0<{c}<{nyq} Hz.")
        Wn = c

    sos = butter(order, Wn, btype=btype, fs=fs, output="sos")

    padlen = 3 * order
    n = x_filled.size
    if n <= padlen:
        if fallback == "median":
            med = float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else 0.0
            return np.full_like(x_filled, med)
        else:
            return x_filled

    y = sosfiltfilt(sos, x_filled, padtype="odd", padlen=padlen)

    y = _safe_fill_numeric(y, fallback=float(np.nanmedian(y)))

    return y


def pad_signal(x: np.ndarray, target_len: int = 2048) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0)

    if len(x) == target_len:
        return x

    else:
        pad_val = np.nanmedian(x)
        pad_left = (target_len - len(x)) // 2
        pad_right = target_len - len(x) - pad_left
        x_padded = np.pad(x, (pad_left, pad_right), mode="constant", constant_values=pad_val)

        return x_padded
