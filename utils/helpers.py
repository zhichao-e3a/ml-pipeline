import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import entropy

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

def butter_filter(x: np.ndarray, cutoff_hz: float, btype: str, order: int = 4) -> np.ndarray:

    nyq = 0.5 * FS
    b, a = butter(order, cutoff_hz/nyq, btype=btype)
    x_in = x.copy().astype(float)

    if not np.isfinite(x_in).all():
        med = np.nanmedian(x_in)
        x_in = np.nan_to_num(x_in, nan=med)

    return filtfilt(b, a, x_in, method="gust")
