from utils.helpers import *

import numpy as np

def hampel_filter_1d(x, k=11, n_sigmas=3.0):

    x = x.astype(float).copy()
    k = int(k) if k % 2 == 1 else int(k) + 1
    half = k // 2
    xp = np.pad(x, (half, half), mode="edge")
    med = np.empty_like(x, dtype=float)
    mad = np.empty_like(x, dtype=float)

    for i in range(len(x)):
        w = xp[i:i+k]
        m = np.median(w)
        med[i] = m
        mad[i] = np.median(np.abs(w - m)) + 1e-9

    thresh = n_sigmas * 1.4826 * mad
    mask = np.abs(x - med) > thresh
    y = x.copy()
    y[mask] = med[mask]

    return y, mask

def uc_preprocess(

    uc: np.ndarray,
    lpf_cutoff_hz: float = 0.1,
    median_k: int = 11,
    baseline_k: int = 61,
    hampel_k: int = 11,
    hampel_sigmas: float = 3.0,

):

    # Input guards
    x = np.asarray(uc, float).copy()
    median_k = median_k if median_k % 2 == 1 else median_k + 1
    baseline_k = baseline_k if baseline_k % 2 == 1 else baseline_k + 1

    # Light denoise (Low-pass + Moving Median) -> Spike suppression -> Baseline subtraction
    x = butter_filter(x, lpf_cutoff_hz, "low")
    x = moving_median(x, median_k)
    x, m = hampel_filter_1d(x, k=hampel_k, n_sigmas=hampel_sigmas)
    base = moving_median(x, baseline_k)
    x = x - base

    # Uncomment to return masks
    # masks = {'spike_hampel' : m}

    return x

def uc_detect_contractions(

    uc_proc: np.ndarray,
    min_distance_s: int = 100,
    prom_scale_mad: float = 1.0

):

    x = uc_proc.astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-6
    prominence = prom_scale_mad * mad
    peaks, props = find_peaks(
        np.nan_to_num(x, nan=med),
        distance=min_distance_s,
        prominence=prominence
    )

    props["mad_used"] = mad
    props["prominence_used"] = prominence

    return peaks, props

def uc_time_features(win: np.ndarray, peaks: np.ndarray, props: dict) -> dict:

    feats = {}

    if peaks.size == 0:

        feats.update(
            {
                "uc_n_contr": 0,
                "uc_amp_mean": 0.0,
                "uc_dur_mean": 0.0,
                "uc_ici_mean": 0.0,
                "uc_duty_cycle": 0.0,
                "uc_rise_slope_mean": 0.0,
                "uc_fall_slope_mean": 0.0,
                "uc_area_mean": 0.0,
                "uc_amp_sd": 0.0,
                "uc_dur_sd": 0.0,
                "uc_ici_sd": 0.0,
            }
        )

        return feats

    amp = props.get("prominences", np.array([])).astype(float)
    left = props.get("left_bases", np.array([])).astype(int)
    right = props.get("right_bases", np.array([])).astype(int)

    n = len(peaks)
    dur = (right - left).astype(float)
    ici = np.diff(peaks).astype(float) if n > 1 else np.array([])

    duty = float(np.sum(dur) / len(win))

    rises, falls, areas = [], [], []
    x = np.nan_to_num(win, nan=np.nanmedian(win))
    for pk, l, r in zip(peaks, left, right):

        l = int(l); r = int(r)

        if r <= l or pk <= l or pk >= r:
            continue

        rises.append((x[pk] - x[l]) / max(1, (pk - l)))
        falls.append((x[r] - x[pk]) / max(1, (r - pk)))
        areas.append(float(np.trapz(x[l:r] - x[l], dx=1.0)))

    feats.update(
        {
            "uc_n_contr": int(n),
            "uc_amp_mean": float(np.mean(amp)) if amp.size else 0.0,
            "uc_dur_mean": float(np.mean(dur)) if dur.size else 0.0,
            "uc_ici_mean": float(np.mean(ici)) if ici.size else 0.0,
            "uc_duty_cycle": duty,
            "uc_rise_slope_mean": float(np.mean(rises)) if rises else 0.0,
            "uc_fall_slope_mean": float(np.mean(falls)) if falls else 0.0,
            "uc_area_mean": float(np.mean(areas)) if areas else 0.0,
            "uc_amp_sd": float(np.std(amp, ddof=1)) if amp.size > 1 else 0.0,
            "uc_dur_sd": float(np.std(dur, ddof=1)) if dur.size > 1 else 0.0,
            "uc_ici_sd": float(np.std(ici, ddof=1)) if ici.size > 1 else 0.0,
        }
    )

    return feats

def uc_spectral_features(win: np.ndarray) -> dict:

    x = np.nan_to_num(win, nan=np.nanmedian(win))
    n = len(x)
    nperseg = min(256, n)  # cap at window length
    f, pxx = welch(x, fs=FS, window="hamming", nperseg=nperseg)  # no noverlap arg
    mask_low = (f >= 0.003) & (f <= 0.02)
    dom_freq = float(f[mask_low][np.argmax(pxx[mask_low])]) if np.any(mask_low) else 0.0
    low_power = float(np.trapz(pxx[f < 0.03], f[f < 0.03]))
    total_power = float(np.trapz(pxx[f <= 0.1], f[f <= 0.1]))
    p = pxx / (pxx.sum() + 1e-12)

    return {
        "uc_dom_freq": dom_freq,
        "uc_power_sub003": low_power,
        "uc_total_power": total_power,
        "uc_spec_entropy": float(np.sum(-p * np.log2(p + 1e-12))),
    }

def extract_uc_features(

        uc_proc: np.ndarray,
        window_s: int = 120,
        stride_s: int = 30,
        coverage_thresh: float = 0.8,
        min_distance_s: int = 100,
        prom_scale_mad: float = 1.0

):

    rows = [] ; n = len(uc_proc)
    for s, e in sliding_windows(n, w=window_s, s=stride_s):

        win = uc_proc[s:e]
        cov = float(np.isfinite(win).mean())

        if cov < coverage_thresh:
            continue

        med = np.nanmedian(win)
        mad = np.nanmedian(np.abs(win - med)) + 1e-6
        prominence = prom_scale_mad * mad
        peaks, props = find_peaks(
            np.nan_to_num(win, nan=med),
            distance=min_distance_s,
            prominence=prominence
        )

        feats = {
            "t_start": int(s),
            "t_end": int(e),
            "coverage": cov,
            "uc_mad_local": float(mad),
            "uc_prominence_used": float(prominence),
        }

        feats.update(uc_time_features(win, peaks, props))
        feats.update(uc_spectral_features(win))

        rows.append(feats)

    return rows