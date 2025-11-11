from utils.helpers import *

import numpy as np
from scipy.signal import welch

FS = 1.0

def _runs(mask: np.ndarray) -> list[tuple[int,int]]:

    if mask.size == 0:
        return []

    m = np.concatenate([[False], mask, [False]])
    idx = np.flatnonzero(m[1:] != m[:-1])

    return list(zip(idx[0::2], idx[1::2]))

def clamp_out_of_range(fhr: np.ndarray, lo: float = 50.0, hi: float = 210.0) -> tuple[np.ndarray, np.ndarray]:

    x = fhr.astype(float).copy()
    bad = (x < lo) | (x > hi)
    x[bad] = np.nan

    return x, bad

def mask_spikes(fhr: np.ndarray, window: int = 11, threshold_bpm: float = 25.0) -> tuple[np.ndarray, np.ndarray]:

    x = fhr.astype(float).copy()
    med = moving_median(x, window)
    spike = np.abs(x - med) > threshold_bpm
    x[spike] = np.nan

    return x, spike

def dropout_detection(fhr: np.ndarray, max_flat_seconds: int = 3, atol: float = 0.1):

    x = fhr.astype(float)

    same_as_prev = np.zeros_like(x, dtype=bool)
    same_as_prev[1:] = np.isfinite(x[1:]) & np.isfinite(x[:-1]) & (np.abs(x[1:] - x[:-1]) <= atol)

    flat_mask = np.zeros_like(x, dtype=bool)
    run_len = 1

    for i in range(1, len(x)):
        if same_as_prev[i]:
            run_len += 1
        else:
            if run_len >= max_flat_seconds:
                flat_mask[i-run_len:i] = True
            run_len = 1

    if run_len >= max_flat_seconds:
        flat_mask[len(x)-run_len:len(x)] = True

    segments = _runs(flat_mask)

    return flat_mask, segments

def fhr_preprocess(

    fhr: np.ndarray,
    lo: float = 50.0,
    hi: float = 210.0,
    spike_window: int = 11,
    spike_thresh_bpm: float = 25.0,
    flat_seconds: int = 3,
    flat_atol: float = 0.1,

):

    # Clamp -> Spike Mask -> Dropout Detection
    x1, m_out = clamp_out_of_range(fhr, lo, hi)
    x2, m_spk = mask_spikes(x1, window=spike_window, threshold_bpm=spike_thresh_bpm)
    m_flat, flat_segments = dropout_detection(x2, max_flat_seconds=flat_seconds, atol=flat_atol)

    x_out = x2.copy()
    x_out[m_flat] = np.nan

    # Uncomment to return masks
    # masks = {
    #     "out_of_range": m_out,
    #     "spikes": m_spk,
    #     "flatline": m_flat,
    #     "flat_segments": flat_segments,
    #     "artifact_mask_any": m_out | m_spk | m_flat,
    # }

    return x_out

def coverage_pct(x: np.ndarray) -> float:

    if x.size == 0:
        return 0.0

    return float(np.isfinite(x).mean() * 100.0)

def make_tracks(fhr_with_nans):

    raw_light = linear_fill_short_gaps(fhr_with_nans, max_len=3)

    x = np.copy(raw_light)
    x[~np.isfinite(x)] = np.nanmedian(x)

    filtered = butter_filter(x, 0.01, "high")
    filtered = butter_filter(filtered, 0.3, "low")

    return raw_light, filtered

def fhr_accel_decel_features(x, accel_thr=15.0, decel_thr=-15.0, min_len=15):

    k = 31
    pad = k//2
    xp = np.pad(x, (pad, pad), mode="edge")
    baseline = np.array([np.median(xp[i:i+k]) for i in range(len(x))])

    d = x - baseline
    a_mask = d >= accel_thr
    d_mask = d <= decel_thr

    def segments(mask):
        m = np.concatenate([[False], mask, [False]])
        idx = np.flatnonzero(m[1:] != m[:-1])
        return [(s, e) for s, e in zip(idx[0::2], idx[1::2]) if (e - s) >= min_len]

    acc = segments(a_mask)
    dec = segments(d_mask)

    def summarize(segs):
        if not segs:
            return dict(count=0, amp_mean=0.0, dur_mean=0.0, auc_mean=0.0)
        amps, durs, aucs = [], [], []
        for s, e in segs:
            seg = d[s:e]
            amps.append(np.max(seg))
            durs.append(e - s)
            aucs.append(np.trapz(seg, dx=1.0))
        return dict(count=len(segs),
                    amp_mean=float(np.mean(amps)),
                    dur_mean=float(np.mean(durs)),
                    auc_mean=float(np.mean(aucs)))

    acc_d = summarize(acc)
    dec_d = summarize(dec)

    return {
        "fhr_accel_count"       : acc_d["count"],
        "fhr_accel_amp_mean"    : acc_d["amp_mean"],
        "fhr_accel_dur_mean"    : acc_d["dur_mean"],
        "fhr_accel_auc_mean"    : acc_d["auc_mean"],
        "fhr_decel_count"       : dec_d["count"],
        "fhr_decel_amp_mean"    : dec_d["amp_mean"],
        "fhr_decel_dur_mean"    : dec_d["dur_mean"],
        "fhr_decel_auc_mean"    : dec_d["auc_mean"],
    }

def fhr_welch_features(x: np.ndarray, fs: float = FS):

    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return {
            "psd_total": np.nan, "psd_vlf": np.nan, "psd_lf": np.nan, "psd_hf": np.nan,
            "ratio_lf_hf": np.nan, "ratio_vlf_total": np.nan
        }

    med = np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0
    x = np.nan_to_num(x, nan=med, posinf=med, neginf=med)

    nperseg = min(256, len(x))
    if nperseg < 64:

        return {
            "psd_total": np.nan, "psd_vlf": np.nan, "psd_lf": np.nan, "psd_hf": np.nan,
            "ratio_lf_hf": np.nan, "ratio_vlf_total": np.nan
        }

    noverlap = nperseg // 2
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    f, pxx = welch(
        x, fs=fs, window="hamming",
        nperseg=nperseg, noverlap=noverlap,
        detrend="constant", scaling="density", average="mean"
    )

    bands = {"vlf": (0.00, 0.03), "lf": (0.03, 0.15), "hf": (0.15, 0.50)}

    def band_power(lo, hi):
        idx = (f >= lo) & (f < hi)
        if not np.any(idx): return 0.0
        return np.trapz(pxx[idx], f[idx])

    vlf = band_power(*bands["vlf"])
    lf  = band_power(*bands["lf"])
    hf  = band_power(*bands["hf"])

    total = vlf + lf + hf

    return {
        "psd_total": total,
        "psd_vlf": vlf, "psd_lf": lf, "psd_hf": hf,
        "ratio_lf_hf": (lf / hf) if hf > 0 else np.nan,
        "ratio_vlf_total": (vlf / total) if total > 0 else np.nan,
    }

def fhr_time_features(x):

    v = x[np.isfinite(x)]

    if v.size < 2:
        return dict(fhr_mean=np.nan, fhr_median=np.nan, fhr_sdnn=np.nan,
                    fhr_rmssd=np.nan, fhr_range=np.nan, fhr_skew=np.nan, fhr_kurt=np.nan)

    dif = np.diff(v)
    mean = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    rmssd = float(np.sqrt(np.mean(dif**2))) if dif.size else 0.0
    r = float(np.max(v) - np.min(v))
    skew = float(((v-mean)**3).mean() / (sd**3 + 1e-12)) if sd>0 else 0.0
    kurt = float(((v-mean)**4).mean() / (sd**4 + 1e-12)) if sd>0 else 0.0

    return dict(fhr_mean=mean, fhr_median=float(np.median(v)),
                fhr_sdnn=sd, fhr_rmssd=rmssd, fhr_range=r,
                fhr_skew=skew, fhr_kurt=kurt)

def extract_fhr_features(

    fhr_clean: np.ndarray,
    window_s: int = 120,
    stride_s: int = 30,
    coverage_thresh: float = 0.8,
    accel_thr: float = 15.0,
    decel_thr: float = -15.0,

):

    raw_light, filtered = make_tracks(fhr_clean)
    rows = [] ; n = len(fhr_clean)

    for s, e in sliding_windows(n, w=window_s, s=stride_s):

        win_raw = raw_light[s:e]
        win_flt = filtered[s:e]
        cov = float(np.isfinite(win_raw).mean())

        if cov < coverage_thresh:
            continue

        feats = dict(t_start=int(s), t_end=int(e), coverage=cov)
        feats.update(fhr_time_features(win_raw))
        feats.update(fhr_accel_decel_features(win_raw, accel_thr=accel_thr, decel_thr=decel_thr))
        feats.update(fhr_welch_features(win_flt))

        rows.append(feats)

    return rows