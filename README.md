# Signal Preprocessing & Feature Extraction

This module provides a complete preprocessing and feature extraction pipeline for Uterine Contraction (UC) and Fetal Heart Rate (FHR) signals.

Converts raw noisy recordings into a structured feature table suitable for downstream machine learning.

---

## 📦 Overview

**Input**

* Raw UC, FHR signasl: 1 Hz NumPy array (each value = bpm/uterine pressure at that second)

* FHR: Contains out-of-range values, spikes, and flatlines

* UC: Contains probe artifacts and baseline drifts

**Output**

* FHR

  * Cleaned FHR array (artifacts replaced by NaN)

  * One row per valid 120 s window (stride = 30 s) with **time-domain, event-based, and frequency-domain features**

* UC

  * Detrended UC signal (zero-centered)

  * Feature table: One row per valid 120 s window (stride=30) with **time-domain and frequency-domain features**

---

## 🧹 Preprocessing

**Fetal Heart Rate**

1. Clamp out-of-range values

    * Sets values outside physiologic range (50–210 bpm) to NaN

2. Spike masking

    * Uses a moving median filter to detect transient spikes ≥ ±25 bpm relative to the local median

3. Dropout detection

   * Marks regions where the signal is constant within ±0.1 bpm for ≥ 3 s

4. Combine masks and produce cleaned signal

   * Final cleaned signal with NaN in artifact regions
   
   * Combined mask

**Uterine Contractions**

1. Low-pass filter

    * Zero-phase Butterworth filter to retain slow contractions (≤ 6 cpm ≈ 0.1 Hz)

2. Median filter

   * Removes isolated spikes and abrupt noise

3. Baseline removal

   * Subtracts local baseline (resting uterine tone) to isolate contractions 
     
   * After baseline subtraction, UC becomes zero-centered

---

### 🧩 FHR Feature Extraction

**Track Generation**

1. Short-gap interpolation: Linearly fill NaN gaps ≤ 3 s (optional)

2. Median fill for remaining NaNs

3. Butterworth filtering

   * High-pass 0.01 Hz (remove slow drift)
   
   * Low-pass 0.3 Hz (remove high-frequency noise)
   
4. Returns:

   * `raw_light`: Lightly filled raw track for event detection (accelerations, decelerations)
   
   * `filtered`: Smoothed band-limited signal for frequency-domain analysis

**Sliding-Window Extraction**

* Window length: 120 s

* Stride: 30 s

* Windows accepted only if `coverage ≥ 0.8`

### 📈 FHR Feature Categories

**Time-Domain Features**

| Name                     | Description                                   |
| ------------------------ | --------------------------------------------- |
| `fhr_mean`, `fhr_median` | Central tendency                              |
| `fhr_sdnn`               | Standard deviation of NN intervals (1 Hz FHR) |
| `fhr_rmssd`              | Root mean square of successive differences    |
| `fhr_range`              | Max–min                                       |
| `fhr_skew`, `fhr_kurt`   | Distribution shape                            |

**Event-Based Features**

| Name                                       | Description                             |
| ------------------------------------------ | --------------------------------------- |
| `fhr_accel_count`, `fhr_decel_count`       | Number of accelerations / decelerations |
| `fhr_accel_amp_mean`, `fhr_decel_amp_mean` | Mean amplitude (bpm)                    |
| `fhr_accel_dur_mean`, `fhr_decel_dur_mean` | Mean duration (s)                       |
| `fhr_accel_auc_mean`, `fhr_decel_auc_mean` | Mean area under curve (bpm · s)         |

**Frequency-Domain Features**

| Name                          | Description                                    |
| ----------------------------- | ---------------------------------------------- |
| `psd_vlf`, `psd_lf`, `psd_hf` | Power in 0–0.03, 0.03–0.15, 0.15–0.50 Hz bands |
| `psd_total`                   | Total (VLF + LF + HF)                          |
| `ratio_lf_hf`                 | LF / HF ratio                                  |
| `ratio_vlf_total`             | VLF / total ratio                              |

**Default Parameters**

| Parameter                | Default      | Purpose                 |
| ------------------------ | ------------ | ----------------------- |
| `lo`, `hi`               | 50, 210 bpm  | Clamp physiologic range |
| `spike_window`           | 11 s         | Local median window     |
| `spike_thresh_bpm`       | 25 bpm       | Spike threshold         |
| `flat_seconds`           | 3 s          | Dropout threshold       |
| `flat_atol`              | 0.1 bpm      | Flatline tolerance      |
| `window_s`               | 120 s        | Feature window length   |
| `stride_s`               | 30 s         | Sliding stride          |
| `coverage_thresh`        | 0.8          | Minimum valid fraction  |
| `accel_thr`, `decel_thr` | +15, –15 bpm | Accel/decel thresholds  |

---

### 🧩 UC Feature Extraction

**Contraction Detection**

* Minimum distance between peaks (typically 100 s), enforces physiological spacing between contractions

* Multiplier for prominence threshold

* Uses adaptive thresholds based on the local Median Absolute Deviation (MAD) to handle amplitude variability across patients and devices

* Returns:

  * `peaks`: Indices of detected contractions

  * `props`: Peak properties (`prominences`, `left_bases`, `right_bases`, etc.)

**Sliding-Window Extraction**

* Window length: 120 s

* Stride: 30 s

### 📈 UC Feature Categories

**Time-Domain Features**

| Feature                                    | Description                                                         |
| ------------------------------------------ | ------------------------------------------------------------------- |
| `uc_n_contr`                               | Number of contractions in window                                    |
| `uc_amp_mean`, `uc_amp_sd`                 | Mean and SD of contraction amplitudes                               |
| `uc_dur_mean`, `uc_dur_sd`                 | Mean and SD of contraction durations                                |
| `uc_ici_mean`, `uc_ici_sd`                 | Mean and SD of inter-contraction intervals                          |
| `uc_duty_cycle`                            | Fraction of time in active contraction (Σ duration / window length) |
| `uc_rise_slope_mean`, `uc_fall_slope_mean` | Mean contraction rise/fall rates                                    |
| `uc_area_mean`                             | Mean area under contraction curve (integrated pressure over time)   |

**Frequency-Domain Features**

| Feature           | Description                                         |
| ----------------- | --------------------------------------------------- |
| `uc_dom_freq`     | Dominant frequency in 0.003–0.02 Hz (≈0.2–1.2 cpm)  |
| `uc_power_sub003` | Integrated power below 0.03 Hz (slow contractions)  |
| `uc_total_power`  | Total power below 0.1 Hz                            |
| `uc_spec_entropy` | Spectral entropy (distribution uniformity of power) |

**Default Parameters**

| Parameter                | Default     | Purpose                                |
| ------------------------ | ----------- | -------------------------------------- |
| `lpf_cutoff_hz`          | 0.1         | Low-pass cutoff (Hz)                   |
| `median_k`, `baseline_k` | 11, 61      | Spike / baseline median filter windows |
| `gap_fill_s`             | 5 s         | Max linear fill length                 |
| `min_distance_s`         | 100 s       | Minimum inter-contraction distance     |
| `prom_scale_mad`         | 1.0         | Prominence = `MAD × scale`             |
| `window_s`, `stride_s`   | 120 s, 30 s | Window / stride                        |
| `coverage_thresh`        | 0.8         | Valid coverage threshold               |

---
