from utils.uc import *
from utils.fhr import *

import numpy as np
import pandas as pd
from datetime import datetime

from typing import Optional, Any

def process_row(row, target):

    # 2 fields
    m_datetime = datetime.strptime(row['measurement_date'], '%Y-%m-%d %H:%M:%S')
    b_datetime = datetime.strptime(row['onset'], '%Y-%m-%d %H:%M')\
    if target == 'onset' else datetime.strptime(row['add'], '%Y-%m-%d %H:%M')

    target = (b_datetime-m_datetime).total_seconds()/60/60/24
    if target < 0:
        return None

    # 8 fields
    static_data = [
        # row['age'] if (row['age'] is not None and pd.notna(row['age'])) else 0,
        # row['bmi'] if (row['bmi'] is not None and pd.notna(row['bmi'])) else 0,
        row['age'],
        row['bmi'],
        row['had_pregnancy'],
        row['had_preterm'],
        row['had_surgery'],
        row['gdm'],
        row['pih'],
        row['gest_age']
    ]

    # 2 fields
    uc  = np.array(row['uc']).astype(float)
    fhr = np.array(row['fhr']).astype(float)

    if len(uc) > 2048:
        return None

    uc_padded = pad_signal(uc) ; fhr_padded = pad_signal(fhr)

    uc_windows  = extract_uc_features(uc_preprocess(uc_padded))
    fhr_windows = extract_fhr_features(fhr_preprocess(fhr_padded))

    record = {
        'mobile'            : row['mobile'],
        'measurement_date'  : row['measurement_date'],
        'static'            : static_data,
        'uc_raw'            : uc_padded.tolist(),
        'fhr_raw'           : fhr_padded.tolist(),
        'uc_windows'        : uc_windows,
        'fhr_windows'       : fhr_windows,
        'target'            : target
    }

    return record

def bmi_choose_weight_kg(height_cm: Any, weight_val: Any) -> Optional[float]:
    """
    Resolve 斤 vs kg:
      - If weight > 110 → treat as 斤 (kg = x * 0.5)
      - Else compute BMI for both kg and 斤 and pick the one within [15, 45].
        If both plausible or both implausible, default to kg when <= 110.
    """

    def _try_float(x: Any) -> Optional[float]:
        try:
            return float(str(x).strip())
        except Exception as e:
            print(e)
            return None

    h_cm = pd.to_numeric(height_cm, errors="coerce")
    w = _try_float(weight_val)
    if pd.isna(h_cm) or h_cm <= 0 or w is None:
        return None

    h_m = h_cm / 100.0
    kg_if_kg = w
    kg_if_jin = w * 0.5

    def _bmi(kg: Optional[float]) -> Optional[float]:
        return (kg / (h_m ** 2)) if (kg and h_m > 0) else None

    b1 = _bmi(kg_if_kg)
    b2 = _bmi(kg_if_jin)

    def plausible(b: Optional[float]) -> bool:
        return (b is not None) and (15.0 <= b <= 45.0)

    if w > 110:
        return round(b2, 1) if b2 is not None else None
    if plausible(b1) and not plausible(b2):
        return round(b1, 1)
    if plausible(b2) and not plausible(b1):
        return round(b2, 1)
    return round(b1, 1) if b1 is not None else None