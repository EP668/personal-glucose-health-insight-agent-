from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Column normalization helpers
# -----------------------------------------------------------------------------

_CGM_ALIASES = {
    "timestamp": [
        "timestamp",
        "displaytime",
        "display_time",
        "time",
        "datetime",
        "date_time",
    ],
    "glucose": [
        "glucose",
        "glucosevalue",
        "glucose_value",
        "glucose_mg_dl",
        "glucose_mg/dl",
        "value",
    ],
    "subject_id": [
        "subject_id",
        "subjectid",
        "userid",
        "userid",
        "id",
    ],
}

def _normalize_colname(col: str) -> str:
    return (
        str(col)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("/", "")
        .replace(".", "")
        .replace("_", "")
    )


def _rename_by_aliases(df: pd.DataFrame, alias_map: dict[str, list[str]]) -> pd.DataFrame:
    rename_dict = {}
    normalized_to_original = {_normalize_colname(c): c for c in df.columns}

    for canonical, aliases in alias_map.items():
        for alias in aliases:
            alias_norm = _normalize_colname(alias)
            if alias_norm in normalized_to_original:
                rename_dict[normalized_to_original[alias_norm]] = canonical
                break

    return df.rename(columns=rename_dict)

_MEAL_ALIASES = {
    "subject_id": ["subject_id", "subjectid", "userid", "user_id", "id"],
    "meal_time": ["meal_time", "mealtime", "time", "event_time"],
    "meal_type": ["meal_type", "mealtype", "meal", "meal_code"],
    "meal_label": ["meal_label", "mealname", "label"],
    "window_start": ["window_start"],
    "window_end": ["window_end"],
}

_PROFILE_ALIASES = {
    "subject_id": ["subject_id", "subjectid", "userid", "user_id", "id"],
    "age": ["age"],
    "sex": ["sex", "gender"],
    "known_diabetes": ["known_diabetes"],
    "glycemic_status": ["glycemic_status", "diagnosis"],
    "bmi": ["bmi"],
    "a1c": ["a1c", "hba1c"],
    "fbg": ["fbg", "fasting_blood_glucose", "fasting_glucose"],
    "ogtt_2hr": ["ogtt_2hr", "ogtt.2hr", "ogtt2hr"],
    "height": ["height"],
    "weight": ["weight"],
    "glucotype": ["glucotype"],
}

_STANDARD_MEAL_MAP = {
    "PB": "bread_peanut_butter",
    "CF": "cornflakes_milk",
    "BAR": "probar_bar",
}


@dataclass
class MealWindow:
    subject_id: str
    meal_time: pd.Timestamp
    meal_type: str
    meal_label: Optional[str]
    window_df: pd.DataFrame
    anchor_shift_min: int = 0


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")



def _rename_by_aliases(df: pd.DataFrame, aliases: Dict[str, List[str]]) -> pd.DataFrame:
    normalized = {_normalize_name(c): c for c in df.columns}
    rename_map: Dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            key = _normalize_name(candidate)
            if key in normalized:
                rename_map[normalized[key]] = canonical
                break
    return df.rename(columns=rename_map)



def _ensure_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")



def _coerce_subject_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()



def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    return float(numerator) / float(denominator)



def map_standardized_meal_code(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    s = str(value).strip()
    upper = s.upper()
    if upper in _STANDARD_MEAL_MAP:
        return _STANDARD_MEAL_MAP[upper]
    if upper.startswith("PB"):
        return _STANDARD_MEAL_MAP["PB"]
    if upper.startswith("CF"):
        return _STANDARD_MEAL_MAP["CF"]
    if upper.startswith("BAR"):
        return _STANDARD_MEAL_MAP["BAR"]
    return s.lower().replace(" ", "_")


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------


def load_cgm_csv(path: str | Path, subject_id: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 先自动识别并重命名论文原始列名
    # 例如 DisplayTime -> timestamp, GlucoseValue -> glucose, subjectId -> subject_id
    df = _rename_by_aliases(df, _CGM_ALIASES)

    # 再检查必需列
    _ensure_columns(df, ["timestamp", "glucose"], "CGM file")

    # subject_id 缺失时补一个默认值
    if "subject_id" not in df.columns:
        df["subject_id"] = "unknown" if subject_id is None else str(subject_id)

    # 类型处理
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
    df["subject_id"] = df["subject_id"].astype(str)

    # 清掉关键列缺失
    df = df.dropna(subset=["timestamp", "glucose"]).copy()

    # 排序
    df = df.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)
    return df



def load_meal_log(path: str | Path, subject_id: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _rename_by_aliases(df, _MEAL_ALIASES)

    if "meal_time" not in df.columns and "window_start" in df.columns:
        df["meal_time"] = df["window_start"]

    _ensure_columns(df, ["meal_time"], "Meal log")

    if "subject_id" not in df.columns:
        df["subject_id"] = "unknown" if subject_id is None else str(subject_id)

    if "meal_type" not in df.columns:
        if "meal_label" in df.columns:
            df["meal_type"] = df["meal_label"].map(map_standardized_meal_code)
        else:
            df["meal_type"] = "meal"

    if "meal_label" not in df.columns:
        df["meal_label"] = df["meal_type"]

    df["subject_id"] = _coerce_subject_id(df["subject_id"])
    df["meal_time"] = pd.to_datetime(df["meal_time"], errors="coerce")
    df["meal_type"] = df["meal_type"].map(map_standardized_meal_code)
    df["meal_label"] = df["meal_label"].astype(str).str.strip()

    df = df[["subject_id", "meal_time", "meal_type", "meal_label"]].dropna(subset=["meal_time"])
    return df.sort_values(["subject_id", "meal_time"]).reset_index(drop=True)



def load_profile(path: str | Path, subject_id: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _rename_by_aliases(df, _PROFILE_ALIASES)

    if "subject_id" not in df.columns:
        df["subject_id"] = "unknown" if subject_id is None else str(subject_id)

    df["subject_id"] = _coerce_subject_id(df["subject_id"])

    expected = [
        "age", "sex", "known_diabetes", "glycemic_status", "bmi", "a1c", "fbg",
        "ogtt_2hr", "height", "weight", "glucotype",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    if df["known_diabetes"].isna().all() and not df["glycemic_status"].isna().all():
        status = df["glycemic_status"].astype(str).str.lower().str.strip()
        df["known_diabetes"] = status.eq("diabetic")

    numeric_cols = ["age", "bmi", "a1c", "fbg", "ogtt_2hr", "height", "weight"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sex"] = df["sex"].astype("string")
    df["known_diabetes"] = df["known_diabetes"].map(lambda x: bool(x) if pd.notna(x) else np.nan)
    df["glycemic_status"] = df["glycemic_status"].astype("string")
    df["glucotype"] = df["glucotype"].astype("string")

    keep = ["subject_id"] + expected
    return df[keep].drop_duplicates(subset=["subject_id"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# CGM preprocessing
# -----------------------------------------------------------------------------


def clean_cgm(
    cgm_df: pd.DataFrame,
    freq: str = "5min",
    glucose_clip: Tuple[float, float] = (40, 400),
    interpolate_limit: int = 3,
    apply_smoothing: bool = True,
    smoothing_window: int = 3,
) -> pd.DataFrame:
    _ensure_columns(cgm_df, ["subject_id", "timestamp", "glucose"], "cgm_df")

    df = cgm_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
    df = df.dropna(subset=["timestamp", "glucose"])
    df = df.sort_values(["subject_id", "timestamp"])
    df = df.drop_duplicates(subset=["subject_id", "timestamp"], keep="last")
    df["glucose"] = df["glucose"].clip(*glucose_clip)

    frames: List[pd.DataFrame] = []
    for sid, group in df.groupby("subject_id", sort=False):
        g = group.set_index("timestamp")[["glucose"]].sort_index()
        g = g.resample(freq).mean()
        g["glucose"] = g["glucose"].interpolate(method="time", limit=interpolate_limit, limit_direction="both")

        if apply_smoothing and smoothing_window > 1:
            g["glucose"] = g["glucose"].rolling(window=smoothing_window, center=True, min_periods=1).median()

        g = g.reset_index()
        g.insert(0, "subject_id", sid)
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["glucose"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Meal window extraction
# -----------------------------------------------------------------------------


def extract_meal_window(
    cgm_df: pd.DataFrame,
    meal_time: pd.Timestamp,
    subject_id: str,
    meal_type: str = "meal",
    meal_label: Optional[str] = None,
    pre_minutes: int = 30,
    post_minutes: int = 120,
    anchor_shift_min: int = 0,
) -> MealWindow:
    """
    Extract one event-centered meal window.

    `anchor_shift_min` is a lightweight robustness trick: if meal logging is noisy,
    we can slightly shift the event anchor by a few minutes and compute partially
    overlapping windows around the same meal.
    """
    subject_id = str(subject_id)
    anchor_time = pd.Timestamp(meal_time) + pd.Timedelta(minutes=anchor_shift_min)
    start = anchor_time - pd.Timedelta(minutes=pre_minutes)
    end = anchor_time + pd.Timedelta(minutes=post_minutes)

    window_df = cgm_df.loc[
        (cgm_df["subject_id"] == subject_id)
        & (cgm_df["timestamp"] >= start)
        & (cgm_df["timestamp"] <= end),
        ["subject_id", "timestamp", "glucose"],
    ].copy()

    if not window_df.empty:
        window_df = window_df.sort_values("timestamp")
        window_df["relative_minutes"] = (window_df["timestamp"] - anchor_time).dt.total_seconds() / 60.0
        window_df["minutes_from_start"] = (window_df["timestamp"] - window_df["timestamp"].min()).dt.total_seconds() / 60.0
        window_df["meal_time"] = pd.Timestamp(meal_time)
        window_df["anchor_time"] = anchor_time
        window_df["meal_type"] = meal_type
        window_df["meal_label"] = meal_label if meal_label is not None else meal_type
        window_df["anchor_shift_min"] = anchor_shift_min

    return MealWindow(
        subject_id=subject_id,
        meal_time=pd.Timestamp(meal_time),
        meal_type=meal_type,
        meal_label=meal_label,
        window_df=window_df,
        anchor_shift_min=anchor_shift_min,
    )



def extract_all_meal_windows(
    cgm_df: pd.DataFrame,
    meals_df: pd.DataFrame,
    pre_minutes: int = 30,
    post_minutes: int = 120,
    min_points: int = 6,
    anchor_shifts: Optional[Iterable[int]] = None,
) -> List[MealWindow]:
    """
    Extract meal windows for all rows in meals_df.

    If `anchor_shifts` is provided (for example [-10, 0, 10]), the function
    returns multiple lightly overlapping windows for the same meal event.
    This is a lightweight substitute for the paper's heavier global overlapping
    sliding-window treatment and helps reduce boundary error around meal slicing.
    """
    _ensure_columns(cgm_df, ["subject_id", "timestamp", "glucose"], "cgm_df")
    _ensure_columns(meals_df, ["subject_id", "meal_time", "meal_type"], "meals_df")

    shifts = list(anchor_shifts) if anchor_shifts is not None else [0]
    windows: List[MealWindow] = []
    for row in meals_df.itertuples(index=False):
        for shift in shifts:
            mw = extract_meal_window(
                cgm_df=cgm_df,
                meal_time=row.meal_time,
                subject_id=row.subject_id,
                meal_type=row.meal_type,
                meal_label=getattr(row, "meal_label", row.meal_type),
                pre_minutes=pre_minutes,
                post_minutes=post_minutes,
                anchor_shift_min=int(shift),
            )
            if len(mw.window_df) >= min_points:
                windows.append(mw)
    return windows


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------


def _nearest_glucose_at_relative_minutes(
    window_df: pd.DataFrame,
    target_min: float,
    tolerance_min: float = 20.0,
) -> float:
    candidates = window_df.copy()
    candidates["abs_diff"] = (candidates["relative_minutes"] - target_min).abs()
    candidates = candidates[candidates["abs_diff"] <= tolerance_min]
    if candidates.empty:
        return np.nan
    idx = candidates["abs_diff"].idxmin()
    return float(window_df.loc[idx, "glucose"])



def compute_window_features(
    window_df: pd.DataFrame,
    baseline_window_minutes: int = 30,
    post_minutes: int = 120,
    thresholds: Tuple[float, float, float] = (140, 180, 200),
    low_threshold: float = 70.0,
) -> Dict[str, Any]:
    if window_df.empty:
        raise ValueError("window_df is empty")

    required = [
        "subject_id", "timestamp", "glucose", "relative_minutes", "meal_time",
        "meal_type", "meal_label",
    ]
    _ensure_columns(window_df, required, "window_df")

    w = window_df.sort_values("timestamp").copy()
    pre = w[(w["relative_minutes"] >= -baseline_window_minutes) & (w["relative_minutes"] < 0)]
    post = w[(w["relative_minutes"] >= 0) & (w["relative_minutes"] <= post_minutes)]

    if pre.empty or post.empty:
        raise ValueError("window_df does not contain both pre-meal and post-meal data")

    baseline = float(pre["glucose"].median())

    peak_idx = post["glucose"].idxmax()
    peak_glucose = float(post.loc[peak_idx, "glucose"])
    time_to_peak_min = float(post.loc[peak_idx, "relative_minutes"])
    peak_delta = peak_glucose - baseline

    nadir_glucose = float(post["glucose"].min())

    recovery_120 = _nearest_glucose_at_relative_minutes(w, target_min=120.0, tolerance_min=20.0)
    recovery_gap = recovery_120 - baseline if pd.notna(recovery_120) else np.nan

    rel_t = post["relative_minutes"].to_numpy(dtype=float)
    incr_g = post["glucose"].to_numpy(dtype=float) - baseline
    auc_incremental = float(np.trapezoid(incr_g, rel_t)) if len(post) >= 2 else np.nan
    auc_positive = float(np.trapezoid(np.clip(incr_g, a_min=0, a_max=None), rel_t)) if len(post) >= 2 else np.nan

    pct_above = {f"pct_above_{int(th)}": float((post["glucose"] > th).mean()) for th in thresholds}
    pct_below_70 = float((post["glucose"] < low_threshold).mean())

    mean_glucose = float(post["glucose"].mean())
    std_glucose = float(post["glucose"].std(ddof=0))
    cv_glucose = _safe_divide(std_glucose, mean_glucose)

    return {
        "subject_id": str(w["subject_id"].iloc[0]),
        "meal_time": pd.Timestamp(w["meal_time"].iloc[0]),
        "meal_type": str(w["meal_type"].iloc[0]),
        "meal_label": str(w["meal_label"].iloc[0]),
        "anchor_shift_min": int(w.get("anchor_shift_min", pd.Series([0])).iloc[0]),
        "n_points": int(len(w)),
        "baseline": baseline,
        "peak_glucose": peak_glucose,
        "peak_delta": peak_delta,
        "time_to_peak_min": time_to_peak_min,
        "nadir_glucose": nadir_glucose,
        "recovery_120": recovery_120,
        "recovery_gap": recovery_gap,
        "auc_incremental": auc_incremental,
        "auc_positive": auc_positive,
        "mean_glucose_post": mean_glucose,
        "std_glucose_post": std_glucose,
        "cv_glucose_post": cv_glucose,
        "pct_below_70": pct_below_70,
        **pct_above,
    }



def _aggregate_overlap_feature_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate lightly overlapping meal windows into one robust meal-level row.

    Strategy:
    - use medians for baseline/recovery/variability summaries
    - use maxima for peak-oriented risk features
    - use minima for nadir

    This keeps the project lightweight while reducing boundary sensitivity.
    """
    df = pd.DataFrame(rows)
    rep_idx = df[["peak_delta", "peak_glucose"]].fillna(-np.inf).sum(axis=1).idxmax()
    rep = df.loc[rep_idx].to_dict()

    out = {
        "subject_id": rep["subject_id"],
        "meal_time": rep["meal_time"],
        "meal_type": rep["meal_type"],
        "meal_label": rep["meal_label"],
        "n_overlap_windows": int(len(df)),
        "anchor_shift_min_used": ",".join(str(int(x)) for x in sorted(df["anchor_shift_min"].dropna().unique())),
        "n_points": int(df["n_points"].max()),
        "baseline": float(df["baseline"].median()),
        "peak_glucose": float(df["peak_glucose"].max()),
        "peak_delta": float(df["peak_delta"].max()),
        "time_to_peak_min": float(df["time_to_peak_min"].median()),
        "nadir_glucose": float(df["nadir_glucose"].min()),
        "recovery_120": float(df["recovery_120"].median()) if not df["recovery_120"].dropna().empty else np.nan,
        "recovery_gap": float(df["recovery_gap"].median()) if not df["recovery_gap"].dropna().empty else np.nan,
        "auc_incremental": float(df["auc_incremental"].median()) if not df["auc_incremental"].dropna().empty else np.nan,
        "auc_positive": float(df["auc_positive"].median()) if not df["auc_positive"].dropna().empty else np.nan,
        "mean_glucose_post": float(df["mean_glucose_post"].median()),
        "std_glucose_post": float(df["std_glucose_post"].median()),
        "cv_glucose_post": float(df["cv_glucose_post"].median()) if not df["cv_glucose_post"].dropna().empty else np.nan,
        "pct_below_70": float(df["pct_below_70"].max()),
        "pct_above_140": float(df["pct_above_140"].max()),
        "pct_above_180": float(df["pct_above_180"].max()),
        "pct_above_200": float(df["pct_above_200"].max()),
    }
    return out



def build_meal_features_df(
    cgm_df: pd.DataFrame,
    meals_df: pd.DataFrame,
    pre_minutes: int = 30,
    post_minutes: int = 120,
    min_points: int = 6,
    use_overlap: bool = True,
    overlap_anchor_shifts: Tuple[int, ...] = (-10, 0, 10),
) -> pd.DataFrame:
    """
    Build one robust row per meal event.

    Lightweight overlap logic:
    - If use_overlap=False: one event window per meal.
    - If use_overlap=True: build a few slightly shifted event windows around the
      same meal time (for example -10, 0, +10 minutes), compute features for each,
      and aggregate them into one meal-level row.

    This is intentionally much lighter than the paper's global overlapping sliding
    windows, but still reduces boundary error from hard slicing.
    """
    _ensure_columns(meals_df, ["subject_id", "meal_time", "meal_type"], "meals_df")

    rows: List[Dict[str, Any]] = []
    shifts = overlap_anchor_shifts if use_overlap else (0,)

    for row in meals_df.itertuples(index=False):
        feature_rows: List[Dict[str, Any]] = []
        for shift in shifts:
            mw = extract_meal_window(
                cgm_df=cgm_df,
                meal_time=row.meal_time,
                subject_id=row.subject_id,
                meal_type=row.meal_type,
                meal_label=getattr(row, "meal_label", row.meal_type),
                pre_minutes=pre_minutes,
                post_minutes=post_minutes,
                anchor_shift_min=int(shift),
            )
            if len(mw.window_df) < min_points:
                continue
            try:
                feature_rows.append(compute_window_features(mw.window_df, post_minutes=post_minutes))
            except ValueError:
                continue

        if not feature_rows:
            continue

        if use_overlap and len(feature_rows) > 1:
            rows.append(_aggregate_overlap_feature_rows(feature_rows))
        else:
            single = feature_rows[0]
            single["n_overlap_windows"] = 1
            single["anchor_shift_min_used"] = str(int(single.get("anchor_shift_min", 0)))
            rows.append(single)

    if not rows:
        return pd.DataFrame(columns=[
            "subject_id", "meal_time", "meal_type", "meal_label", "n_points",
            "n_overlap_windows", "anchor_shift_min_used", "baseline", "peak_glucose",
            "peak_delta", "time_to_peak_min", "nadir_glucose", "recovery_120",
            "recovery_gap", "auc_incremental", "auc_positive", "mean_glucose_post",
            "std_glucose_post", "cv_glucose_post", "pct_below_70", "pct_above_140",
            "pct_above_180", "pct_above_200",
        ])

    df = pd.DataFrame(rows)
    return df.sort_values(["subject_id", "meal_time"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Daily summaries
# -----------------------------------------------------------------------------


def _count_upward_crossings(values: pd.Series, threshold: float) -> int:
    above = values > threshold
    crossings = (~above.shift(1, fill_value=False)) & above
    return int(crossings.sum())



def build_daily_features_df(
    cgm_df: pd.DataFrame,
    tir_low: float = 70.0,
    tir_high: float = 140.0,
    tar_180_threshold: float = 180.0,
    spike_threshold: float = 140.0,
) -> pd.DataFrame:
    _ensure_columns(cgm_df, ["subject_id", "timestamp", "glucose"], "cgm_df")

    df = cgm_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    rows: List[Dict[str, Any]] = []
    for (sid, date), g in df.groupby(["subject_id", "date"], sort=True):
        g = g.sort_values("timestamp")
        glucose = g["glucose"].astype(float)

        mean_glucose = float(glucose.mean())
        std_glucose = float(glucose.std(ddof=0))
        cv_glucose = _safe_divide(std_glucose, mean_glucose)

        rows.append({
            "subject_id": str(sid),
            "date": pd.to_datetime(date),
            "n_points": int(len(g)),
            "mean_glucose": mean_glucose,
            "median_glucose": float(glucose.median()),
            "min_glucose": float(glucose.min()),
            "max_glucose": float(glucose.max()),
            "std_glucose": std_glucose,
            "cv_glucose": cv_glucose,
            "tir": float(((glucose >= tir_low) & (glucose <= tir_high)).mean()),
            "tar_140": float((glucose > tir_high).mean()),
            "tar_180": float((glucose > tar_180_threshold).mean()),
            "tbr_70": float((glucose < tir_low).mean()),
            "num_spikes": _count_upward_crossings(glucose, threshold=spike_threshold),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Convenience wrapper
# -----------------------------------------------------------------------------


def load_and_prepare_all(
    cgm_path: str | Path,
    meals_path: str | Path,
    profile_path: str | Path,
    clean: bool = True,
    use_overlap: bool = True,
    overlap_anchor_shifts: Tuple[int, ...] = (-10, 0, 10),
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end convenience loader for notebooks / demos.

    Returns:
    - cgm_df
    - meals_df
    - profile_df
    - meal_features_df
    - daily_df

    Overlap handling is lightweight and meal-centered. It does NOT reproduce the
    paper's heavy global sliding-window clustering, but it does help reduce hard
    slicing error around meal events.
    """
    cgm_df = load_cgm_csv(cgm_path)
    meals_df = load_meal_log(meals_path)
    profile_df = load_profile(profile_path)

    if clean:
        cgm_df = clean_cgm(cgm_df)

    meal_features_df = build_meal_features_df(
        cgm_df,
        meals_df,
        use_overlap=use_overlap,
        overlap_anchor_shifts=overlap_anchor_shifts,
    )
    daily_df = build_daily_features_df(cgm_df)

    return {
        "cgm_df": cgm_df,
        "meals_df": meals_df,
        "profile_df": profile_df,
        "meal_features_df": meal_features_df,
        "daily_df": daily_df,
    }
