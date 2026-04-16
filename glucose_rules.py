from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class RuleConfig:
    """
    Lightweight rule configuration grounded in the paper's logic and adapted for
    a non-clinical glucose insight project.

    Paper-aligned logic retained:
    - Focus on dynamic glucose behavior, especially postprandial excursions.
    - Use 140 mg/dL as a prediabetic-range postprandial threshold.
    - Use 200 mg/dL as a diabetic-range postprandial threshold.
    - Analyze meal-centered windows first, then summarize at subject level.

    Lightweight simplification:
    - No DTW / spectral clustering / formal glucotype discovery.
    - No medical diagnosis; only risk reminders or control-pattern summaries.
    """

    prediabetes_range_threshold: float = 140.0
    diabetic_range_threshold: float = 200.0
    low_glucose_threshold: float = 70.0

    # Pragmatic engineering thresholds for a lightweight project.
    spike_delta_threshold: float = 50.0
    slow_recovery_gap_threshold: float = 20.0
    meal_cv_high: float = 0.25
    daily_cv_high: float = 0.20

    tar140_attention_threshold: float = 0.10
    tar180_attention_threshold: float = 0.05
    tbr70_attention_threshold: float = 0.03
    persistent_high_mean_threshold_known_diabetes: float = 160.0
    persistent_high_mean_threshold_general_user: float = 125.0

    repeated_attention_events: int = 2
    repeated_screening_events: int = 2
    repeated_hypo_events: int = 1


GENERAL_RISK_LABELS = {
    "low_risk",
    "attention_needed",
    "screening_recommended",
}

KNOWN_DIABETES_RISK_LABELS = {
    "relatively_stable",
    "needs_attention",
    "needs_review",
}

PATTERN_LABELS = {
    "stable",
    "postprandial_spike",
    "slow_recovery",
    "persistently_high",
    "high_variability",
    "hypoglycemia_risk",
}


def _boolish(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t", "diabetic"}



def _safe_float(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan



def classify_meal_pattern(
    row: pd.Series,
    known_diabetes: bool = False,
    config: RuleConfig = RuleConfig(),
) -> str:
    """
    Classify one meal-centered response pattern.

    Inputs expected in row:
    - baseline
    - peak_glucose
    - peak_delta
    - recovery_gap
    - recovery_120
    - nadir_glucose
    - cv_glucose_post
    - pct_below_70
    - pct_above_140
    - pct_above_180
    - pct_above_200
    """
    baseline = _safe_float(row.get("baseline"))
    peak_glucose = _safe_float(row.get("peak_glucose"))
    peak_delta = _safe_float(row.get("peak_delta"))
    recovery_gap = _safe_float(row.get("recovery_gap"))
    recovery_120 = _safe_float(row.get("recovery_120"))
    nadir_glucose = _safe_float(row.get("nadir_glucose"))
    cv_post = _safe_float(row.get("cv_glucose_post"))
    pct_below_70 = _safe_float(row.get("pct_below_70"))
    pct_above_140 = _safe_float(row.get("pct_above_140"))
    pct_above_180 = _safe_float(row.get("pct_above_180"))
    pct_above_200 = _safe_float(row.get("pct_above_200"))

    low_event = (
        (not np.isnan(nadir_glucose) and nadir_glucose < config.low_glucose_threshold)
        or (not np.isnan(pct_below_70) and pct_below_70 > 0)
    )
    if low_event:
        return "hypoglycemia_risk"

    persistent_high = False
    if known_diabetes:
        persistent_high = (
            (not np.isnan(baseline) and baseline >= config.prediabetes_range_threshold)
            and (
                (not np.isnan(recovery_120) and recovery_120 >= 180)
                or (not np.isnan(pct_above_180) and pct_above_180 >= 0.5)
            )
        )
    else:
        persistent_high = (
            (not np.isnan(baseline) and baseline >= 126)
            and (
                (not np.isnan(recovery_120) and recovery_120 >= config.prediabetes_range_threshold)
                or (not np.isnan(pct_above_140) and pct_above_140 >= 0.5)
            )
        )
    if persistent_high:
        return "persistently_high"

    spike_event = (
        (not np.isnan(peak_glucose) and peak_glucose >= config.prediabetes_range_threshold)
        or (not np.isnan(peak_delta) and peak_delta >= config.spike_delta_threshold)
    )

    diabetic_range_excursion = (
        (not np.isnan(peak_glucose) and peak_glucose >= config.diabetic_range_threshold)
        or (not np.isnan(pct_above_200) and pct_above_200 > 0)
    )

    slow_recovery = (
        spike_event
        and (not np.isnan(recovery_gap))
        and recovery_gap > config.slow_recovery_gap_threshold
    )

    high_variability = (
        (not np.isnan(cv_post) and cv_post >= config.meal_cv_high)
        and (
            spike_event
            or (not np.isnan(pct_above_180) and pct_above_180 > 0)
        )
    )

    if diabetic_range_excursion and high_variability:
        return "high_variability"
    if diabetic_range_excursion and slow_recovery:
        return "slow_recovery"
    if high_variability:
        return "high_variability"
    if slow_recovery:
        return "slow_recovery"
    if spike_event:
        return "postprandial_spike"
    return "stable"



def classify_meal_risk(
    row: pd.Series,
    known_diabetes: bool = False,
    config: RuleConfig = RuleConfig(),
) -> str:
    """
    Risk label for one meal event.

    General users:
    - low_risk / attention_needed / screening_recommended

    Users with known diabetes:
    - relatively_stable / needs_attention / needs_review
    """
    peak_glucose = _safe_float(row.get("peak_glucose"))
    recovery_gap = _safe_float(row.get("recovery_gap"))
    pct_above_200 = _safe_float(row.get("pct_above_200"))
    pct_above_180 = _safe_float(row.get("pct_above_180"))
    nadir_glucose = _safe_float(row.get("nadir_glucose"))

    diabetic_range = (
        (not np.isnan(peak_glucose) and peak_glucose >= config.diabetic_range_threshold)
        or (not np.isnan(pct_above_200) and pct_above_200 > 0)
    )
    elevated_range = (
        (not np.isnan(peak_glucose) and peak_glucose >= config.prediabetes_range_threshold)
        or (not np.isnan(pct_above_180) and pct_above_180 > 0)
    )
    low_event = (not np.isnan(nadir_glucose) and nadir_glucose < config.low_glucose_threshold)
    slow_recovery = (not np.isnan(recovery_gap) and recovery_gap > config.slow_recovery_gap_threshold)

    if known_diabetes:
        if low_event or diabetic_range or (not np.isnan(pct_above_180) and pct_above_180 >= 0.25):
            return "needs_review"
        if elevated_range or slow_recovery:
            return "needs_attention"
        return "relatively_stable"

    if diabetic_range:
        return "screening_recommended"
    if elevated_range or slow_recovery:
        return "attention_needed"
    return "low_risk"



def explain_meal_rule_trigger(
    row: pd.Series,
    known_diabetes: bool = False,
    config: RuleConfig = RuleConfig(),
) -> str:
    """Short evidence string for one meal event."""
    parts = []
    peak_glucose = _safe_float(row.get("peak_glucose"))
    peak_delta = _safe_float(row.get("peak_delta"))
    recovery_gap = _safe_float(row.get("recovery_gap"))
    cv_post = _safe_float(row.get("cv_glucose_post"))
    nadir_glucose = _safe_float(row.get("nadir_glucose"))

    if not np.isnan(peak_glucose):
        if peak_glucose >= config.diabetic_range_threshold:
            parts.append("post-meal peak reached >=200 mg/dL")
        elif peak_glucose >= config.prediabetes_range_threshold:
            parts.append("post-meal peak reached >=140 mg/dL")
    if not np.isnan(peak_delta) and peak_delta >= config.spike_delta_threshold:
        parts.append("post-meal rise was large")
    if not np.isnan(recovery_gap) and recovery_gap > config.slow_recovery_gap_threshold:
        parts.append("2-hour recovery remained above baseline")
    if not np.isnan(cv_post) and cv_post >= config.meal_cv_high:
        parts.append("post-meal variability was high")
    if not np.isnan(nadir_glucose) and nadir_glucose < config.low_glucose_threshold:
        parts.append("low-glucose signal appeared")
    if not parts:
        parts.append("post-meal response stayed relatively stable")
    return "; ".join(parts)



def enrich_meal_features_with_rules(
    meal_features_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    config: RuleConfig = RuleConfig(),
) -> pd.DataFrame:
    """
    Add meal-level pattern labels, risk labels, and evidence strings.
    """
    if meal_features_df.empty:
        return meal_features_df.copy()

    meal_df = meal_features_df.copy()
    profile_sub = profile_df[[c for c in ["subject_id", "known_diabetes"] if c in profile_df.columns]].copy()
    if "known_diabetes" not in profile_sub.columns:
        profile_sub["known_diabetes"] = False

    meal_df = meal_df.merge(profile_sub, on="subject_id", how="left")
    meal_df["known_diabetes"] = meal_df["known_diabetes"].map(_boolish)

    meal_patterns = []
    meal_risks = []
    triggers = []

    for row in meal_df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        kd = _boolish(row_s.get("known_diabetes"))
        pattern = classify_meal_pattern(row_s, known_diabetes=kd, config=config)
        risk = classify_meal_risk(row_s, known_diabetes=kd, config=config)
        trigger = explain_meal_rule_trigger(row_s, known_diabetes=kd, config=config)
        meal_patterns.append(pattern)
        meal_risks.append(risk)
        triggers.append(trigger)

    meal_df["meal_pattern"] = meal_patterns
    meal_df["meal_risk"] = meal_risks
    meal_df["meal_rule_trigger"] = triggers
    return meal_df



def classify_daily_pattern(
    row: pd.Series,
    known_diabetes: bool = False,
    config: RuleConfig = RuleConfig(),
) -> str:
    mean_glucose = _safe_float(row.get("mean_glucose"))
    tar_140 = _safe_float(row.get("tar_140"))
    tar_180 = _safe_float(row.get("tar_180"))
    tbr_70 = _safe_float(row.get("tbr_70"))
    cv_glucose = _safe_float(row.get("cv_glucose"))
    num_spikes = _safe_float(row.get("num_spikes"))

    if not np.isnan(tbr_70) and tbr_70 >= config.tbr70_attention_threshold:
        return "hypoglycemia_risk"

    if known_diabetes:
        if (
            (not np.isnan(mean_glucose) and mean_glucose >= config.persistent_high_mean_threshold_known_diabetes)
            or (not np.isnan(tar_180) and tar_180 >= config.tar180_attention_threshold)
        ):
            return "persistently_high"
    else:
        if (
            (not np.isnan(mean_glucose) and mean_glucose >= config.persistent_high_mean_threshold_general_user)
            and (not np.isnan(tar_140) and tar_140 >= config.tar140_attention_threshold)
        ):
            return "persistently_high"

    if not np.isnan(cv_glucose) and cv_glucose >= config.daily_cv_high:
        return "high_variability"

    if (not np.isnan(num_spikes) and num_spikes >= 2) or (not np.isnan(tar_140) and tar_140 >= config.tar140_attention_threshold):
        return "postprandial_spike"

    return "stable"



def summarize_subject_from_rules(
    meal_rule_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    profile_row: pd.Series,
    config: RuleConfig = RuleConfig(),
) -> Dict[str, Any]:
    """
    Create a subject-level summary from meal-level and day-level rule outputs.
    """
    subject_id = str(profile_row.get("subject_id"))
    known_diabetes = _boolish(profile_row.get("known_diabetes"))
    glycemic_status = profile_row.get("glycemic_status", np.nan)
    paper_glucotype = profile_row.get("glucotype", np.nan)

    meal_sub = meal_rule_df[meal_rule_df["subject_id"].astype(str) == subject_id].copy()
    daily_sub = daily_df[daily_df["subject_id"].astype(str) == subject_id].copy() if not daily_df.empty else pd.DataFrame()

    meal_pattern_counts = meal_sub["meal_pattern"].value_counts(dropna=True).to_dict() if not meal_sub.empty else {}
    dominant_pattern = max(meal_pattern_counts, key=meal_pattern_counts.get) if meal_pattern_counts else "insufficient_data"

    n_meals = int(len(meal_sub))
    meals_ge_140 = int((meal_sub.get("peak_glucose", pd.Series(dtype=float)) >= config.prediabetes_range_threshold).sum()) if not meal_sub.empty else 0
    meals_ge_200 = int((meal_sub.get("peak_glucose", pd.Series(dtype=float)) >= config.diabetic_range_threshold).sum()) if not meal_sub.empty else 0
    slow_recovery_count = int((meal_sub.get("meal_pattern", pd.Series(dtype=object)) == "slow_recovery").sum()) if not meal_sub.empty else 0
    high_variability_count = int((meal_sub.get("meal_pattern", pd.Series(dtype=object)) == "high_variability").sum()) if not meal_sub.empty else 0
    hypo_count = int((meal_sub.get("meal_pattern", pd.Series(dtype=object)) == "hypoglycemia_risk").sum()) if not meal_sub.empty else 0

    mean_daily_glucose = float(daily_sub["mean_glucose"].mean()) if not daily_sub.empty else np.nan
    mean_daily_cv = float(daily_sub["cv_glucose"].mean()) if not daily_sub.empty else np.nan
    mean_tar_140 = float(daily_sub["tar_140"].mean()) if (not daily_sub.empty and "tar_140" in daily_sub.columns) else np.nan
    mean_tar_180 = float(daily_sub["tar_180"].mean()) if (not daily_sub.empty and "tar_180" in daily_sub.columns) else np.nan
    mean_tbr_70 = float(daily_sub["tbr_70"].mean()) if (not daily_sub.empty and "tbr_70" in daily_sub.columns) else np.nan

    user_group = "known_diabetes_user" if known_diabetes else "general_user"

    if known_diabetes:
        if hypo_count >= config.repeated_hypo_events or meals_ge_200 >= config.repeated_attention_events or (not np.isnan(mean_tar_180) and mean_tar_180 >= config.tar180_attention_threshold):
            overall_risk = "needs_review"
        elif slow_recovery_count >= config.repeated_attention_events or high_variability_count >= config.repeated_attention_events or (not np.isnan(mean_daily_cv) and mean_daily_cv >= config.daily_cv_high):
            overall_risk = "needs_attention"
        else:
            overall_risk = "relatively_stable"
    else:
        if meals_ge_200 >= config.repeated_screening_events:
            overall_risk = "screening_recommended"
        elif meals_ge_140 >= config.repeated_attention_events or slow_recovery_count >= config.repeated_attention_events or high_variability_count >= config.repeated_attention_events:
            overall_risk = "attention_needed"
        else:
            overall_risk = "low_risk"

    evidence_parts = []
    if n_meals > 0:
        evidence_parts.append(f"analyzed {n_meals} meals")
    if meals_ge_140 > 0:
        evidence_parts.append(f"{meals_ge_140} meals had peak >=140 mg/dL")
    if meals_ge_200 > 0:
        evidence_parts.append(f"{meals_ge_200} meals had peak >=200 mg/dL")
    if slow_recovery_count > 0:
        evidence_parts.append(f"{slow_recovery_count} meals showed slow recovery")
    if high_variability_count > 0:
        evidence_parts.append(f"{high_variability_count} meals showed high variability")
    if hypo_count > 0:
        evidence_parts.append(f"{hypo_count} meals showed low-glucose risk")
    if not np.isnan(mean_daily_glucose):
        evidence_parts.append(f"mean daily glucose was about {mean_daily_glucose:.1f} mg/dL")
    if not np.isnan(mean_daily_cv):
        evidence_parts.append(f"mean daily CV was about {mean_daily_cv:.2f}")

    return {
        "subject_id": subject_id,
        "known_diabetes": known_diabetes,
        "user_group": user_group,
        "glycemic_status": glycemic_status,
        "paper_glucotype": paper_glucotype,
        "dominant_pattern": dominant_pattern,
        "overall_risk": overall_risk,
        "n_meals_analyzed": n_meals,
        "meals_ge_140": meals_ge_140,
        "meals_ge_200": meals_ge_200,
        "slow_recovery_count": slow_recovery_count,
        "high_variability_count": high_variability_count,
        "hypoglycemia_count": hypo_count,
        "mean_daily_glucose": mean_daily_glucose,
        "mean_daily_cv": mean_daily_cv,
        "mean_tar_140": mean_tar_140,
        "mean_tar_180": mean_tar_180,
        "mean_tbr_70": mean_tbr_70,
        "key_evidence": "; ".join(evidence_parts) if evidence_parts else "insufficient_data",
    }



def generate_advice(
    pattern: str,
    overall_risk: str,
    known_diabetes: bool = False,
) -> str:
    """
    Generate lightweight, non-diagnostic advice.
    """
    if pattern == "stable":
        if known_diabetes:
            return (
                "Glucose responses look relatively stable overall. Continue routine monitoring, "
                "regular meals, and usual activity. If more high or low events appear later, review "
                "them together with your care plan."
            )
        return (
            "Glucose dynamics look relatively stable overall. Continue regular meals, moderate "
            "activity, and basic monitoring."
        )

    if pattern == "postprandial_spike":
        if known_diabetes:
            return (
                "The main issue is post-meal elevation. Focus first on refined carbohydrates, breakfast "
                "structure, and light movement after meals. If this pattern repeats, review post-meal "
                "management with your clinician."
            )
        return (
            "The main issue is post-meal elevation. Start with reducing refined carbohydrates, adding "
            "fiber/protein, and doing light activity after meals. If it repeats, consider formal glucose "
            "screening."
        )

    if pattern == "slow_recovery":
        if known_diabetes:
            return (
                "Post-meal recovery appears slow. Review total carbohydrate load, meal order, and "
                "post-meal activity. If this remains common, discuss whether your management plan needs "
                "adjustment."
            )
        return (
            "Post-meal recovery appears slow, suggesting reduced efficiency in handling meal-related "
            "glucose load. Consider lowering high-glycemic meals and, if this pattern repeats, consider "
            "HbA1c or OGTT."
        )

    if pattern == "persistently_high":
        if known_diabetes:
            return (
                "Glucose appears elevated beyond isolated meal spikes. Review recent meals, activity, "
                "and treatment adherence, and discuss persistent elevation with your clinician."
            )
        return (
            "Glucose appears elevated beyond isolated meal spikes. This pattern deserves attention. "
            "Consider formal glucose evaluation such as fasting glucose, HbA1c, or OGTT."
        )

    if pattern == "high_variability":
        if known_diabetes:
            return (
                "The main issue is large fluctuation rather than a single high level. Track meal/events, "
                "look for triggers of big swings, and discuss recurrent variability with your clinician."
            )
        return (
            "The main issue is large fluctuation. Record possible triggers such as meal composition, "
            "timing, sleep, and activity. If high variability continues, consider further screening."
        )

    if pattern == "hypoglycemia_risk":
        if known_diabetes:
            return (
                "The most important concern is low-glucose risk. Review meal spacing, activity timing, "
                "and treatment-related factors, and discuss recurrent lows with your clinician soon."
            )
        return (
            "Low-glucose signals appeared. Review prolonged fasting, intense exercise, or insufficient "
            "meal intake. If low values repeat, seek professional advice."
        )

    if known_diabetes:
        return (
            "Continue reviewing meal responses, daily variability, and treatment context together. "
            "This system provides pattern summaries and does not replace medical diagnosis."
        )
    return (
        "Continue reviewing meal responses and day-level variability together. This system provides "
        "risk reminders and does not replace medical diagnosis."
    )



def build_risk_df(
    meal_features_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    config: RuleConfig = RuleConfig(),
) -> pd.DataFrame:
    """
    Build a subject-level risk and pattern summary table.
    """
    meal_rule_df = enrich_meal_features_with_rules(meal_features_df, profile_df, config=config)

    daily_aug = daily_df.copy()
    if not daily_aug.empty:
        profile_map = profile_df[[c for c in ["subject_id", "known_diabetes"] if c in profile_df.columns]].copy()
        if "known_diabetes" not in profile_map.columns:
            profile_map["known_diabetes"] = False
        daily_aug = daily_aug.merge(profile_map, on="subject_id", how="left")
        daily_aug["known_diabetes"] = daily_aug["known_diabetes"].map(_boolish)
        daily_aug["daily_pattern"] = [
            classify_daily_pattern(pd.Series(r._asdict()), known_diabetes=_boolish(getattr(r, "known_diabetes", False)), config=config)
            for r in daily_aug.itertuples(index=False)
        ]

    rows = []
    for row in profile_df.itertuples(index=False):
        profile_row = pd.Series(row._asdict())
        subject_summary = summarize_subject_from_rules(
            meal_rule_df=meal_rule_df,
            daily_df=daily_aug,
            profile_row=profile_row,
            config=config,
        )
        subject_summary["advice"] = generate_advice(
            pattern=subject_summary["dominant_pattern"],
            overall_risk=subject_summary["overall_risk"],
            known_diabetes=_boolish(subject_summary["known_diabetes"]),
        )
        rows.append(subject_summary)

    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)



def run_all_rules(
    meal_features_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    config: RuleConfig = RuleConfig(),
) -> Dict[str, pd.DataFrame]:
    """
    Convenience wrapper returning both meal-level and subject-level outputs.
    """
    meal_rule_df = enrich_meal_features_with_rules(meal_features_df, profile_df, config=config)
    risk_df = build_risk_df(meal_features_df, daily_df, profile_df, config=config)
    return {
        "meal_rule_df": meal_rule_df,
        "risk_df": risk_df,
    }
