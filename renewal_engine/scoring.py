from __future__ import annotations

from typing import Dict

import pandas as pd


RISK_WEIGHTS = {
    "usage_drop": 0.25,
    "ticket_severity": 0.20,
    "nps": 0.15,
    "csm": 0.20,
    "changelog": 0.20,
}


def _norm_clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def changelog_risk_for_account(account_row: pd.Series, changelog_signals: Dict) -> float:
    """Map account product-version fields to changelog deprecation/breaking risk."""
    version = str(account_row.get("sdk_version", "")).lower()
    has_not_upgraded = bool(account_row.get("not_upgraded", False))

    risk = 0.0
    if "v3" in version:
        risk = max(risk, 1.0)
    if has_not_upgraded:
        risk = max(risk, 0.6)

    rules = changelog_signals.get("risk_rules", []) if isinstance(changelog_signals, dict) else []
    for rule in rules:
        pattern = str(rule.get("pattern", "")).lower()
        if pattern and pattern in f"uses sdk {version}":
            r = str(rule.get("risk", "medium")).lower()
            risk = max(risk, {"low": 0.3, "medium": 0.6, "high": 1.0}.get(r, 0.6))
    return _norm_clip(risk)


def compute_risk_scores(df: pd.DataFrame, changelog_signals: Dict) -> pd.DataFrame:
    out = df.copy()

    out["usage_drop_component"] = out["usage_trend"].apply(lambda t: _norm_clip(-t))

    p1_ratio = (out["p1_tickets"] / out["total_tickets"].replace(0, 1)).fillna(0)
    unresolved_factor = (out["unresolved_tickets"] / 10.0).fillna(0)
    out["ticket_component"] = (p1_ratio + unresolved_factor).clip(lower=0, upper=1)

    nps_map = {"positive": 0.1, "neutral": 0.5, "negative": 1.0}
    out["nps_component"] = out["nps_comment_sentiment"].map(nps_map).fillna(0.5)

    csm_map = {"low": 0.2, "medium": 0.6, "high": 1.0}
    out["csm_component"] = out["csm_risk_level"].astype(str).str.lower().map(csm_map).fillna(0.5)

    out["changelog_component"] = out.apply(lambda r: changelog_risk_for_account(r, changelog_signals), axis=1)

    out["risk_score"] = (
        out["usage_drop_component"] * RISK_WEIGHTS["usage_drop"]
        + out["ticket_component"] * RISK_WEIGHTS["ticket_severity"]
        + out["nps_component"] * RISK_WEIGHTS["nps"]
        + out["csm_component"] * RISK_WEIGHTS["csm"]
        + out["changelog_component"] * RISK_WEIGHTS["changelog"]
    )

    def category(score: float) -> str:
        if score > 0.6:
            return "HIGH"
        if score >= 0.3:
            return "MEDIUM"
        return "LOW"

    out["risk_category"] = out["risk_score"].map(category)
    return out


def detect_non_obvious_insights(df: pd.DataFrame) -> pd.DataFrame:
    insights = []
    median_usage = float(df["avg_api_usage"].median()) if "avg_api_usage" in df.columns and len(df) else 0.0
    for _, r in df.iterrows():
        account = r.get("account_name", r.get("account_name_norm", "Unknown"))
        if r.get("nps_score", 0) >= 9 and r.get("risk_score", 0) > 0.6:
            insights.append({"account": account, "insight": "Silent churn risk: strong NPS but overall high renewal risk."})
        if r.get("avg_api_usage", 0) > median_usage and r.get("csm_component", 0) > 0.8:
            insights.append({"account": account, "insight": "High product usage but poor relationship health from CSM signals."})
        if r.get("active_user_pct", 0) < 0.25 and r.get("licensed_users", 0) >= 100:
            insights.append({"account": account, "insight": "Low adoption with large seat footprint: downgrade risk likely."})

    if not insights:
        insights.append({"account": "GLOBAL", "insight": "No advanced patterns triggered; monitor week-over-week trend shifts."})
    return pd.DataFrame(insights)
