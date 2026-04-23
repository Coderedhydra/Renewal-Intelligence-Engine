from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .llm import OllamaClient


def _fallback_reasons(row: pd.Series) -> List[str]:
    reasons: List[str] = []
    if row.get("usage_trend", 0) < -0.1:
        reasons.append("Usage is declining heading into renewal.")
    if row.get("p1_tickets", 0) > 0:
        reasons.append("There are high-severity P1 support incidents.")
    if str(row.get("nps_comment_sentiment", "")).lower() == "negative":
        reasons.append("NPS feedback sentiment is negative.")
    if row.get("csm_risk_level", "medium") == "high":
        reasons.append("CSM notes flag elevated commercial or relationship risk.")
    if row.get("changelog_component", 0) >= 0.6:
        reasons.append("Changelog indicates deprecation or migration risk exposure.")
    return (reasons + ["Composite risk model indicates elevated non-renewal likelihood."])[:3]


def _fallback_actions(row: pd.Series) -> List[str]:
    actions = ["Schedule executive-level renewal alignment call this week."]
    if row.get("changelog_component", 0) >= 0.6:
        actions.append("Assign migration support engineer and share upgrade plan.")
    else:
        actions.append("Run adoption workshop and define success milestones.")
    return actions[:2]


def generate_explanations(df: pd.DataFrame, llm: OllamaClient) -> pd.DataFrame:
    enriched: List[Dict] = []
    for _, row in df.iterrows():
        payload = {
            "account_name": row.get("account_name", ""),
            "risk_score": round(float(row.get("risk_score", 0)), 4),
            "risk_category": row.get("risk_category", "MEDIUM"),
            "signals": {
                "usage_trend": row.get("usage_trend", 0),
                "p1_tickets": row.get("p1_tickets", 0),
                "unresolved_tickets": row.get("unresolved_tickets", 0),
                "nps_score": row.get("nps_score", 0),
                "nps_sentiment": row.get("nps_comment_sentiment", "neutral"),
                "csm_risk_level": row.get("csm_risk_level", "medium"),
                "competitors": row.get("competitors", []),
                "key_flags": row.get("key_flags", []),
                "changelog_component": row.get("changelog_component", 0),
            },
        }
        try:
            llm_resp = llm.explain_account(payload)
        except Exception:
            llm_resp = {}

        reasons = llm_resp.get("reasons", []) if isinstance(llm_resp, dict) else []
        actions = llm_resp.get("actions", []) if isinstance(llm_resp, dict) else []

        if not isinstance(reasons, list) or len(reasons) < 3:
            reasons = _fallback_reasons(row)
        if not isinstance(actions, list) or len(actions) < 2:
            actions = _fallback_actions(row)

        enriched.append({
            **row.to_dict(),
            "reasons": reasons[:3],
            "actions": actions[:2],
        })

    return pd.DataFrame(enriched)
