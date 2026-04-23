from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .llm import OllamaClient
from .utils import (
    build_name_map,
    filter_renewals_next_90_days,
    load_csv,
    normalize_account_name,
    read_text_file,
)


@dataclass
class EngineData:
    accounts: pd.DataFrame
    usage: pd.DataFrame
    tickets: pd.DataFrame
    nps: pd.DataFrame
    csm_signals: pd.DataFrame
    changelog_signals: Dict


def _apply_name_mapping(df: pd.DataFrame, account_col: str, canonical_names: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["account_name_norm"] = df[account_col].fillna("").map(normalize_account_name)
    mapping = build_name_map(df[account_col].fillna(""), canonical_names)
    map_df = pd.DataFrame([{"source": m.source_name, "match": m.matched_name} for m in mapping])
    df = df.merge(map_df, left_on=account_col, right_on="source", how="left")
    df["account_name_norm"] = df["match"].fillna(df["account_name_norm"])
    return df.drop(columns=["source", "match"])


def load_all_inputs(data_dir: str | Path, llm: OllamaClient) -> EngineData:
    base = Path(data_dir)
    accounts = load_csv(base / "accounts.csv")
    usage = load_csv(base / "usage_metrics.csv")
    tickets = load_csv(base / "support_tickets.csv")
    nps = load_csv(base / "nps_responses.csv")
    csm_notes_text = read_text_file(base / "csm_notes.txt")
    changelog_md = read_text_file(base / "changelog.md")

    accounts["account_name_norm"] = accounts["account_name"].map(normalize_account_name)
    canonical_names = accounts["account_name_norm"].dropna().unique().tolist()

    usage = _apply_name_mapping(usage, "account_name", canonical_names)
    tickets = _apply_name_mapping(tickets, "account_name", canonical_names)
    nps = _apply_name_mapping(nps, "account_name", canonical_names)

    # Missing value handling.
    usage = usage.fillna({"api_calls": 0, "active_users": 0, "licensed_users": 0})
    tickets = tickets.fillna({"severity": "P3", "status": "open"})
    nps = nps.fillna({"score": 0, "comment": ""})

    csm_json = llm.parse_csm_notes(csm_notes_text) if csm_notes_text else []
    csm_signals = pd.DataFrame(csm_json)
    if csm_signals.empty:
        csm_signals = pd.DataFrame(
            columns=["account_name", "risk_level", "issues", "competitors", "sentiment", "key_flags"]
        )
    csm_signals["account_name_norm"] = csm_signals["account_name"].fillna("").map(normalize_account_name)

    changelog_signals = llm.analyze_changelog(changelog_md) if changelog_md else {}

    renewals_90 = filter_renewals_next_90_days(accounts, renewal_col="renewal_date")

    return EngineData(
        accounts=renewals_90,
        usage=usage,
        tickets=tickets,
        nps=nps,
        csm_signals=csm_signals,
        changelog_signals=changelog_signals,
    )


def build_usage_features(usage_df: pd.DataFrame) -> pd.DataFrame:
    u = usage_df.copy()
    u["date"] = pd.to_datetime(u.get("date"), errors="coerce")
    u = u.sort_values(["account_name_norm", "date"])

    grouped = []
    for acc, g in u.groupby("account_name_norm", dropna=False):
        g = g.dropna(subset=["date"]) if "date" in g.columns else g
        avg_usage = g["api_calls"].mean() if not g.empty else 0
        first = g["api_calls"].iloc[0] if len(g) > 0 else 0
        last = g["api_calls"].iloc[-1] if len(g) > 0 else 0
        trend = (last - first) / first if first and first != 0 else 0
        active_pct = (g["active_users"] / g["licensed_users"].replace(0, pd.NA)).mean()
        grouped.append(
            {
                "account_name_norm": acc,
                "avg_api_usage": float(avg_usage or 0),
                "usage_trend": float(trend or 0),
                "active_user_pct": float(active_pct if pd.notna(active_pct) else 0),
            }
        )
    return pd.DataFrame(grouped)


def build_ticket_features(ticket_df: pd.DataFrame) -> pd.DataFrame:
    t = ticket_df.copy()
    agg = (
        t.groupby("account_name_norm")
        .agg(
            total_tickets=("account_name_norm", "size"),
            p1_tickets=("severity", lambda s: (s.astype(str).str.upper() == "P1").sum()),
            unresolved_tickets=("status", lambda s: (~s.astype(str).str.lower().isin(["resolved", "closed"]).values).sum()),
        )
        .reset_index()
    )
    return agg


def build_nps_features(nps_df: pd.DataFrame, llm: OllamaClient) -> pd.DataFrame:
    n = nps_df.copy()
    n["score"] = pd.to_numeric(n["score"], errors="coerce").fillna(0)

    def score_to_sentiment(score: float) -> str:
        if score < 7:
            return "negative"
        if score <= 8:
            return "neutral"
        return "positive"

    n["nps_sentiment"] = n["score"].map(score_to_sentiment)

    llm_sentiments = []
    llm_risk_clues = []
    for _, row in n.iterrows():
        comment = str(row.get("comment", "")).strip()
        if not comment:
            llm_sentiments.append(row["nps_sentiment"])
            llm_risk_clues.append([])
            continue
        analysis = llm.analyze_nps_comment(comment)
        llm_sentiments.append(analysis.get("sentiment", row["nps_sentiment"]))
        llm_risk_clues.append(analysis.get("risk_clues", []))

    n["nps_comment_sentiment"] = llm_sentiments
    n["nps_risk_clues"] = llm_risk_clues

    agg = (
        n.groupby("account_name_norm")
        .agg(
            nps_score=("score", "mean"),
            nps_sentiment=("nps_sentiment", lambda s: s.mode().iloc[0] if not s.mode().empty else "neutral"),
            nps_comment_sentiment=(
                "nps_comment_sentiment",
                lambda s: s.mode().iloc[0] if not s.mode().empty else "neutral",
            ),
            nps_risk_clues=("nps_risk_clues", lambda s: sum((x for x in s if isinstance(x, list)), [])),
        )
        .reset_index()
    )
    return agg


def build_csm_features(csm_df: pd.DataFrame) -> pd.DataFrame:
    c = csm_df.copy()
    if c.empty:
        return pd.DataFrame(columns=["account_name_norm", "csm_risk_level", "csm_issues", "competitors", "key_flags"])

    def _flatten(series: pd.Series) -> List[str]:
        out: List[str] = []
        for item in series:
            if isinstance(item, list):
                out.extend([str(i) for i in item])
            elif isinstance(item, str) and item.strip():
                out.append(item.strip())
        return sorted(set(out))

    agg = (
        c.groupby("account_name_norm")
        .agg(
            csm_risk_level=("risk_level", lambda s: s.mode().iloc[0] if not s.mode().empty else "medium"),
            csm_issues=("issues", _flatten),
            competitors=("competitors", _flatten),
            key_flags=("key_flags", _flatten),
        )
        .reset_index()
    )
    return agg
