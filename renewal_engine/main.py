from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .explain import generate_explanations
from .features import (
    build_csm_features,
    build_nps_features,
    build_ticket_features,
    build_usage_features,
    load_all_inputs,
)
from .llm import OllamaClient
from .scoring import compute_risk_scores, detect_non_obvious_insights


def assemble_feature_table(data_dir: str, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    llm = OllamaClient(model=model)
    engine_data = load_all_inputs(data_dir, llm)

    usage_f = build_usage_features(engine_data.usage)
    tickets_f = build_ticket_features(engine_data.tickets)
    nps_f = build_nps_features(engine_data.nps, llm)
    csm_f = build_csm_features(engine_data.csm_signals)

    base = engine_data.accounts.copy()
    base = base.merge(usage_f, on="account_name_norm", how="left")
    base = base.merge(tickets_f, on="account_name_norm", how="left")
    base = base.merge(nps_f, on="account_name_norm", how="left")
    base = base.merge(csm_f, on="account_name_norm", how="left")

    # Fill defaults after merging features.
    base = base.fillna(
        {
            "avg_api_usage": 0,
            "usage_trend": 0,
            "active_user_pct": 0,
            "total_tickets": 0,
            "p1_tickets": 0,
            "unresolved_tickets": 0,
            "nps_score": 0,
            "nps_sentiment": "neutral",
            "nps_comment_sentiment": "neutral",
            "nps_risk_clues": "[]",
            "csm_risk_level": "medium",
            "csm_issues": "[]",
            "competitors": "[]",
            "key_flags": "[]",
        }
    )

    scored = compute_risk_scores(base, engine_data.changelog_signals)
    explained = generate_explanations(scored, llm)
    insights = detect_non_obvious_insights(explained)
    return explained, insights


def print_cli_output(df: pd.DataFrame, insights: pd.DataFrame) -> None:
    for _, row in df.sort_values("risk_score", ascending=False).iterrows():
        print("-" * 40)
        print(f"Account: {row.get('account_name', 'Unknown')}")
        print(f"Risk: {row.get('risk_category', 'MEDIUM')}")
        print(f"Score: {row.get('risk_score', 0):.2f}\n")

        print("Reasons:")
        for reason in row.get("reasons", []):
            print(f"- {reason}")

        print("\nActions:")
        for action in row.get("actions", []):
            print(f"- {action}")
    print("-" * 40)

    print("\nNON-OBVIOUS INSIGHTS")
    print("-" * 40)
    for _, insight in insights.iterrows():
        print(f"- {insight['account']}: {insight['insight']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Renewal Risk Intelligence Engine")
    parser.add_argument("--data-dir", default=".", help="Directory containing required input files")
    parser.add_argument("--model", default="llama3", help="Ollama model name")
    parser.add_argument("--output", default="renewal_risk_scores.csv", help="CSV output path")
    args = parser.parse_args()

    explained, insights = assemble_feature_table(args.data_dir, args.model)
    print_cli_output(explained, insights)

    out_path = Path(args.output)
    explained.to_csv(out_path, index=False)
    print(f"\nSaved scored output to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
