import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class JoinResult:
    """Represents a fuzzy join mapping."""

    source_name: str
    matched_name: Optional[str]
    score: float


def normalize_account_name(name: str) -> str:
    """Normalize account names for robust joins across messy sources."""
    if pd.isna(name):
        return ""
    text = str(name).lower().strip()
    # Remove punctuation and legal suffixes that are often inconsistent.
    text = re.sub(r"[&]", " and ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(inc|llc|ltd|corp|corporation|co|company|gmbh|sa|plc)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fuzzy_match_name(name: str, candidates: Iterable[str], threshold: float = 0.78) -> Tuple[Optional[str], float]:
    """Find the best fuzzy candidate for a given name."""
    best_name = None
    best_score = 0.0
    for candidate in candidates:
        score = SequenceMatcher(None, name, candidate).ratio()
        if score > best_score:
            best_name = candidate
            best_score = score
    if best_score < threshold:
        return None, best_score
    return best_name, best_score


def build_name_map(source_names: Iterable[str], canonical_names: Iterable[str], threshold: float = 0.78) -> List[JoinResult]:
    """Map a set of source names to canonical names with fuzzy matching."""
    canonical_list = list(canonical_names)
    results: List[JoinResult] = []
    for raw_name in source_names:
        normalized = normalize_account_name(raw_name)
        direct = next((c for c in canonical_list if c == normalized), None)
        if direct:
            results.append(JoinResult(raw_name, direct, 1.0))
            continue
        matched, score = fuzzy_match_name(normalized, canonical_list, threshold=threshold)
        results.append(JoinResult(raw_name, matched, score))
    return results


def safe_json_loads(text: str, default):
    """Load JSON safely and return a default value if decoding fails."""
    try:
        return json.loads(text)
    except Exception:
        return default


def parse_date_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Parse a dataframe date column while preserving NaT for bad values."""
    return pd.to_datetime(df[col], errors="coerce", utc=True)


def filter_renewals_next_90_days(accounts_df: pd.DataFrame, renewal_col: str = "renewal_date") -> pd.DataFrame:
    """Keep accounts with renewals in the next 90 days from now (UTC)."""
    now = datetime.utcnow()
    upper = now + timedelta(days=90)
    dates = parse_date_column(accounts_df, renewal_col)
    mask = dates.dt.tz_localize(None).between(now, upper, inclusive="both")
    out = accounts_df.loc[mask].copy()
    out[renewal_col] = dates.loc[mask].dt.tz_localize(None)
    return out


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)
