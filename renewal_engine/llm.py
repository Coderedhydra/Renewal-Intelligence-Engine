import json
from typing import Any, Dict, List

import requests

from .utils import safe_json_loads

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"


class OllamaClient:
    """Small wrapper around Ollama's local generate API."""

    def __init__(self, model: str = DEFAULT_MODEL, timeout: int = 90):
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, temperature: float = 0.1, fmt: str | None = None) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if fmt:
            payload["format"] = fmt

        response = requests.post(OLLAMA_URL, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def parse_csm_notes(self, notes_text: str) -> List[Dict[str, Any]]:
        prompt = f"""
You are a B2B renewal risk analyst.
Parse the following CSM notes into a JSON array. Return only valid JSON.
Schema per item:
{{
  "account_name": "",
  "risk_level": "low|medium|high",
  "issues": [],
  "competitors": [],
  "sentiment": "",
  "key_flags": []
}}
Also detect explicit/implicit mentions of: budget cuts, competitor mentions, executive involvement,
migration issues, and churn intent phrases.

CSM notes:
{notes_text}
"""
        raw = self.generate(prompt, temperature=0.0)
        parsed = safe_json_loads(raw, [])
        return parsed if isinstance(parsed, list) else []

    def analyze_nps_comment(self, comment: str) -> Dict[str, Any]:
        prompt = f"""
Analyze this NPS comment. If not English, translate to English.
Extract concise JSON only:
{{
  "translated_comment": "",
  "sentiment": "positive|neutral|negative",
  "risk_clues": []
}}
Comment:
{comment}
"""
        raw = self.generate(prompt, temperature=0.0)
        parsed = safe_json_loads(raw, {})
        return parsed if isinstance(parsed, dict) else {}

    def analyze_changelog(self, changelog_markdown: str) -> Dict[str, Any]:
        prompt = f"""
You are reviewing product changelogs for renewal risk.
Return JSON only with this schema:
{{
  "breaking_changes": [{{"feature": "", "impact": ""}}],
  "deprecations": [{{"feature": "", "version": "", "impact": ""}}],
  "migration_requirements": [{{"from": "", "to": "", "risk": "low|medium|high", "notes": ""}}],
  "risk_rules": [
    {{"pattern": "uses SDK v3", "risk": "high", "reason": "SDK v3 deprecated"}}
  ]
}}
Changelog:
{changelog_markdown}
"""
        raw = self.generate(prompt, temperature=0.0)
        parsed = safe_json_loads(raw, {})
        return parsed if isinstance(parsed, dict) else {}

    def explain_account(self, account_payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
Given this account risk payload, produce concise JSON with exactly 3 reasons and 2 actions.
JSON schema:
{{
  "reasons": ["", "", ""],
  "actions": ["", ""]
}}
Payload:
{json.dumps(account_payload, indent=2)}
"""
        raw = self.generate(prompt, temperature=0.2)
        parsed = safe_json_loads(raw, {})
        if not isinstance(parsed, dict):
            return {"reasons": [], "actions": []}
        return parsed
