"""
pipeline/intel_02_llm_extract.py
Stage B LLM extraction for free-text press articles (redesign §5.3).

Gemini Flash (same client/retry pattern as intel_05) with a strict JSON
contract, grounded with the club's actual FPL roster. Only invoked for club
sections where the stage-A regex found < 2 players; rows become claims
tagged extractor="llm" and flow through the same reconciler as every other
claim — the LLM never produces conclusions, only claims.

Both dependencies are optional: available() gates usage on GEMINI_API_KEY
and the google-genai package, and the FFS adapter degrades to regex-only
when the extractor is absent or exhausted (MAX_CALLS_PER_RUN cost bound).
"""

import json
import os
import re
import time

MODEL_ID          = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE       = 0
MAX_CALLS_PER_RUN = 40      # §5.3: <= ~40 short calls per GW
MAX_SECTION_CHARS = 6000

SYSTEM_PROMPT = """You extract player availability facts from Premier League team-news text.
You receive one club's section of a press-conference liveblog and that club's
player roster. Return ONLY a JSON array — one object per player whose
availability for the upcoming match is stated or implied — with exactly
these keys:
  "player": the player's name as written in the text
  "status": one of "out", "doubtful", "available", "suspended", "unknown"
  "injury": short lowercase injury/reason label ("hamstring", "knock"), or ""
  "quote":  the sentence from the text supporting the status (max 200 chars)
Rules:
- Only include players that appear in the provided roster.
- Do not guess: if the text says nothing about a player, omit them.
- If the text contains no availability information at all, return [].
- No prose, no markdown fences — the raw JSON array only."""


class GeminiExtractor:
    """extract_club() -> list of {player,status,injury,quote}; [] on failure."""

    def __init__(self, api_key: str | None = None,
                 max_calls: int = MAX_CALLS_PER_RUN, verbose: bool = True):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.max_calls = max_calls
        self.calls = 0
        self.verbose = verbose
        self._client = None

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            from google import genai            # noqa: F401
            return True
        except ImportError:
            return False

    def extract_club(self, club_name: str, section_text: str,
                     roster: list) -> list:
        if not self.available() or self.calls >= self.max_calls:
            return []
        self.calls += 1
        user_msg = (f"Club: {club_name}\n"
                    f"Roster: {', '.join(roster)}\n\n"
                    f"Text:\n{section_text[:MAX_SECTION_CHARS]}")
        try:
            return self._parse(self._call(user_msg))
        except Exception as e:                              # noqa: BLE001
            if self.verbose:
                print(f"    [llm] extraction failed for {club_name}: "
                      f"{str(e)[:120]}")
            return []

    # -- Gemini call (intel_05 retry pattern) ---------------------------------

    def _call(self, user_msg: str) -> str:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        delays = [0, 5, 15]
        last_err = None
        for delay in delays:
            if delay:
                time.sleep(delay)
            try:
                response = self._client.models.generate_content(
                    model=MODEL_ID,
                    contents=user_msg,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    ),
                )
                return response.text or ""
            except Exception as e:                          # noqa: BLE001
                last_err = e
                s = str(e)
                if any(k in s for k in ("503", "UNAVAILABLE", "429",
                                        "RESOURCE_EXHAUSTED")):
                    continue                                # retriable
                raise
        raise last_err

    @staticmethod
    def _parse(text: str) -> list:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "",
                      (text or "").strip(), flags=re.MULTILINE).strip()
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        try:
            rows = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        out = []
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict) and r.get("player"):
                out.append({
                    "player": str(r.get("player", "")).strip(),
                    "status": str(r.get("status", "unknown")).strip().lower(),
                    "injury": str(r.get("injury", "") or "").strip().lower(),
                    "quote":  str(r.get("quote", "") or "").strip(),
                })
        return out
