---
type: reference
status: active
tags:
  - reference
  - apis
  - llm
---

# Reference: External APIs

Canonical list of the third-party LLM APIs the project calls. Usage rationale and
how these fit the intelligence/narrative layers is in [[llm-layers]]. None of
these make squad decisions.

| API | Where | Model | Key params | Auth |
|-----|-------|-------|-----------|------|
| **Claude** (Anthropic) | Stage 9 narrative | `claude-sonnet-4-6` | `MAX_TOKENS=2500`, `TEMPERATURE=0`, 1 call/GW | Anthropic key in `.env` |
| **Gemini** (Google) | `intel_05` recommendations | `gemini-2.5-flash` | per-GW, structured JSON | `GEMINI_API_KEY` in `.env` |
| **Gemini** (Google) | `intel_02` press extraction | `gemini-2.5-flash` | `MAX_OUTPUT_TOKENS=2048`, `TEMPERATURE=0`, ≤40 calls/run | `GEMINI_API_KEY` in `.env` |

## Notes
- Values above are taken from the current source
  (`llm_agent_stage9.py`, `intel_05_recommendations.py`,
  `intel_02_llm_extract.py`); [`CLAUDE.md`](../../CLAUDE.md) lists an older Stage 9
  model id/token budget — treat the code as authoritative.
- The two Gemini uses share one client/retry pattern and one key. The press
  extractor is gated by `available()` and degrades to regex-only without the key
  or the `google-genai` package.
- Missing keys disable the affected layer but do not block the core
  predict/optimize pipeline.

## Related Source Files
- `pipeline/llm_agent_stage9.py`
- `pipeline/intel_05_recommendations.py`
- `pipeline/intel_02_llm_extract.py`

---
Hubs: [[system-overview]]
