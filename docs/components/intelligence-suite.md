---
type: component
status: active
tags:
  - component
  - intel
---

# Intelligence Suite

The pre-deadline intelligence subsystem (`intel_01`–`intel_08`): it gathers
real-world signals before each deadline and converts them into per-player
adjustments that make squad decisions injury- and rotation-aware. This is the
"pre-deadline intelligence" layer of [[system-overview]].

## Responsibility
Produce, per player per gameweek, an **availability** score and a **rotation
risk** score (plus supporting signals), which the [[season-simulator]] turns into
prediction multipliers. The stages:

| Stage | Produces |
|------|----------|
| `intel_01` | Live FPL data — injuries, prices, ownership, transfer pressure |
| `intel_02` | Press-conference intelligence (multi-source scraper; see below) |
| `intel_03` | Availability 0–100 = 65% press + 35% FPL (+5 if both agree) |
| `intel_04` | Rotation risk 0–100 (start rate, minutes volatility, bench rate, trend, press keywords) |
| `intel_05` | LLM recommendations — captain/differentials/transfers/risks (Gemini) |
| `intel_06` | Intel-injected optimizer run (reuses Stage 8 ILP) → `final_squad.json` |
| `intel_07` | Bench intelligence — bench-boost targeting and bench-candidate scoring |
| `intel_08` | Effective ownership from the top-10k (per-GW EO archive) |

## Why it exists
Models trained on historical stats cannot know Friday's press conference. This
suite injects fresh, human-world information (who is injured, who will be rested)
so the optimizer avoids players who won't feature — the single biggest source of
"predicted but scored zero" errors.

## How it interacts
The suite's per-player scores become prediction multipliers in the
[[season-simulator]] (`adjusted = pred × avail_mult × rot_mult`; exact tier
thresholds in [[tuned-parameters]]). The collection sequence is the
[[intelligence-gathering]] workflow. `intel_05` (Gemini) and the `intel_02` LLM
extractor are the intelligence half of [[llm-layers]]; `intel_08` EO and the
availability/rotation outputs are surfaced by the [[web-ui]]. Design rationale:
[[recommendation_layer]]; scraper design: [[press_scraper_redesign]].

## Depends on
- External sources (FPL API via `intel_01`, press sites via `intel_02`,
  LiveFPL top-10k via `intel_08`) and the LLM providers for `intel_05`/`intel_02`.

## Depended on by
- [[season-simulator]] (availability/rotation multipliers).
- [[llm-layers]] (shares the LLM-calling stages).
- [[web-ui]] (availability/rotation warning icons).

## Assumptions & limitations
- **Press-scraper popularity bias** — clubs absent from article headers can have
  injuries missed (the Newcastle case); a cross-club fallback was tried and
  reverted. Full detail in [[known-limitations]].
- `intel_08` EO cannot be backfilled — only forward snapshots exist.
- Recommendation coverage in `intel_05` is per-GW and historically scoped to
  GW1–10 in the shipped output.

## Related Source Files
- `pipeline/intel_01_fpl_live.py` … `pipeline/intel_08_effective_ownership.py`
- `pipeline/intel_02_scrape.py`, `intel_02_sources.py`, `intel_02_ledger.py`,
  `intel_02_llm_extract.py`, `pipeline/intel_identity.py`
- `data/intel/availability.json`, `rotation_risk.json`, `effective_ownership.json`

---
Hubs: [[system-overview]] · [[data-flow]] · [[repository-map]]
