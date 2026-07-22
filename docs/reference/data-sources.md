---
type: reference
status: active
tags:
  - reference
  - data
  - integrations
---

# Reference: Data Sources

Canonical list of the external data the project ingests, and which subsystem
consumes each. Ingestion mechanics are in [[data-pipeline]]; live-signal sources
are in [[intelligence-suite]].

| Source | Provides | Used by |
|--------|----------|---------|
| **FPL API** | Current-season players, fixtures, per-player history, prices, ownership | [[data-pipeline]] (Stage 1), [[intelligence-suite]] (`intel_01`) |
| **Vaastav** (historical + repo) | Multi-season historical gameweek data (per-fixture) | [[data-pipeline]] (Stage 2), [[cross-season-harness]] |
| **Understat** | Team xG / xGA | [[data-pipeline]] (Stage 3 team form) |
| **FBref** | New-signing & debutant stats | [[data-pipeline]] (Stage 4a/4b) |
| **Transfermarkt** | Signing identification | [[data-pipeline]] (Stage 4) |
| **Press sites** (Fantasy Football Scout + Tier 3/4) | Team-news / press-conference text | [[intelligence-suite]] (`intel_02`), [[press_scraper_redesign]] |
| **LiveFPL top-10k** | Effective ownership of elite managers | [[intelligence-suite]] (`intel_08`) |

## Notes
- LLM-based sources (press extraction, recommendations) call external APIs — see
  [[external-apis]].
- On this clone the raw data is absent and the FPL API cannot be re-fetched; see
  [[environment-and-docker]].
- `intel_08` effective ownership is forward-only and cannot be backfilled.

## Related Source Files
- `pipeline/data_fetcher_stage1.py`, `data_loader_stage2.py`, `team_form_stage3.py`
- `pipeline/new_signings_stage4a.py`, `data_loader_stage4b.py`
- `pipeline/intel_01_fpl_live.py`, `intel_02_sources.py`, `intel_08_effective_ownership.py`

---
Hubs: [[system-overview]] · [[data-flow]] · [[repository-map]]
