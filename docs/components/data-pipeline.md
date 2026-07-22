---
type: component
status: active
tags:
  - component
  - data
---

# Data Pipeline

The ingestion and preparation subsystem: it collects raw football data from
several external sources and shapes it into the tables the rest of the system
consumes. Corresponds to **Stages 1–4** in [[data-flow]].

## Responsibility
Fetch and normalize current-season and historical data:
- **Stage 1** — current-season FPL API (players, fixtures, per-player history).
- **Stage 2** — Vaastav historical gameweek data (multiple past seasons).
- **Stage 3** — team form built from Vaastav results plus Understat xG.
- **Stage 4a/4b** — new-signing stats from FBref and debutant previous-league
  stats (with Transfermarkt used to identify signings).

Stage 5 (matchup stats) was **dropped** — [`CLAUDE.md`](../../CLAUDE.md) records
it as insufficient signal.

## Why it exists
The models need a clean, leakage-free, multi-season history keyed consistently
across sources. No single public source provides FPL points *and* xG *and*
new-signing form, so the pipeline stitches them together and lands them in
`data/raw/` and `data/processed/`.

## How it interacts
It is the head of the flow in [[data-flow]]: its outputs feed
[[feature-engineering]], which in turn feeds [[prediction-models]]. It does not
call any downstream component.

## Depends on
- External sources only (FPL API, Vaastav, FBref, Understat, Transfermarkt) —
  catalogued in [[data-sources]].

## Depended on by
- [[feature-engineering]] (consumes the raw/processed tables).
- [[intelligence-suite]] indirectly — `intel_01` fetches live FPL API data on the
  same footing, though it is documented as part of the intelligence subsystem.

## Assumptions & limitations
- **Data absent on this clone** — `data/raw/` is gitignored and the FPL API cannot
  be re-fetched; see [[environment-and-docker]]. Cross-season inputs are rebuilt
  via [[cross-season-harness]].
- Source scrapes (FBref, Understat) are point-in-time and not re-validated by
  this note.

## Related Source Files
- `pipeline/data_fetcher_stage1.py`
- `pipeline/data_loader_stage2.py`
- `pipeline/team_form_stage3.py`
- `pipeline/new_signings_stage4a.py`, `pipeline/data_loader_stage4b.py`

---
Hubs: [[system-overview]] · [[data-flow]] · [[repository-map]]
