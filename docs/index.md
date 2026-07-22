---
type: moc
status: active
tags:
  - moc
  - home
---

# FPL AI — Project Knowledge Base

Conceptual knowledge base for the **Fantasy Premier League Predictive Management
System** (FINKI thesis). This vault explains the project's *concepts,
architecture, workflows, and relationships* — it is a technical wiki, not
generated API documentation. It complements, and does not replace,
[`CLAUDE.md`](../CLAUDE.md) (operational/session memory) and the existing design
reports under `docs/`.

## What the project is

A **3-layer hybrid AI** with two supporting layers:

1. **Prediction** — four LightGBM models (GK/DEF/MID/FWD) predict player points.
2. **Optimization** — an ILP/MILP optimizer picks the squad, captain, and chips.
3. **Narrative** — an LLM (Claude) explains each gameweek's decisions.
4. **Pre-deadline intelligence** — live data, press, availability, and rotation
   signals that adjust predictions before each deadline.
5. **Web UI** — a Flask app that visualizes the simulated season.

See [[system-overview]] for the full picture.

## Start here

- [[system-overview]] — the layered architecture and how the pieces fit together.
- [[data-flow]] — how one gameweek's data moves from raw sources to a decision.
- [[repository-map]] — where each concern lives in the repository.

These three notes are the hubs of the graph. Every component, workflow, and
decision note links back to at least one of them.

## Components

The subsystems that make up the project (Phase 2):

- [[data-pipeline]] — ingest & prepare data (Stages 1–4)
- [[feature-engineering]] — build the position-split training files (Stage 6)
- [[prediction-models]] — four LightGBM models (Stage 7)
- [[legacy-ilp-optimizer]] — production PuLP squad optimizer (Stage 8)
- [[milp-optimizer]] — multi-period MILP redesign (in progress)
- [[season-simulator]] — the per-gameweek orchestrator
- [[chip-scheduler]] — chip timing policy (v2)
- [[intelligence-suite]] — pre-deadline intel (intel_01–08)
- [[llm-layers]] — Claude narrative + Gemini recommendations + press extraction
- [[hyperparameter-search]] — random + Optuna tuning
- [[cross-season-harness]] — generalization on unseen seasons
- [[web-ui]] — Flask dashboard

## Workflows

How the system behaves over time (Phase 3) — the bridge between architecture and
components:

- [[model-training|Model Training]] — build & retrain the four models
- [[season-simulation|Season Simulation]] — play a full GW1–38 season
- [[intelligence-gathering|Intelligence Gathering]] — collect pre-deadline signals
- [[hyperparameter-tuning|Hyperparameter Tuning]] — search model/strategy constants
- [[cross-season-generalization|Cross-Season Generalization]] — replay unseen seasons

## Decisions

Why the project is built the way it is (Phase 4, ADR-style):

- [[four-position-models|Four separate position models]]
- [[walkforward-no-leakage|Walk-forward validation & no leakage]]
- [[lightgbm-over-xgboost|LightGBM over XGBoost]]
- [[corrected-vs-legacy-rules|Corrected vs legacy FPL rules]]
- [[optimizer-redesign|Optimizer redesign (ILP → MILP)]]
- [[chip-strategy-v2|Chip Strategy v2]]
- [[known-limitations|Accepted known limitations]]

## Reference

Canonical homes for facts used across the vault (Phase 4):

- [[fpl-glossary|FPL glossary]] — domain terms
- [[data-sources|Data sources]] — external data & integrations
- [[external-apis|External APIs]] — Claude & Gemini usage
- [[environment-and-docker|Environment & Docker]] — how to run; why scores are env-bound
- [[tuned-parameters|Tuned parameters]] — simulator constants & flags
- [[evaluation-metrics-and-results|Evaluation metrics & results]] — MAE, benchmarks

## Documentation roadmap

Built incrementally. **Phases 1–4 are complete**:

```
docs/
├── index.md                     ✅ this note
├── architecture/                ✅ Phase 1 (system-overview, data-flow, repository-map)
├── components/                  ✅ Phase 2 (12 subsystem notes)
├── workflows/                   ✅ Phase 3 (5 end-to-end process notes)
├── decisions/                   ✅ Phase 4 (7 ADR-style notes)
└── reference/                   ✅ Phase 4 (6 canonical reference notes)
```

The existing design reports (`HANDOFF.md`, `optimizer_redesign.md`,
`phase0_baseline.md`–`phase4_report.md`, `chip_strategy_redesign.md`,
`generalization_report.md`, and others) stay where they are and serve as the
**evidence layer** that conceptual notes cite. Entry point: [[HANDOFF]].

## Conventions

- **Links:** Obsidian wikilinks — `[[note-name]]`.
- **Frontmatter:** every note has `type`, `status`, and `tags`.
- **Note kinds** (`type`): `moc`, `architecture`, `component`, `workflow`,
  `decision`, `reference`.
- **Grounding:** every statement is supported by the repository or an existing
  report. Assumptions and unverified claims are marked with a callout.
- **No per-file notes** — notes describe concepts and components, not individual
  scripts, classes, or endpoints.

> [!note] Source-of-truth caveat
> `AGENTS.md` in the repository root is **stale** (it references "Codex", a
> GW1–28 best of 1799 pts, and omits the optimizer redesign and `intel_08`).
> When accounts differ, trust the code, [`CLAUDE.md`](../CLAUDE.md), and
> [[HANDOFF]] over `AGENTS.md`.
