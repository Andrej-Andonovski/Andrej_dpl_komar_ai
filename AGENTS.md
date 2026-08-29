# FPL AI — Codex Project Memory

> **Canonical source:** [`CLAUDE.md`](CLAUDE.md) is the maintained operational
> memory for this project — read it first, every session. This file is a thin
> Codex-facing pointer + snapshot so it doesn't rot out of sync again (it
> previously drifted ~3 months behind: missing the optimizer redesign,
> `intel_08`, corrected FPL rules, and the MILP work entirely). Update the
> **date line and snapshot** below whenever you make a significant change;
> do not re-duplicate `CLAUDE.md`'s full content here.

Last synced: 2026-07-27 (see `CLAUDE.md` + `docs/HANDOFF.md` for full detail).

## What this project is
Fantasy Premier League Predictive Management System (thesis). 3-layer hybrid
AI: LightGBM models (per position) → ILP/MILP optimizer → Claude LLM agent
(narrative) + Gemini (recommendations). Full architecture: `docs/index.md`
(Obsidian vault) — canonical for concepts/architecture/decisions.

## Snapshot (2026-07-27)
- Original pipeline (Stages 1-9, Intel 01-08): complete. Thesis headline
  result: **2468 pts** full GW1-38 season run (env-bound to the original
  machine's library stack — see `docs/reference/environment-and-docker.md`).
- Parallel optimizer redesign (legacy PuLP ILP → multi-period MILP,
  `pipeline/milp_core.py`, `OPTIMIZER=mp`): Phases 0-4 complete, chip
  scarcity fix shipped, hit/churn discipline shipped and validated
  cross-season (2023-24, 2024-25, 2025-26 calendars). Fair Docker baseline
  legacy = 2252; mp travels better cross-season (proven generalization —
  legacy's edge is mostly memorized calendar).
- Phase 6 cross-season Optuna sweep (`pipeline/optuna_mp_search.py`): in
  progress — see `data/intel/optuna_mp/summary.json` for the live
  leaderboard and `docs/HANDOFF.md` §4-5 for protocol + remaining backlog.
- `pipeline/minutes_model.py` (learned play-probability, replacing a
  heuristic): built, wired into `season_simulator.py` under `OPTIMIZER=mp`,
  unit-tested — not yet backtested for score impact.
- Full test suite runs **natively on Windows now** (no Docker required for
  dev): `python tests/test_*.py`. Docker (`fpl-sim` image) remains the
  reference environment for any score that must match a documented number.

## Critical rules — never break
1. GW1 blind test — zero 2025-26 data in training, ever.
2. No leakage — features must be knowable before kickoff.
3. No cross-season bleed — rolling windows partition by season.
4. Four separate position models (GK/DEF/MID/FWD) — never mix.
5. Walk-forward validation only — never shuffle.
6. Confirmation gate after every step — never auto-advance.
7. Online retraining — full retrain per GW with actuals appended.
8. FPL free transfer cap — max 5 banked (2025-26 rules).
9. Penalty subtraction — transfer hits are SUBTRACTED, not added.

## Where to look next
- Full current status + next steps: `CLAUDE.md`.
- Optimizer redesign detail + open backlog: `docs/HANDOFF.md`,
  `docs/fixing_backlog.md`.
- Architecture/concepts: `docs/index.md` (Obsidian vault entry point).
