# Phase 5 — MEC session management and dynamic reallocation

Implements Design Revision 2 (primary configuration). Phase 1–4 code, data, models and frozen
configuration are reused unchanged; the frozen Phase 4 allocator is the only optimiser.

## Modules (`src/mec/`)
| Module | Role |
|---|---|
| `config.py` | Strict loader for `config/mec/phase5.json`; verifies the pinned Phase 4 hashes; refuses to use decisions still marked `pending` |
| `inputs.py` | `Phase5ScenarioInputs` = Phase 4 `ScenarioInputs` + in-memory flags: `feasible_unaware` (ignores only the LSTM stages), `operational` (links an existing session may keep) |
| `state.py` | `StateManager`: SUMO presence, provider status, request windows, per-second `NetworkSnapshot` (E_t policy-eligible, O_t operational, Phase 4 problem) |
| `events.py` | `EventMonitor`: terminal / hard-failure / capacity / advisory events from observable-at-t data |
| `failure.py` | `FailureDetector`: one action per session (terminate > release > rebalance > keep) + transitive rebalance groups |
| `session.py` | Session state machine and `SessionManager` (reallocation, rebalance, recovery, interruption, drop counters) |
| `accounting.py` | `Ledger`: per-second service/payment rows via the frozen `PostedPricePolicy`; conservation checks |
| `allocation_service.py` | Builds Phase 4 `EpochProblem`s on request subsets, filtered links, residual capacities; calls the frozen allocator; audits every solution |
| `reallocation.py` | `joint_rebalance` (current provider's operational link only) and `reallocate` (E_t links) |
| `coordinator.py` | `MecCoordinator`: the per-second control loop + full per-second audit |

Allowed imports: `src.marketplace`, `src.common`, `src.mobility.trace`, `src.utils` (tested). The LSTM is
used only through recorded Phase 3 columns; Phase 3 episode ground truth is never read online (tested).

## Control loop (per second t)
snapshot → detect events → decide → terminate → release hard failures → joint rebalance of capacity-affected
groups (one Phase 4 call) → keep the rest (gNodeB share follows handovers) → allocate released + interrupted +
new arrivals over E_t with residual capacity (one Phase 4 call) → ledger → interruption tolerance → audit
(Phase 4 C1–C9 + phantom on ALL assignments, plus session invariants).

Semantics: `time_window` service; `block` arrivals (a request not admitted at its arrival second is blocked);
continuing sessions keep working links even if p̂ < θ (O_t ⊇ E_t); new and re-allocated sessions use E_t only;
capacity is always the realized C3; terminal events fire at the first second after the window; sessions still
open after the trace are CENSORED (or COMPLETED if the window ended exactly at trace end).

## Running
```
.venv/bin/python scripts/run_phase5_sessions.py demo
.venv/bin/python scripts/run_phase5_sessions.py run --scenario C09_withdrawals_some__run_006__r0 --max-interruption-s 10
.venv/bin/python -m pytest tests/test_mec_state.py tests/test_mec_m2.py tests/test_mec_e2e.py -q
```
`run` writes once to `data/mec/phase5/runs/<label>/<scenario>/` and refuses the test split (Phase 5 not frozen).

## Not implemented / pending decisions
* Proactive reliability switching (sensitivity experiment) — needs k, cooldown, cap, ping-pong definition.
* Signalling delay 1 s sensitivity.
* Selection of `max_interruption_s` on validation (candidates 0/5/10/30/60; rule pending) — runs are provisional.
* Online-vs-ground-truth link-loss classification check, full validation grid, Phase 5 freeze, test split.
