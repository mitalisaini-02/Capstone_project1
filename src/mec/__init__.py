"""
Phase 5 MEC session management and dynamic reallocation (Design Revision 2).

Implemented so far (M1):
  config   Phase5Config: strict loader, pending-decision guard, Phase 4 dependency pins
  inputs   Phase5ScenarioInputs: Phase 4 ScenarioInputs + policy-eligible / operational link flags
  state    StateManager: SUMO presence, provider status, request windows, per-second snapshots

Import rule (enforced by tests/test_mec_state.py): this package may import only
src.marketplace (frozen Phase 4 interfaces), src.common, src.mobility.trace and src.utils.
It must NOT import reliability, models, network, features, simulation, datagen or training code:
the LSTM reaches Phase 5 only as recorded Phase 3 columns, and nothing is recomputed.
Phase 3 episode ground truth (end causes) is evaluation-only and is never read by online modules.
"""
