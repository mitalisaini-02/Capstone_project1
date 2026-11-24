"""
Phase 3 mobility and network simulation (no pricing, no allocation, no MEC sessions).

  channel_layer     per-run cache: building-aware channel states (G/F1/F2/F3), per-episode
                    realised channel, EXPECTED vs REALIZED radio capacity, link episodes
  reliability_layer per-run cache: LSTM survival probability for every candidate row
  feasibility       per-scenario: provider status, reliability, wireless and PUE resource
                    feasibility of every candidate link + gNodeB coverage per request-second
  validate, stats   checks and descriptive statistics
"""
