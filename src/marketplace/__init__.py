"""
Phase 4 marketplace allocation (per-second snapshot; no sessions or reallocation).

Consumes Phase 2/3 FILES only: this package must not import reliability, models, network,
features or simulation code, so reliability and wireless feasibility are never recomputed.
  problem      EpochProblem: plain data for one second
  eligibility  builds EpochProblems from Phase 3 eligible links + Phase 2 prices/budgets
  pricing      posted-price payment and revenue split
  admission    stand-alone options, best option under remaining resources, rejection reasons
  greedy       GreedyAllocator          milp  MilpAllocator (two-stage lexicographic, HiGHS)
  allocation   Allocation + independent constraint audit (C1-C9 + phantom-PUE checks)
  evaluation   metrics and paired comparisons
"""
