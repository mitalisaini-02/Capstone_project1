"""
Reliability models: predict P(link survives the next horizon) for PUE–CUE pairs.

Kept independent of wireless link models, pricing and optimisation. Submodules are
not imported here on purpose (avoids import cycles; import what you need explicitly):
  base.ReliabilityModel                 common interface
  lstm_distance.LSTMDistanceSurvivalModel  adapter around the existing LinkSurvivalPredictor
  kinematic.KinematicSurvivalModel      kinematic-extrapolation alternative
  policy.ThresholdPolicy                theta decision rule (0.62 primary / 0.65 sensitivity)
"""
