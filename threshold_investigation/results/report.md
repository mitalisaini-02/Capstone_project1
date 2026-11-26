# Validation-only LSTM reliability threshold investigation

## Recommendation

Retain **0.62 as the primary threshold** under the existing point-estimate policy. It is the smallest cutoff on the predeclared 0.50–0.90 grid (step 0.01) with at least 90% precision and 50% recall. It yielded **90.010% precision**, **66.720% recall** and approved **11,952** labeled link windows. Keep **0.65** as the stricter sensitivity setting. Its estimated one-sided 95% pair-cluster precision lower bound was **90.11%**, compared with **88.63%** at 0.62. The corresponding recall lower bound at 0.65 was **60.01%**, above 50%. At 0.62 the worst validation run had **89.27%** precision. The point criterion and the lower-bound criterion answer different questions. The recommendation assumes the original point criterion. The immediately lower grid point, 0.61, had 89.537% precision. Selecting the smallest qualifying threshold retains the largest candidate set among the qualifying grid points.

## Repository audit

- Phase 1's [selection script](../../scripts/sweep_reliability_threshold.py) tried 41 thresholds and applied both floors to unrounded validation rates. Its predeclared fallback was maximum balanced accuracy, which was unnecessary. The historical [selection record](../../outputs/large_dataset_experiment/reliability_threshold/selected_threshold.json) lists qualifying thresholds 0.62–0.74. Marketplace outcomes were outside the selection rule. On this grid, maximum F1 selects 0.50, and maximum balanced accuracy selects 0.53. These objectives trade false approvals and missed survivals differently and do not enforce the precision floor.
- [Checkpoint metadata](../../outputs/large_dataset_experiment/checkpoints/checkpoint_metadata.json) records 0.62 for eligibility and the earlier 0.53 classifier operating point separately. The checkpoint's stored default is 0.50. [Phase 3 configuration](../../config/simulation/phase3.json) resolves primary and sensitivity metadata. [Recorded validation settings](../../data/simulation/phase3/val/experiment_config.json) confirm 0.62 and 0.65. [Phase 5](../../config/mec/phase5.json) uses policy tags for these cutoffs, and the [four-algorithm experiment](../../objective_comparison/config.json) explicitly pins 0.62.
- [Grouped confidence analysis](../../scripts/threshold_grouped_ci.py) chose 0.65 as the smallest threshold meeting its lower-bound rule. It used 2,000 bootstrap resamples (seed 42), with 594 validation pair clusters, 108 consumer clusters and four mobility runs. Shared mobility conditions make pair-cluster uncertainty estimates potentially optimistic.

## Method

The fixed LSTM scores a 10-second mobility history for 30-second distance survival within 1,000 m. The rule approves scores greater than or equal to θ. Changing θ changes which links pass the filter and leaves the network weights and scores intact. Classification metrics were reconstructed from the original validation confusion counts for 25,602 labeled windows across runs `run_000`, `run_001`, `run_006`, and `run_013`. Existing validation confidence bounds were reused.

The marketplace sweep replayed the same 156 validation scenarios for all nine thresholds, using the existing Phase 5 MEC coordinator and the frozen Phase 4 **greedy** allocator. Channel G, proactive switching, the 60-second interruption timeout, the arrival policy and scenario replicates were fixed. Existing marketplace base seeds were 20261124 and the existing channel seed was 20261125. Only the reliability gate was recomputed in memory from the recorded probabilities. The physical availability and radio/resource gates were unchanged. A working link can continue below θ, but a sustained low score can trigger a proactive switch. Candidate link counts are link-seconds. Eligible provider counts sum the distinct providers in each scenario. Active request-seconds come from the session ledger. Cost per GB pools Phase 5 payments and delivered traffic. This protocol measures the frozen marketplace method's threshold sensitivity.

The sweep verified the reproduced 0.62 and 0.65 candidate flags against the recorded Phase 3 flags. For both thresholds, all eight checked outcome fields matched the corresponding existing Phase 5 validation runs. Solver/ledger audits and independent event checks passed. Guard checks used the protected artifact manifest, Phase 4 dependency hashes and the Phase 2/3 validation input manifests. The analysis used validation scenario files and recorded validation threshold tables. The protected artifact guard also checks file hashes without parsing held-out test metrics. Results were written to a separate directory. The scenario table uses the common outcome and audit fields across scenarios. Optional coordinator diagnostics have varying keys, so their counts are excluded from the common table. The standard proactive switch and reallocation counts are included.

## Validation classification

| threshold | precision_raw | recall_raw | specificity | f1_score | balanced_accuracy_raw | false_positive_rate | false_negative_rate | predicted_positive_count | predicted_ineligible_count | precision_lb95_onesided_pair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.850741 | 0.797135 | 0.762081 | 0.823066 | 0.779608 | 0.237919 | 0.202865 | 15108 | 10494 | 0.8347 |
| 0.55 | 0.869253 | 0.750434 | 0.807976 | 0.805485 | 0.779205 | 0.192024 | 0.249566 | 13920 | 11682 | 0.854 |
| 0.6 | 0.890118 | 0.69381 | 0.854294 | 0.779799 | 0.774052 | 0.145706 | 0.30619 | 12568 | 13034 | 0.8757 |
| 0.62 | 0.9001 | 0.667204 | 0.874024 | 0.766348 | 0.770614 | 0.125976 | 0.332796 | 11952 | 13650 | 0.8863 |
| 0.63 | 0.904569 | 0.651947 | 0.882992 | 0.757758 | 0.76747 | 0.117008 | 0.348053 | 11621 | 13981 | 0.891 |
| 0.65 | 0.913631 | 0.625217 | 0.899451 | 0.742396 | 0.762334 | 0.100549 | 0.374783 | 11034 | 14568 | 0.9011 |
| 0.67 | 0.921138 | 0.598363 | 0.912851 | 0.725468 | 0.755607 | 0.0871492 | 0.401637 | 10474 | 15128 | 0.9091 |
| 0.7 | 0.93051 | 0.557244 | 0.929204 | 0.697052 | 0.743224 | 0.0707955 | 0.442756 | 9656 | 15946 | 0.9188 |
| 0.75 | 0.941806 | 0.496837 | 0.947774 | 0.650508 | 0.722305 | 0.0522262 | 0.503163 | 8506 | 17096 | 0.9305 |

False-positive rate is the fraction of truly non-surviving windows that pass. The fraction of approved windows that later fail equals one minus precision. These metrics have different denominators.

[Classification plots](figures/classification.png) · [All 41 thresholds and confidence bounds](classification.csv)

## Validation marketplace

| threshold | eligible_links | eligible_providers | eligible_request_seconds | admitted | admission_pct | active_seconds | pue_traffic_pct | gnb_traffic_pct | cost_per_gb | delivered_demand_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 20743 | 610 | 19949 | 1347 | 51.1195 | 98359 | 4.61467 | 95.3853 | 0.987903 | 46.2323 |
| 0.55 | 19268 | 596 | 18549 | 1346 | 51.0816 | 98291 | 4.57486 | 95.4251 | 0.988052 | 46.2182 |
| 0.6 | 17839 | 564 | 17255 | 1346 | 51.0816 | 98275 | 4.48815 | 95.5118 | 0.988399 | 46.2072 |
| 0.62 | 17106 | 564 | 16575 | 1339 | 50.8159 | 97637 | 4.05781 | 95.9422 | 0.989948 | 46.0009 |
| 0.63 | 16691 | 564 | 16187 | 1338 | 50.778 | 97606 | 4.02758 | 95.9724 | 0.990017 | 45.9865 |
| 0.65 | 16010 | 564 | 15539 | 1336 | 50.7021 | 97516 | 3.86842 | 96.1316 | 0.990743 | 45.866 |
| 0.67 | 15200 | 561 | 14761 | 1336 | 50.7021 | 97485 | 3.71168 | 96.2883 | 0.990875 | 45.844 |
| 0.7 | 14204 | 559 | 13793 | 1334 | 50.6262 | 97424 | 3.59267 | 96.4073 | 0.991282 | 45.8061 |
| 0.75 | 13023 | 558 | 12663 | 1334 | 50.6262 | 97413 | 3.58708 | 96.4129 | 0.99128 | 45.7975 |

| threshold | interruption_episodes | interrupted_seconds | dropped | recoveries | reallocations | proactive_switches | interruptions_per_100_admitted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 574 | 6377 | 15 | 462 | 78 | 7 | 42.6132 |
| 0.55 | 572 | 6396 | 16 | 461 | 76 | 5 | 42.4963 |
| 0.6 | 572 | 6412 | 16 | 460 | 72 | 6 | 42.4963 |
| 0.62 | 566 | 6211 | 16 | 461 | 70 | 6 | 42.2704 |
| 0.63 | 565 | 6210 | 16 | 460 | 71 | 6 | 42.2272 |
| 0.65 | 560 | 6214 | 16 | 457 | 70 | 6 | 41.9162 |
| 0.67 | 558 | 6245 | 16 | 455 | 62 | 6 | 41.7665 |
| 0.7 | 559 | 6266 | 16 | 456 | 58 | 6 | 41.904 |
| 0.75 | 560 | 6266 | 17 | 457 | 54 | 6 | 41.979 |

[Marketplace plots](figures/marketplace.png) · [Scenario-level results](marketplace_runs.csv) · [Pooled marketplace metrics](marketplace_summary.csv) · [Metrics by mobility run](marketplace_by_mobility_run.csv)

## Sensitivity relative to 0.62

| θ | Δ precision (pp) | Δ recall (pp) | Δ eligible links (%) | Δ sessions admitted | Δ PUE traffic (pp) | Δ cost/GB | Δ interruption episodes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | -4.93591 | 12.9931 | 21.2615 | 8 | 0.55686 | -0.00204465 | 8 |
| 0.55 | -3.08475 | 8.323 | 12.6388 | 7 | 0.517045 | -0.00189629 | 6 |
| 0.6 | -0.998264 | 2.66063 | 4.28505 | 7 | 0.430341 | -0.00154899 | 6 |
| 0.62 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.63 | 0.446891 | -1.52568 | -2.42605 | -1 | -0.0302365 | 6.90983e-05 | -1 |
| 0.65 | 1.35302 | -4.19871 | -6.40711 | -3 | -0.189389 | 0.000795302 | -6 |
| 0.67 | 2.10377 | -6.88415 | -11.1423 | -3 | -0.346129 | 0.000927317 | -8 |
| 0.7 | 3.04091 | -10.996 | -16.9648 | -5 | -0.465138 | 0.00133367 | -7 |
| 0.75 | 4.17054 | -17.0367 | -23.8688 | -5 | -0.470736 | 0.00133252 | -6 |

At 0.60, precision falls to 89.01% and recall increases to 69.38%. At 0.67, precision increases to 92.11% and recall falls to 59.84%. Moving from 0.62 to 0.67 excludes 368 additional false approvals and 1,110 additional true survivals. Moving from 0.62 down to 0.60 adds 187 false approvals and 429 true survivals. Relative to 0.62, 0.60 exposed +733 eligible link-seconds, 0.65 exposed -1,096, and 0.67 exposed -1,906. Across these four settings the admission-rate spread was 0.380 percentage points, the PUE traffic share spread was 0.776 percentage points, the cost spread was 0.002476 price units/GB, and the range in interruption counts was 14 episodes. These are descriptive sensitivity effects for this scenario set. No operational materiality margin was specified. Marketplace interruptions also depend on alternate suppliers, gNodeB resources, radio outages, mobility departures and the coordinator's switching policy. Compare normalized interruption counts alongside raw counts because the set of admitted requests may change.

### Existing allocator cross-check at 0.62 and 0.65

| allocator | threshold | admission_pct | pue_traffic_pct | cost_per_gb | interruption_episodes | dropped | recoveries | reallocations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| greedy | 0.62 | 50.8159 | 4.05781 | 0.989948 | 566 | 16 | 461 | 70 |
| greedy | 0.65 | 50.7021 | 3.86842 | 0.990743 | 560 | 16 | 457 | 70 |
| milp | 0.62 | 50.8159 | 4.04609 | 0.989968 | 559 | 16 | 454 | 66 |
| milp | 0.65 | 50.7021 | 3.85836 | 0.99076 | 552 | 16 | 449 | 67 |

These rows come from the recorded Phase 5 validation grid. The complete nine-threshold replay covers the configured allocator shown above. [Reference summary](reference_summary.csv)

## Limitations

The 90% requirement represents an operational target for approved, labeled validation link windows. Windows without a defined 30-second label were outside the classification sample. The session replay separately counts mobility departures and interruptions. It does not automatically guarantee reliable sessions or generalization to new mobility environments. Selection used the same four validation mobility runs as the threshold assessment. Temporally adjacent windows and pairs can share vehicles, and marketplace replicates reuse each of the four mobility traces. The small number of independent traces limits uncertainty claims. This study holds the radio setting and interruption policy fixed. A broader sweep across the other channel settings, additional independent mobility runs and an all-threshold MILP comparison could strengthen generalization. The test split remained untouched in this investigation.

## Viva answer

“We tried 41 cutoffs on validation data. We wanted at least 90% of approved links to survive while retaining at least half of the survivable links. At 0.60 we missed the precision target. At 0.67 we gained precision but discarded more usable links. 0.62 was the lowest tested cutoff that met both targets, and we tested 0.65 separately to check a stricter confidence-based setting.”

## Reproduction

From the repository root with the existing environment and validation inputs:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m threshold_investigation.run   threshold_investigation/reproduction --workers 8
MPLCONFIGDIR=/tmp/mec-matplotlib .venv/bin/python -m threshold_investigation.report   threshold_investigation/reproduction
```

[Protocol and input hashes](protocol.json) · [Validation checks](validation_checks.json)
