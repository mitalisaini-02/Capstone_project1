# Persistent MEC simulation with immediate gNodeB fallback

## Design

Experiment A allocates all active requests each second. Experiment B assigns each request once, when it first becomes active. The simulator retains the serving PUE and its reserved rates and bandwidth while the allocation meets the current link eligibility, radio, coverage, budget and resource checks. Every policy runs with independent lifecycle and resource state. The [audit](PERSISTENT_SIMULATION_AUDIT.md) identifies the source formulas, all four original policies and the interface differences with Phase 5.

An invalid PUE reservation is released in the same second. The request enters permanent gNodeB-only mode and attempts full-demand service immediately. An outage is recorded when the gNodeB check fails. An outage can clear when gNodeB service becomes feasible later in the request window. Calls to the four policies contain only new requests. Initial gNodeB-only service also stays on gNodeB for its lifetime. Initial failure leaves the request unserved for its window. Older reservations have priority, with request ID breaking ties. Fallback service claims free resources ahead of arrivals. Used bandwidth, including previously scheduled spare bandwidth, stays reserved while service remains valid.

The PHY and accounting use the same Mbps, MHz and GB units, capped PC5 radio rate, gNodeB Shannon rate, posted price and budget rules. The MEC fee of 5e-05 applies to every second with a PUE assignment. It ceases in a failed PUE's transition second. The LSTM is the existing inference pipeline with a 0.62 threshold. Both experiments use the same mobility and interference inputs. Random uses 16 trials with master seed 42 and the original per-scenario and per-second seed derivation. Exact solves its original cost-minimizing then rate-maximizing MILP on new requests using residual capacity and cost tolerance 1e-06.

## Workload and results

Experiment B ran 156 test scenarios and 110,287 active decisions per method. The primary cost column sums actual supplied-traffic payments, including valid service during epochs in which some other requests were unserved. Whole-epoch objective cost retains the original undefined value for incomplete service. Means divide by all active decisions. Lower total payment can reflect lower delivery.

| algorithm | served_seconds | unserved_seconds | full_service_pct | payment | mean_achievable_mbps | mean_delivered_mbps | pue_gb | gnb_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 82012 | 139518 | 37.0207 | 41.5276 | 14.4177 | 3.02166 | 0.436484 | 41.2197 |
| Random | 81834 | 139696 | 36.9404 | 41.0759 | 14.4504 | 2.98862 | 0.442098 | 40.7587 |
| Best-SINR | 82092 | 139438 | 37.0568 | 41.7952 | 14.4781 | 3.03875 | 0.568388 | 41.3235 |
| Exact | 80579 | 140951 | 36.3739 | 40.5205 | 14.2485 | 2.94849 | 0.444671 | 40.2029 |

| algorithm | new_requests | initial_pue | initial_gnb | initial_allocation_failures | pue_failures | immediate_fallbacks | gnb_fallbacks_ever_served | total_outage_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 2514 | 88 | 971 | 1455 | 88 | 32 | 56 | 139518 |
| Random | 2514 | 88 | 968 | 1458 | 88 | 38 | 57 | 139696 |
| Best-SINR | 2514 | 116 | 943 | 1455 | 116 | 58 | 83 | 139438 |
| Exact | 2514 | 89 | 954 | 1471 | 89 | 39 | 58 | 140951 |

A transition counts entry into gNodeB-only mode. An immediate fallback counts full gNodeB service in the same second as failure. The summary CSV records delayed gNodeB service, PUE duration, gNodeB-only duration, service rates, average cost, PUE and gNodeB payments, MEC fees and utilization. The experiment retains full service per request-second. An unserved second adds no traffic or charge.

### Exact solver status for initial-arrival batches

| stage1_status | stage2_status | initial_batches |
| --- | --- | --- |
| 0 | 0 | 1043 |
| 2 | — | 1436 |

HiGHS status 0 is optimal and 2 is infeasible. A blank stage-two status means the stage was skipped. The epoch output also retains stage costs, rates and solver gaps.

### Stateless reference over the same workload

| algorithm | served_seconds | unserved_seconds | full_service_pct | payment | mean_achievable_mbps | mean_delivered_mbps | pue_gb | gnb_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Greedy | 108004 | 113526 | 48.7537 | 52.5501 | 19.4623 | 3.92606 | 4.94154 | 49.1826 |
| Random | 108086 | 113444 | 48.7907 | 53.367 | 20.4416 | 3.93934 | 3.86843 | 50.4388 |
| Best-SINR | 108051 | 113479 | 48.7749 | 53.0506 | 20.2576 | 3.92738 | 6.75993 | 47.3824 |
| Exact | 45818 | 175712 | 20.6825 | 23.7002 | 8.9773 | 1.77934 | 2.34405 | 22.1857 |

Exact rejects an infeasible batch of new requests. Existing feasible persistent sessions continue independently. The heuristics can serve a feasible subset of a simultaneous arrival batch. The primary outcome includes every unserved request and uses the same workload denominator across algorithms.

### Common-feasible secondary comparison

24,571 scenario-seconds had complete service for all four algorithms in both experiments. The charges below use complete objective bills. This comparison describes only that subset.

| algorithm | decisions | cost_a | cost_b | mean_rate_a | mean_rate_b |
| --- | --- | --- | --- | --- | --- |
| Best-SINR | 24571 | 16.2714 | 16.5383 | 22.7198 | 21.3194 |
| Exact | 24571 | 16.1394 | 16.5271 | 23.0824 | 21.4154 |
| Greedy | 24571 | 16.1382 | 16.5269 | 22.0862 | 21.2194 |
| Random | 24571 | 16.206 | 16.5313 | 23.0473 | 21.4043 |

## Figures and validation

- [Cost, rate and service](persistent_results/figures/cost_rate_service.png)
- [Connection and resource metrics](persistent_results/figures/connections_resources.png)
- [Outages and utilization](persistent_results/figures/outages_utilization.png)
- [Timeline and PUE connection duration distribution](persistent_results/figures/timeline.png)
- [Full persistent summary](persistent_results/summary.csv) and [stateless reference](persistent_results/stateless_reference.csv)

The per-scenario tables record epoch metrics, request histories and fallback transitions. Reservations were checked against the original evaluator at every decision. The report reconciles arrivals, failures, outages, costs and delivered volumes across epoch, request and event outputs. Tests compare initial allocations and Exact objectives at equivalent inputs, compare fresh inference and radio efficiencies with the reference runner, and verify retention, shared capacity, blocked fallback and permanent gNodeB mode for all four algorithms.

## Limitations and reproduction

The original radio interference is fixed by each network snapshot. Session continuity can reserve spare bandwidth and reduce later admission. Testing a different reservation policy or allowing reallocation after failure would be a separate temporal experiment. Outage and service times reflect observed request windows, including those ending at the trace boundary.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q objective_comparison/test_objectives.py objective_comparison/test_persistent.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m objective_comparison.persistent_run
MPLCONFIGDIR=/tmp/mec-matplotlib .venv/bin/python -m objective_comparison.persistent_report
```
