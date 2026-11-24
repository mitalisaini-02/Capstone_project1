# Professor's Guide to the Project

## MEC-Assisted Internet Sharing Marketplace with LSTM-Based Link Reliability Prediction and Resource Allocation

*A readable walk-through of what was built, why, and what the results show. Technical details and every
source file are in [COMPLETE_PROJECT_DOCUMENTATION.md](COMPLETE_PROJECT_DOCUMENTATION.md); diagrams are in
[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md).*

---

## 1. The idea in one paragraph

Some vehicles have spare mobile data; others need Internet. This project builds a **marketplace** in which a
provider vehicle (**PUE**) sells part of its connection, over a direct vehicle-to-vehicle radio link, to a nearby
consumer vehicle (**CUE**). A **MEC server** (an edge server near the base station) runs the market: it finds
nearby providers, checks whether the connection will last, checks whether the radio link actually works, and
decides who serves whom at the lowest price. The **gNodeB** (5G base station) is always there as a fallback
and can top up what a provider cannot supply.

The hard part is **mobility**: providers drive away, buildings block the signal, and providers stop selling or
leave the road.

## 2. Where the project stands

| Phase | What it does | Status |
|---|---|---|
| 1. LSTM reliability | Predicts if two vehicles stay within 1000 m for the next 30 s | ✅ Complete |
| 2. Marketplace data | Creates offers, requests, prices and base-station load on real traffic | ✅ Complete |
| 3. Wireless channel | Checks with building outlines and 3GPP equations whether the radio link works | ✅ Complete |
| 4. Allocation | Every second, decides who serves whom (Greedy vs exact MILP vs base station only) | ✅ Complete (validation and test) |
| 5. Dynamic sessions | Keeps consumers served when providers leave; moves them to another provider | 🟡 Started: the state-tracking layer is built and tested; sessions and reallocation are **not yet built** |

---

## 3. Phase 1 — Will the link last? (LSTM)

### Motivation
Assigning a provider that drives away seconds later wastes the assignment and breaks the consumer's service.
Distance alone is a weak clue: two cars 300 m apart and approaching are a better match than two cars 300 m
apart and separating.

### What was done
- Generated **17 traffic simulations** with SUMO (a standard traffic simulator) on a real OpenStreetMap city
  map, each 1000 s long, with 46 to 136 vehicles.
- For every provider–consumer pair within 1000 m, took the **last 10 seconds** of seven motion features
  (distance, speeds, relative speed, position offsets, closing speed).
- Labelled each sample **1** if the pair stays within 1000 m for **all of the next 30 s**, **0** otherwise.
  If a vehicle disappears within those 30 s, the sample has no clear answer and is left out.
- Trained a 2-layer LSTM (a neural network for sequences, 54,081 parameters).
- Split the data **by simulation run** (9 training, 4 validation, 4 test), so no vehicle appears in two splits.

### Results (test runs, never used for any choice)

| Model | ROC-AUC | Precision | Recall |
|---|---|---|---|
| **LSTM** | **0.851** | **0.845** | 0.738 |
| Constant-velocity extrapolation | 0.835 | 0.829 | 0.762 |
| Simple distance threshold | 0.656 | 0.724 | 0.622 |

The LSTM ranks links best and is better calibrated, but it does **not** beat the extrapolation baseline on
recall or F1; the project reports this openly.

### Turning the prediction into a decision
The MEC accepts a link only if the predicted survival probability is at least **θ**. θ was chosen on
**validation data only**: the smallest θ giving at least 90 % precision with at least 50 % recall →
**θ = 0.62**. On test data the precision at θ = 0.62 is **0.88**, slightly below the 90 % target; θ was not
changed after seeing this. A stricter variant θ = 0.65 is kept for sensitivity analysis.

### What the LSTM does *not* do
It predicts whether the cars **stay close**, not how fast the link is. Speed of the link comes from Phase 3.

---

## 4. Phase 2 — Building the marketplace data

### Motivation
No real dataset exists of vehicles trading Internet. To test an optimizer fairly, the project needs offers,
requests, prices and base-station capacity that are realistic in shape, documented, and reproducible.

### What was done
- **Mobility is real** (SUMO); **everything economic is synthetic** and labelled as such.
- Providers register with some probability, switch on and off, have a backhaul capacity, a radio (access)
  capacity, a link-speed cap, and a price between 0.3 and 1.5 times the base-station price.
- Consumers request service in four classes (web, SD video, HD video, bulk download) with a rate, a duration and
  a maximum price per GB.
- Six base stations with time-varying background load.
- **13 configurations**: a baseline plus changing one factor at a time (provider availability, demand, prices,
  base-station capacity, coverage gaps, provider withdrawals) and three stress combinations.
- 13 configurations × 4 runs × 3 random repetitions = **156 scenarios** for validation and another 156 for test.
- A few parameters (registration probability, base-station capacity) were **calibrated on validation runs**,
  then frozen before test data was generated.

### Quality checks
Every scenario passed 40 automatic checks (6,216 per split), and regenerating everything from the seeds
reproduced all files byte-for-byte.

---

## 5. Phase 3 — Does the radio link actually work?

### Motivation
Two links of the same length can behave completely differently. Using the 3GPP vehicle-to-vehicle channel
model (TR 37.885) with this project's settings:

| Distance | Same street, clear view | Blocked by a building |
|---|---|---|
| 100 m | 88 Mbps | 40 Mbps |
| 500 m | 88 Mbps | 0 Mbps |

### What was done
- Extracted **3,951 building outlines** from the same OpenStreetMap file that produced the road network, in
  the same coordinates.
- For every candidate link and second: if the straight line between the two vehicles passes **through a
  building**, the link is **NLOS** (blocked). Otherwise it is in clear view (LOS) or partially blocked by other
  vehicles (NLOSv), with 3GPP probabilities.
- Computed the signal-to-noise ratio and the data rate from 3GPP formulas, with one random channel draw per
  link episode (so the channel does not flicker randomly every second).
- Applied all checks in a fixed order and recorded the **first reason** a link fails: provider not available
  → not enough history → LSTM below θ → radio outage → zero capacity.
- Because OpenStreetMap buildings are incomplete, three **alternative scenarios** bound the uncertainty:
  F1 (road-based guess), F2 (assume everything blocked — pessimistic) and F3 (assume nothing blocked —
  optimistic). G (geometry) is the primary scenario.

### Results (primary scenario G)

| | Validation | Test |
|---|---|---|
| Link-seconds in radio outage | 38 % | 40 % |
| Link-seconds that pass every check | 4.4 % | 4.3 % |
| Main reason links fail | provider not available (88 %) | provider not available (85 %) |

The LSTM rejects about 6–9 % of candidate link-seconds and buildings about 2 %. The low feasible share mainly
reflects the assumed provider registration rate.

### A distinction worth stressing
- **LSTM reliability** looks **30 seconds ahead** at **movement**.
- **Wireless feasibility** looks at **this second** and **physics** (buildings, signal strength).

Both must say "yes".

---

## 6. Phase 4 — Who serves whom?

### Motivation
Even when several links are usable, providers have limited capacity, the base station has limited capacity,
and consumers have budgets. The assignment determines how many people are served and what they pay.

### Rules of the market (constraints)
1. A consumer uses **at most one provider** (plus the base station for any remainder).
2. Each link, each provider's radio and each provider's Internet connection have **separate** limits.
3. An admitted consumer receives its **full requested rate**.
4. Base-station cells have limited capacity and only cover nearby consumers.
5. The consumer's **average price per GB** may not exceed its maximum.

### Three methods, same inputs
- **Two-stage MILP (exact):** first serve the **largest possible number** of consumers; then, among those
  solutions, choose the **cheapest** for consumers. Doing it in this order prevents the trivial "reject everyone
  to pay nothing" answer.
- **Greedy (fast):** serve consumers in arrival order, each taking its cheapest affordable option.
- **Base station only:** the no-marketplace reference.

Every second is solved independently in this phase. Every solution is re-checked by an **independent
audit**: **zero violations** on validation and test.

### Results (156 scenarios, primary scenario G, θ = 0.62)

| | Validation | Test |
|---|---|---|
| Share of request-seconds served — base station only | 53.0 % | 46.5 % |
| Share served — MILP | 54.9 % | 48.1 % |
| Improvement (95 % CI) | +1.9 pp [1.5, 2.3] | +1.65 pp [1.3, 2.0] |
| Price per GB, MILP (base station = 1.00) | 0.975 | 0.971 |
| Share of data carried by providers | 8.8 % | 7.8 % |
| MILP better than greedy (scenarios) | 4 of 156 | 6 of 156 |
| Average solve time per second | MILP 4.3 ms, Greedy 0.08 ms | — |

Plain-language takeaways:
- The marketplace **helps**, consistently, and never serves fewer consumers than the base station alone.
- The gain is **larger** when providers are plentiful or cheap, or when the base station is congested or has
  coverage gaps (for example, with an overloaded base station validation admission rises from 39.4 % to 42.9 %).
- **Greedy is nearly optimal** here because each second has only a handful of requests; the exact MILP is
  slower but still only milliseconds.
- The **realistic building model matters**: providers carry 4–5 % of data if every link is assumed blocked,
  8–9 % with real buildings, and 11–13 % if no link is assumed blocked.
- Many consumers cannot be served at all because their maximum price is below the base-station price
  (an assumption of the synthetic data): about 70 % of rejections are budget-related.

---

## 7. Phase 5 — Keeping sessions alive (in progress)

### Motivation
Phase 4 decides each second from scratch. Real users need a **continuous session**: if their provider leaves,
the MEC should notice and move them to another provider or the base station, and they should pay only for data
actually delivered.

### What is built and tested (35 tests)
- A strict configuration that refuses undecided parameters and verifies, by file hashes, that it uses exactly
  the frozen Phase 4 code.
- A **state manager** that tells, for any second, *why* a provider can or cannot serve a consumer:

| Situation | Label |
|---|---|
| Provider left the road network | **Departed** |
| Provider permanently stopped selling | **Withdrawn** |
| Provider temporarily switched off | **Unavailable** |
| Provider present but more than 1000 m away | **Out of range** |
| Simulation ended while the vehicle was still driving | **Censored** (not counted as a departure) |

- Two sets of links: **operational** (the link physically works, so an existing session may continue) and
  **policy-eligible** (stricter: also passes the LSTM threshold, so a new assignment may be made). A session is
  not cut just because the prediction worsened.

### What is not built yet
Session tracking, failure handling, reallocation, payment for delivered volume, the MEC control loop and the
Phase 5 experiments. Some parameters (how long an interrupted session may wait, settings for switching
*before* failure) are still undecided. Reacting **after** a failure is planned as the main approach;
switching **before** a failure is planned only as a comparison.

---

## 8. How trustworthy are the results?

- **No test-data tuning**: model, thresholds and calibration chosen on validation; test used once.
- **No overlap**: training, validation and test use different simulation runs and vehicles.
- **Frozen code**: the allocation code was hash-frozen before the test run; the test run refuses to start if
  anything changed.
- **Reproducible**: synthetic data and simulation outputs regenerate byte-for-byte; allocation results reproduce
  exactly (only timing measurements differ).
- **Protected artifacts**: 53 key files (model, metrics, datasets) are hash-checked; all 53 verified unchanged
  on 2026-11-25.
- **186 automated tests** in the repository.

## 9. Honest limitations

- Prices, demands and provider behaviour are **assumptions**, not measurements.
- Markets per second are **small** (up to about 7 requests), so the exact optimizer has little room to beat
  greedy, and large-scale behaviour is untested.
- OpenStreetMap buildings are **incomplete**, so the primary radio scenario is optimistic; there is no
  interference model.
- The LSTM's test precision (0.88) is below the 0.90 design target, and it cannot predict a provider simply
  leaving the road.
- The dynamic-session phase is **not finished**.

## 10. What the project contributes

Not a new neural network or a new solver, but a **carefully separated and verified pipeline** that keeps three
different questions apart — *Will they stay close? Does the radio work? Is it affordable and within capacity?* —
measures how much each one limits the market, feeds only the realized radio capacity to an exact
admission-then-cost optimizer, and compares it fairly with a fast heuristic and a no-marketplace baseline on
held-out data.
