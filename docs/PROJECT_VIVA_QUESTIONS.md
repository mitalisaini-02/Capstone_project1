# Viva Preparation — Questions and Answers

*Answers are written to be spoken. Numbers come from the repository outputs; sources are listed in
[COMPLETE_PROJECT_DOCUMENTATION.md](COMPLETE_PROJECT_DOCUMENTATION.md). "pp" = percentage points.*

---

## Part A — Spoken explanations

### A.1 Two-minute explanation

"My project is a marketplace where vehicles with spare mobile data — providers — sell it to nearby vehicles
that need Internet — consumers. An edge server, the MEC, runs the market. The difficulty is that vehicles move,
so a link that works now can fail soon.

I split the problem into stages. First, an LSTM looks at the last 10 seconds of how two vehicles move and
predicts whether they will stay within 1000 metres for the next 30 seconds; on unseen test runs it reaches a
ROC-AUC of 0.85. Second, a building-aware radio model using 3GPP equations and OpenStreetMap buildings checks
whether the link actually works and how many Mbps it carries. Third, every second, an optimizer decides who
serves whom: a two-stage MILP that first maximizes the number of admitted consumers and then minimizes what they
pay, compared with a greedy heuristic and a base-station-only baseline.

On held-out test scenarios the marketplace serves about 1.65 percentage points more request-seconds than the base
station alone and lowers the price per GB by about 3 %. Greedy is almost as good as the MILP on these small
markets and far faster. The next phase, which is partly built, adds session management so that when a provider
leaves, the consumer is moved to another provider or the base station."

### A.2 Five-minute explanation (outline to speak from)

1. **Problem.** Providers, consumers, MEC coordinator, gNodeB fallback. Mobility breaks links: distance,
   buildings, provider departure or withdrawal.
2. **Data.** 17 SUMO runs, real OSM city, 46–136 vehicles, 1000 s. Run-disjoint split 9/4/4. Marketplace
   attributes synthetic: 13 configurations × 4 runs × 3 replicates = 156 scenarios per split.
3. **LSTM.** 7 motion features × 10 s. Label: stays within 1000 m for all of next 30 s. Test ROC-AUC 0.8506
   vs 0.8354 (kinematic) and 0.6562 (distance). θ = 0.62 chosen on validation for 90 % precision; test
   precision 0.88, reported, not re-tuned.
4. **Channel.** 3,951 building polygons; straight line through a building → NLOS. TR 37.885 pathloss, shadowing,
   vehicle blockage; TR 36.942 rate mapping. One realized channel per link episode. Four scenarios G/F1/F2/F3.
   About 40 % of G link-seconds in outage.
5. **Allocation.** C1–C9: one PUE per consumer, link/access/backhaul capacity, full service, cell capacity,
   coverage, budget. Stage 1 max admitted, Stage 2 min cost. Zero audit violations. MILP vs gNodeB-only:
   +1.9 pp (val), +1.65 pp (test); price per GB −2.5 % / −2.9 %. Greedy equals MILP in most scenarios.
6. **Phase 5 and limits.** State manager built (departed / withdrawn / unavailable / out of range / censored;
   operational vs policy-eligible). Sessions and reallocation pending. Limits: synthetic economics, small
   markets, optimistic building data, no interference.

---

## Part B — Questions and answers

### Problem and motivation

**Q1. What problem are you solving, in one sentence?**
How an edge server can match moving vehicles that have spare mobile data with nearby vehicles that need it, so
that links are likely to last, actually work at the radio level, and are as cheap as possible for consumers.

**Q2. Why not just use the base station for everyone?**
The base station has limited capacity and coverage, and its price is fixed. In my results, adding providers
serves about 1.65–1.9 pp more request-seconds and lowers price per GB by 2.5–2.9 %. The gain is largest when the
base station is congested or has coverage gaps.

**Q3. Why is mobility the main challenge?**
Because the value of a link depends on the future. Two vehicles can be close now and 1000 m apart in 20 s, a
building can come between them, or the provider can leave the road. In my Phase 3 data a link episode lasts
about 46 s on average.

### LSTM

**Q4. What exactly does the LSTM predict?**
The probability that a provider–consumer pair stays within 1000 m at every second of the next 30 s, given
10 s of motion history. It is a survival-of-proximity prediction.

**Q5. Why an LSTM and not something simpler?**
The input is a short time series where trends matter — for example, accelerating apart. An LSTM handles that
naturally. I compared it with a constant-velocity extrapolation and a distance threshold. It is best on ROC-AUC
(0.851 vs 0.835 and 0.656), precision and calibration, but not on recall or F1, so the gain is real but modest.

**Q6. How did you avoid data leakage?**
The split is by simulation run, with zero overlap of runs, vehicles and pairs. The scaler was fitted on training
data only. Both thresholds were chosen on validation, and the test set was used once.

**Q7. How was θ = 0.62 chosen?**
On validation only: the smallest threshold between 0.50 and 0.90 where precision is at least 0.90 and recall at
least 0.50. That gives 0.62. Precision 0.90 means about 10 % of approved links break within 30 s.

**Q8. Your test precision is 0.88, below the 0.90 target. Why didn't you change θ?**
Because changing θ after seeing test results would make the test result optimistic and invalid. I report the
gap. I also keep θ = 0.65 as a sensitivity variant, chosen on validation using a bootstrap lower bound on
precision.

**Q9. What happens to samples where a vehicle leaves within 30 s?**
They get no label and are not used for training, because we cannot know whether the pair would have stayed in
range. The consequence is that the LSTM does not predict departures; that is a session-management problem for
Phase 5.

**Q10. What is the difference between predicting link survival and predicting bandwidth?**
Survival is a yes/no question about proximity in the future. Bandwidth is the number of Mbps the radio delivers,
which depends on buildings, blockage and signal strength. In my system bandwidth comes from the Phase 3 channel
model, not from the LSTM.

### Wireless channel

**Q11. Why is distance not enough for wireless feasibility?**
At 500 m a same-street link can carry 88 Mbps while a building-blocked link carries nothing. In my validation
data, the share of building-blocked links grows from 1 % below 100 m to 59 % at 750–1000 m.

**Q12. How do you decide whether a link is blocked by a building?**
I take the building outlines from the same OpenStreetMap file as the road network, in the same coordinates, and
test whether the straight 2-D line between the vehicles passes through a building's interior. If yes, the link is
NLOS. A grid index makes this fast, and tests confirm it matches a brute-force check.

**Q13. What are LOS, NLOSv and NLOS?**
LOS: clear line of sight. NLOSv: blocked only by other vehicles, which adds a random extra loss. NLOS: blocked by
buildings, which uses a much steeper path-loss formula. These are the three states of 3GPP TR 37.885.

**Q14. What are G, F1, F2 and F3?**
Four ways to decide the state. G uses building geometry (primary). F1 uses road identity: same road means
LOS/NLOSv, otherwise NLOS. F2 assumes every link is blocked (pessimistic bound). F3 assumes no building blocks
(optimistic bound). Because OSM buildings are incomplete, G is optimistic, and F2/F3 bracket the truth.

**Q15. What is the difference between expected and realized capacity, and which do you use?**
Expected capacity averages over all possible channel states and random losses. Realized capacity is the capacity
of the one channel that is sampled for that link episode. The optimizer uses only realized capacity, because
allocation must be based on the channel that occurs. A test proves expected values cannot change any feasibility
result.

**Q16. Why keep the channel state fixed over a link episode?**
Redrawing it every second would make a link flicker between working and failing randomly, which is not
physical. The state changes only when distance or building blockage actually changes. This is not full
spatially-correlated shadowing; that is a stated limitation.

**Q17. LSTM says "reliable" but the channel says "outage". What happens?**
The link is not eligible. Both stages must pass. The LSTM answers "will they stay close?", the channel answers
"does it work now?". Phase 3 records which stage failed first.

### Allocation

**Q18. What are your decision variables?**
For each eligible link, a binary x (link selected) and a rate r in Mbps. For each request, a binary y (admitted)
and a base-station rate g in Mbps.

**Q19. Walk me through the constraints.**
C1: at most one provider per request, only if admitted. C2: only eligible links exist. C3: rate on a link at
most its realized capacity. C4: a provider's total rate at most its radio access capacity. C5: at most its
backhaul. C6: an admitted request gets exactly its demand from provider plus base station. C7: base-station
cell capacity. C8: base station only for covered requests. C9: the request's cost at most its maximum price
times its demand.

**Q20. Why maximize admissions first and minimize cost second?**
Minimizing cost alone is solved by rejecting everyone. Admission first guarantees we serve as many consumers as
possible; among those solutions we pick the cheapest. Because every admitted request gets exactly its demand,
throughput is already fixed by Stage 1, so a throughput objective would add nothing.

**Q21. Why one provider per consumer?**
It keeps each consumer attached to one relay, which is simpler to manage and closer to how a U2N relay
association works. The base station tops up any remainder, so a consumer is not limited by one provider's
capacity.

**Q22. Why three separate provider capacities (C3, C4, C5)?**
They are different physical limits: one radio link, the provider's radio shared by all its consumers, and the
provider's own Internet connection. Any of them can be the bottleneck; in scenario G, the link limit binds in
about 52 % and the backhaul in about 44 % of cases.

**Q23. How does the greedy algorithm work?**
Requests are sorted by weight, then arrival time. Each takes its cheapest affordable option given what is left:
base station only, or one provider at its cost-minimizing rate plus the base station for the rest. Then the
used capacity is subtracted.

**Q24. Greedy vs MILP — which is better and by how much?**
The MILP is never worse by construction, and the audit confirms it. In practice they are identical in 140 of 156
validation scenarios and 127 of 156 test scenarios. The MILP admits about 0.01–0.02 pp more. Greedy takes about
0.08 ms per second, the MILP about 4 ms. The gap is small because each second has at most about 7 requests;
with dense markets the MILP should matter more, but I have not tested that.

**Q25. If the MILP barely beats greedy, why keep it?**
It gives the exact optimum, so it measures how good the heuristic is. Without it I could not say greedy is
near-optimal. It is also fast enough here to run every second.

**Q26. Why does total payment go up compared with base-station only, if the marketplace is "cheaper"?**
Because more requests are admitted, more data is bought. The fair comparison is price per GB, which falls by
2.5–2.9 %.

**Q27. Why are so many requests rejected even with the marketplace?**
Mostly budgets: in the synthetic data the consumer's maximum price is 0.6–1.5 times the base-station price, so
roughly 46 % of requests cannot afford the base station. About 70 % of base-station-only rejections are
budget-related. This is an assumption of the data, not a property of the algorithm.

**Q28. What is a "phantom x" and why does it matter?**
The solver sometimes sets a link's binary x to 1 while the rate on it is zero. If I recorded the provider from x,
I would report assignments that carry no traffic. I record the provider only from positive rates, and I count
these events: about 20,000 per split.

**Q29. How do you know the optimizer's output is correct?**
An independent audit re-checks every constraint from the output alone, and dominance checks confirm the MILP
never admits fewer than greedy or base-station only and never costs more at equal admission. Zero violations on
validation and test. Hand-built test cases with known optimal answers are also in the test suite.

### Dynamic sessions (Phase 5)

**Q30. What is implemented in Phase 5?**
The configuration, an input adapter, and a state manager, with 35 tests. The state manager can say for any
second whether a provider is present, departed, withdrawn, temporarily unavailable or out of range, and whether
a link is operational or policy-eligible. Sessions, reallocation, accounting and the experiments are not
implemented yet.

**Q31. Departure vs withdrawal vs temporary unavailability?**
Departure: the vehicle leaves the road network. Withdrawal: it keeps driving but permanently stops selling.
Unavailability: it is temporarily switched off and may come back. They need different reactions — a temporary
outage may be worth waiting for; departure and withdrawal are not.

**Q32. What is trace-end censoring and why does it matter?**
When the simulation ends, vehicles still driving simply stop being observed. Calling that a departure would
invent failures. So they are marked censored, in the LSTM labels, the Phase 3 episodes and the Phase 5 state.

**Q33. Operational vs policy-eligible links?**
Operational means the link physically works now — provider available, radio and capacity positive — so an
existing session can continue. Policy-eligible additionally needs the LSTM to say "reliable", and is used for
new assignments. A session is not cut just because the prediction dropped while the link still works.

**Q34. Reactive vs proactive switching — which is primary and why?**
Reactive (switch after the link stops working) is primary because it is simple, interpretable and has no
untuned parameters. Proactive switching (move when the prediction stays low) is planned only as a sensitivity
experiment; its parameters, such as how many low seconds trigger a switch and the cooldown, are still undecided.

**Q35. How will payment work when a session breaks?**
Planned: the consumer pays only for volume actually delivered, at the same posted prices and units as Phase 4
(1 GB = 8000 Mb). This is designed, not implemented.

### Assumptions, limitations and novelty

**Q36. What is genuinely new here?**
Not the LSTM or the MILP on their own. The contribution is the integrated, verified pipeline: it keeps proximity
survival, radio feasibility and economic feasibility as separate recorded stages, uses only realized
building-aware capacity in an exact admission-then-cost optimizer, and compares it fairly against a heuristic and
a no-marketplace baseline on held-out data with frozen code.

**Q37. What are the biggest limitations?**
Synthetic economics; small markets per second; incomplete building data and no interference; LSTM test precision
below target and no prediction of departures; Phase 5 not finished.

**Q38. Would your results generalize to a real city?**
The methods would transfer, but the numbers depend on assumptions: provider registration rate, prices, budgets,
one city map and up to 136 vehicles. I would expect larger gains with more providers or a congested base station,
as the configuration results suggest, but that needs testing.

**Q39. Is the 1000 m range realistic for a sidelink?**
It is the discovery range used to define the LSTM labels. Whether a link at that distance works is decided
separately by the channel model: a same-street line-of-sight link still carries about 72 Mbps at 1000 m at the
median SNR, while building-blocked links carry nothing beyond about 300 m. Changing the range would require retraining the LSTM.

**Q40. What would you do next?**
Finish Phase 5 (sessions, reallocation, accounting, validation experiments, then one test evaluation); test denser
markets; add elastic service rates; consider a survival model that also predicts departures.
