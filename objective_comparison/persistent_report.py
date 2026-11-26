"""Compare persistent Experiment B with the frozen stateless Experiment A."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from objective_comparison.report import markdown_table
from objective_comparison.run import read_json

ROOT = Path("objective_comparison")


def read_runs(path, ids, kind):
    return pd.concat([pd.read_csv(path / f"{sid}.{kind}.csv.gz").assign(scenario_id=sid)
                      for sid in ids], ignore_index=True)


def summary(epochs, sessions, methods):
    rows = []
    for method in methods:
        e = epochs[epochs.algorithm == method]
        s = sessions[sessions.algorithm == method] if sessions is not None else None
        row = dict(algorithm=method, decisions=len(e), requests_seconds=e.requests.sum(),
                   served_seconds=e.served.sum(), unserved_seconds=e.unserved.sum(),
                   partial_seconds=e.partial.sum(), full_service_pct=100 * e.served.sum() / e.requests.sum(),
                   unserved_pct=100 * e.unserved.sum() / e.requests.sum(),
                   partial_pct=100 * e.partial.sum() / e.requests.sum(),
                   payment=e.supplied_traffic_payment.sum(), pue_cost=e.pue_cost.sum(),
                   gnb_cost=e.gnb_cost.sum(), mec_fee=e.mec_fee.sum(),
                   mean_achievable_mbps=e.rate.mean(), mean_delivered_mbps=e.delivered_mbps.mean(),
                   pue_gb=e.pue_gb.sum(), gnb_gb=e.gnb_gb.sum(),
                   decisions_with_all_requests_served=int(e.cost.notna().sum()),
                   runtime_s=e.runtime_s.sum())
        row["pue_traffic_pct"] = 100 * row["pue_gb"] / (row["pue_gb"] + row["gnb_gb"])
        row["mean_cost_per_served_second"] = row["payment"] / row["served_seconds"]
        if s is not None:
            failures = s.failure_s.notna()
            pue = s.pue_seconds.gt(0)
            gnb = s.gnb_seconds.gt(0)
            row.update(new_requests=len(s), successful_initial_allocations=int(e.admitted.sum()),
                       initial_allocation_failures=int((e.arrivals - e.admitted).sum()),
                       initial_pue=int(e.initial_pue.sum()), initial_gnb=int(e.initial_gnb.sum()),
                       initial_success_pct=100 * e.admitted.sum() / len(s),
                       requests_ever_served=int(s.delivered_gb.gt(0).sum()),
                       requests_served_entire_window=int((s.pue_seconds + s.gnb_seconds).eq(s.requested_seconds).sum()),
                       requests_never_served=int(s.delivered_gb.eq(0).sum()),
                       cost_per_new_request=row["payment"] / len(s),
                       cost_per_ever_served_request=row["payment"] / s.delivered_gb.gt(0).sum(),
                       total_delivered_gb=s.delivered_gb.sum(), pue_failures=int(failures.sum()),
                       pue_to_gnb=int(failures.sum()), immediate_fallbacks=int(e.immediate_fallbacks.sum()),
                       gnb_fallbacks_ever_served=int((failures & gnb).sum()),
                       pue_connection_mean_s=s.loc[pue, "pue_seconds"].mean(),
                       gnb_fallback_mean_served_s=s.loc[failures, "gnb_seconds"].mean(),
                       gnb_fallback_mean_mode_s=(s.loc[failures, "last_s"] - s.loc[failures, "failure_s"] + 1).mean(),
                       total_pue_seconds=s.pue_seconds.sum(), total_gnb_only_seconds=s.gnb_seconds.sum(),
                       total_outage_seconds=s.outage_seconds.sum(), observed_requests_ended=len(s))
            for col in (key for key in e if key.endswith("_utilization")):
                row[f"mean_{col}"] = e[col].mean()
                row[f"peak_{col}"] = e[col].max()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bars(frame, output, columns, title):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, (key, label) in zip(axes.flat, columns):
        values = frame[key].fillna(0)
        bars = ax.bar(frame.algorithm, values, color=["#245b76", "#529b96", "#d08244", "#97557d"])
        ax.bar_label(bars, labels=[f"{v:.3g}" if pd.notna(v) else "n/a" for v in frame[key]], padding=3)
        ax.set(ylabel=label, ylim=(0, max(values.max() * 1.2, 1e-9)))
        ax.tick_params(axis="x", labelrotation=12)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(title)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_timeline(epochs, sessions, figs):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for method, frame in epochs.groupby("algorithm", sort=False):
        g = frame.groupby("time_s").agg(cost=("supplied_traffic_payment", "sum"), served=("served", "sum"),
                                         delivered=("delivered_mbps", "sum"), requested=("requests", "sum"))
        axs[0,0].plot(g.index, g.cost.cumsum(), label=method)
        axs[0,1].plot(g.index, 100*g.served/g.requested, label=method)
        axs[1,0].plot(g.index, g.delivered, label=method)
        durations = sessions.loc[(sessions.algorithm == method) & sessions.pue_seconds.gt(0), "pue_seconds"]
        if len(durations):
            x = np.sort(durations.to_numpy())
            axs[1,1].step(x, np.arange(1, len(x)+1) / len(x), where="post", label=method)
    for ax, x, y in zip(axs.flat, ["Trace second"] * 3 + ["PUE connection duration (s)"],
                        ["Cumulative payment (price units)", "Full-service share (%)",
                         "Delivered rate summed over scenarios (Mbps)", "Cumulative fraction"]):
        ax.set(xlabel=x, ylabel=y)
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.savefig(figs / "timeline.png", dpi=150)
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path, nargs="?", default=ROOT / "persistent_results")
    args = cli.parse_args()
    output = args.output
    meta = read_json(output / "metadata.json")
    ids, cfg = meta["scenario_ids"], meta["config"]
    methods = cfg["algorithms"]
    data = read_runs(output, ids, "epochs")
    sessions = read_runs(output, ids, "sessions")
    events = read_runs(output, ids, "events")
    reference = ROOT / "results"
    baseline_meta = read_json(reference / "metadata.json")
    ignored = {"experiment", "output"}
    if ({k: v for k, v in baseline_meta["config"].items() if k not in ignored}
            != {k: v for k, v in cfg.items() if k not in ignored}
            or any(meta[key] != baseline_meta[key] for key in ("manifest_sha256", "checkpoint_hashes"))):
        raise ValueError("Experiments use different network, allocation or input settings")
    a = read_runs(reference, ids, "epochs")
    keys = ["scenario_id", "time_s", "algorithm"]
    if meta["epoch_limit"] is not None:
        a = a.merge(data[keys], on=keys, validate="one_to_one")
    joined = data.merge(a, on=keys, suffixes=("_b", "_a"), validate="one_to_one")
    if len(joined) != len(data) or len(joined) != len(a) or not joined.requests_a.eq(joined.requests_b).all():
        raise ValueError("Experiments do not use the same workload")
    if (not data.violations.eq(data.violations_demand).all()
            or not data.arrivals.sum() == len(sessions)
            or not data.failures.sum() == len(events)):
        raise ValueError("Reservation or session accounting audit failed")
    for kind, expected in [("requested_seconds", "requests"), ("outage_seconds", "unserved")]:
        if not sessions.groupby("algorithm")[kind].sum().sort_index().equals(
                data.groupby("algorithm")[expected].sum().sort_index()):
            raise ValueError(f"Session totals do not reconcile for {kind}")
    audits = sessions.groupby("algorithm")[["payment", "delivered_gb"]].sum().sort_index()
    checks = data.groupby("algorithm")[["supplied_traffic_payment", "pue_gb", "gnb_gb"]].sum().sort_index()
    np.testing.assert_allclose(audits.payment, checks.supplied_traffic_payment, atol=1e-9, rtol=0.)
    np.testing.assert_allclose(audits.delivered_gb, checks.pue_gb + checks.gnb_gb, atol=1e-9, rtol=0.)
    stats = summary(data, sessions, methods)
    baseline = summary(a, None, methods)
    stats.to_csv(output / "summary.csv", index=False)
    baseline.to_csv(output / "stateless_reference.csv", index=False)
    paired = joined.groupby(["scenario_id", "time_s"])[["cost_a", "cost_b"]].count()
    pair_keys = paired.loc[(paired == len(methods)).all(axis=1)].index
    pair = joined[joined.set_index(["scenario_id", "time_s"]).index.isin(pair_keys)]
    pair_summary = pair.groupby("algorithm").agg(decisions=("time_s", "size"), cost_a=("cost_a", "sum"),
                                                cost_b=("cost_b", "sum"), mean_rate_a=("rate_a", "mean"),
                                                mean_rate_b=("rate_b", "mean")).reset_index()
    pair_summary.to_csv(output / "common_feasible.csv", index=False)
    figs = output / "figures"
    figs.mkdir(exist_ok=True)
    plot_bars(stats, figs / "cost_rate_service.png",
              [("payment", "Total paid (price units)"), ("cost_per_new_request", "Cost per request (price units)"),
               ("mean_achievable_mbps", "Achievable rate (Mbps per decision)"),
               ("mean_delivered_mbps", "Delivered rate (Mbps per decision)"),
               ("initial_success_pct", "Initial allocation success (%)"),
               ("full_service_pct", "Full service (%) of requested seconds")],
              "Experiment B · all requests and decisions")
    plot_bars(stats, figs / "connections_resources.png",
              [("pue_connection_mean_s", "Mean time on PUE (seconds)"),
               ("gnb_fallback_mean_mode_s", "Mean time in fallback mode (seconds)"),
               ("pue_failures", "PUE failures and transitions"),
               ("immediate_fallbacks", "Failures served immediately"),
               ("pue_traffic_pct", "Delivered traffic from PUEs (%)"),
               ("mean_gnb_mhz_utilization", "Mean gNodeB bandwidth utilization (fraction)")],
              "Experiment B · persistence, fallback and resource utilization")
    plot_bars(stats, figs / "outages_utilization.png",
              [("unserved_pct", "Unserved request-seconds (%)"), ("partial_pct", "Partial service (%)"),
               ("total_outage_seconds", "Unserved request-seconds"),
               ("mean_pue_mbps_utilization", "Mean PUE throughput utilization (fraction)"),
               ("mean_gnb_mbps_utilization", "Mean gNodeB throughput utilization (fraction)"),
               ("mean_pue_mhz_utilization", "Mean PUE bandwidth utilization (fraction)")],
              "Experiment B · outages and resource use")
    plot_timeline(data, sessions, figs)
    exact = data.loc[(data.algorithm == "Exact") & data.arrivals.gt(0)]
    statuses = exact.groupby(["stage1_status", "stage2_status"], dropna=False).size().rename("initial_batches").reset_index()
    statuses.to_csv(output / "exact_initial_solver_status.csv", index=False)
    figure_path = "persistent_results/" if output.resolve() == (ROOT / "persistent_results").resolve() else ""
    connection_columns = ["algorithm", "new_requests", "initial_pue", "initial_gnb", "initial_allocation_failures",
                          "pue_failures", "immediate_fallbacks", "gnb_fallbacks_ever_served", "total_outage_seconds"]
    outcome_columns = ["algorithm", "served_seconds", "unserved_seconds", "full_service_pct", "payment",
                       "mean_achievable_mbps", "mean_delivered_mbps", "pue_gb", "gnb_gb"]
    report = f"""# Persistent MEC simulation with immediate gNodeB fallback

## Design

Experiment A allocates all active requests each second. Experiment B assigns each request once, when it first becomes active. The simulator retains the serving PUE and its reserved rates and bandwidth while the allocation meets the current link eligibility, radio, coverage, budget and resource checks. Every policy runs with independent lifecycle and resource state. The [audit](PERSISTENT_SIMULATION_AUDIT.md) identifies the source formulas, all four original policies and the interface differences with Phase 5.

An invalid PUE reservation is released in the same second. The request enters permanent gNodeB-only mode and attempts full-demand service immediately. An outage is recorded when the gNodeB check fails. An outage can clear when gNodeB service becomes feasible later in the request window. Calls to the four policies contain only new requests. Initial gNodeB-only service also stays on gNodeB for its lifetime. Initial failure leaves the request unserved for its window. Older reservations have priority, with request ID breaking ties. Fallback service claims free resources ahead of arrivals. Used bandwidth, including previously scheduled spare bandwidth, stays reserved while service remains valid.

The PHY and accounting use the same Mbps, MHz and GB units, capped PC5 radio rate, gNodeB Shannon rate, posted price and budget rules. The MEC fee of {cfg["eta"]} applies to every second with a PUE assignment. It ceases in a failed PUE's transition second. The LSTM is the existing inference pipeline with a {cfg["reliability_threshold"]} threshold. Both experiments use the same mobility and interference inputs. Random uses {cfg["random_trials"]} trials with master seed {cfg["seed"]} and the original per-scenario and per-second seed derivation. Exact solves its original cost-minimizing then rate-maximizing MILP on new requests using residual capacity and cost tolerance {cfg["epsilon"]}.

## Workload and results

Experiment B ran {len(ids)} {cfg["split"]} scenarios and {len(data)//len(methods):,} active decisions per method. The primary cost column sums actual supplied-traffic payments, including valid service during epochs in which some other requests were unserved. Whole-epoch objective cost retains the original undefined value for incomplete service. Means divide by all active decisions. Lower total payment can reflect lower delivery.

{markdown_table(stats[outcome_columns])}

{markdown_table(stats[connection_columns])}

A transition counts entry into gNodeB-only mode. An immediate fallback counts full gNodeB service in the same second as failure. The summary CSV records delayed gNodeB service, PUE duration, gNodeB-only duration, service rates, average cost, PUE and gNodeB payments, MEC fees and utilization. The experiment retains full service per request-second. An unserved second adds no traffic or charge.

### Exact solver status for initial-arrival batches

{markdown_table(statuses)}

HiGHS status 0 is optimal and 2 is infeasible. A blank stage-two status means the stage was skipped. The epoch output also retains stage costs, rates and solver gaps.

### Stateless reference over the same workload

{markdown_table(baseline[outcome_columns])}

Exact rejects an infeasible batch of new requests. Existing feasible persistent sessions continue independently. The heuristics can serve a feasible subset of a simultaneous arrival batch. The primary outcome includes every unserved request and uses the same workload denominator across algorithms.

### Common-feasible secondary comparison

{len(pair_keys):,} scenario-seconds had complete service for all four algorithms in both experiments. The charges below use complete objective bills. This comparison describes only that subset.

{markdown_table(pair_summary)}

## Figures and validation

- [Cost, rate and service]({figure_path}figures/cost_rate_service.png)
- [Connection and resource metrics]({figure_path}figures/connections_resources.png)
- [Outages and utilization]({figure_path}figures/outages_utilization.png)
- [Timeline and PUE connection duration distribution]({figure_path}figures/timeline.png)
- [Full persistent summary]({figure_path}summary.csv) and [stateless reference]({figure_path}stateless_reference.csv)

The per-scenario tables record epoch metrics, request histories and fallback transitions. Reservations were checked against the original evaluator at every decision. The report reconciles arrivals, failures, outages, costs and delivered volumes across epoch, request and event outputs. Tests compare initial allocations and Exact objectives at equivalent inputs, compare fresh inference and radio efficiencies with the reference runner, and verify retention, shared capacity, blocked fallback and permanent gNodeB mode for all four algorithms.

## Limitations and reproduction

The original radio interference is fixed by each network snapshot. Session continuity can reserve spare bandwidth and reduce later admission. Testing a different reservation policy or allowing reallocation after failure would be a separate temporal experiment. Outage and service times reflect observed request windows, including those ending at the trace boundary.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q objective_comparison/test_objectives.py objective_comparison/test_persistent.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m objective_comparison.persistent_run
MPLCONFIGDIR=/tmp/mec-matplotlib .venv/bin/python -m objective_comparison.persistent_report
```
"""
    target = ROOT / "PERSISTENT_SIMULATION_REPORT.md" if output.resolve() == (ROOT / "persistent_results").resolve() else output / "report.md"
    target.write_text(report)
    print(stats[outcome_columns + ["pue_failures", "immediate_fallbacks"]].to_string(index=False))


if __name__ == "__main__":
    main()
