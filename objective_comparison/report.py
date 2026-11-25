"""Summarize completed snapshots and produce reproducible comparison figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from objective_comparison.run import read_json
from src.utils.integrity import sha256_file


def read_records(root, scenario_ids, kind):
    return pd.concat((pd.read_csv(root / f"{sid}.{kind}.csv.gz") for sid in scenario_ids),
                     ignore_index=True)


def divide(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def summarize(group):
    volume = group.pue_gb.sum() + group.gnb_gb.sum()
    feasible = group.violations.eq(0)
    costs = group.loc[feasible, "cost"]
    return dict(
        eta=group.eta.iloc[0], scenarios=group.scenario_id.nunique(), decisions=len(group),
        requests=group.requests.sum(), served=group.served.sum(), partial=group.partial.sum(),
        unserved=group.unserved.sum(), feasible_decisions=feasible.sum(), feasibility_pct=100 * feasible.mean(),
        total_cost=costs.sum(min_count=1), mean_cost_per_consumer=divide(costs.sum(), group.loc[feasible, "requests"].sum()),
        cost_per_gb=divide(costs.sum(), group.loc[feasible, ["pue_gb", "gnb_gb"]].sum().sum()),
        pue_cost=group.loc[feasible, "pue_cost"].sum(), gnb_cost=group.loc[feasible, "gnb_cost"].sum(),
        mec_fee=group.loc[feasible, "mec_fee"].sum(),
        sum_snapshot_rate_mbps=group.rate.sum(), mean_snapshot_rate_mbps=group.rate.mean(),
        mean_rate_per_consumer=divide(group.rate.sum(), group.requests.sum()),
        pue_rate_sum_mbps=group.pue_rate.sum(), gnb_rate_sum_mbps=group.gnb_rate.sum(),
        mean_delivered_mbps=group.delivered_mbps.mean(), pue_gb=group.pue_gb.sum(), gnb_gb=group.gnb_gb.sum(),
        pue_pct=100 * divide(group.pue_gb.sum(), volume), gnb_pct=100 * divide(group.gnb_gb.sum(), volume),
        total_runtime_s=group.runtime_s.sum(), mean_runtime_ms=group.runtime_s.mean() * 1000,
        mean_scenario_runtime_s=group.groupby("scenario_id").runtime_s.sum().mean(),
        reliability_eligible_pct=100 * divide(group.reliable_selected.sum(), group.selected.sum()),
        violations=group.violations.sum(),
        **{name: group[name].sum() for name in group if name.startswith("violations_")},
    )


def markdown_table(frame):
    def fmt(val):
        if isinstance(val, (float, np.floating)):
            return "—" if np.isnan(val) else f"{val:,.6g}"
        return str(val)

    cols = list(frame.columns)
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    return "\n".join([header, divider] + [
        "| " + " | ".join(map(fmt, row)) + " |" for row in frame.itertuples(index=False, name=None)])


def figures(dest, paired, full, random_paired, exact):
    dest.mkdir(exist_ok=True)
    colors = ["#245b76", "#529b96", "#d08244", "#97557d"]
    fig, axs = plt.subplots(3, 3, figsize=(15, 11), layout="constrained")
    charts = [
        (paired, "total_cost", "Total cost", "Price units"),
        (paired, "mean_cost_per_consumer", "Mean cost per request-second", "Price units"),
        (paired, "cost_per_gb", "Cost per traffic volume", "Price units/GB"),
        (paired, "mean_snapshot_rate_mbps", "Mean achievable snapshot rate", "Mbps"),
        (paired, "mean_rate_per_consumer", "Achievable rate per request-second", "Mbps"),
        (full, "served", "Full-demand request-seconds served", "Request-seconds"),
        (full, "mean_runtime_ms", "Mean allocation and evaluation runtime", "ms/decision"),
        (full, "feasibility_pct", "Fully feasible decisions", "% of decisions"),
        (full, "violations_demand", "Unmet request-seconds in returned outputs", "Request-seconds"),
    ]
    for ax, (frame, metric, title, ylabel) in zip(axs.flat, charts):
        ax.bar(frame.algorithm, frame[metric], color=colors)
        ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("MEC allocation comparison · paired cost and rate, all-epoch diagnostics", fontsize=15)
    fig.savefig(dest / "comparison.png", dpi=155)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    ax = axs[0]
    ax.bar(paired.algorithm, paired.pue_gb, label="PUE", color=colors[0])
    ax.bar(paired.algorithm, paired.gnb_gb, bottom=paired.pue_gb, label="gNodeB", color=colors[2])
    ax.set(title="Traffic supplied on paired epochs", ylabel="GB")
    ax.legend()
    ax = axs[1]
    for color, rec in zip(colors, paired.itertuples(index=False)):
        ax.scatter(rec.cost_per_gb, rec.mean_snapshot_rate_mbps, label=rec.algorithm, color=color)
    ax.set(title="Paired cost and achievable rate", xlabel="Price units/GB", ylabel="Mean Mbps")
    ax.legend()
    ax = axs[2]
    random_by_epoch = random_paired.groupby(["scenario_id", "time_s"]).rate.agg(["mean", "std"])
    ax.bar(["Candidate mean", "Selected mean"], [random_by_epoch["mean"].mean(),
           random_paired.loc[random_paired.selected_trial, "rate"].mean()],
           yerr=[random_by_epoch["std"].mean(), 0], capsize=5, color=colors[1:3])
    ax.set(title="Random variability on paired epochs", ylabel="Mbps")
    fig.savefig(dest / "traffic_tradeoff_random.png", dpi=155)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    cols = ["stage1_gap", "stage2_gap"]
    gaps = exact[cols].max().fillna(0) * 100
    ax.bar(["Cost stage", "Rate stage"], gaps, color=colors[:2])
    for i, value in enumerate(gaps):
        count = exact[cols[i]].notna().sum()
        ax.annotate(f"{value:.4g}% across {count:,} gaps", (i, value), xytext=(0, 8),
                    textcoords="offset points", ha="center")
    ax.set(title="Maximum reported Exact optimality gaps", ylabel="%")
    fig.savefig(dest / "solver_gaps.png", dpi=155)
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    dest = cli.parse_args().output
    meta = read_json(dest / "metadata.json")
    cfg = meta["config"]
    epochs = read_records(dest, meta["scenario_ids"], "epochs")
    random = read_records(dest, meta["scenario_ids"], "trials")
    methods = cfg["algorithms"]
    keys = ["scenario_id", "time_s"]
    epochs["feasible"] = epochs.violations.eq(0)
    feasible = epochs.pivot(index=keys, columns="algorithm", values="feasible")
    if feasible.isna().any().any() or set(feasible.columns) != set(methods):
        raise RuntimeError("The experiment is missing paired decisions")
    manifest = Path(meta["phase3_config"]["output_root"]) / cfg["split"] / "phase3_manifest.csv"
    if sha256_file(manifest) != meta["manifest_sha256"]:
        raise RuntimeError("Scenario manifest changed after the experiment")
    expected = pd.read_csv(manifest).set_index("scenario_id").loc[meta["scenario_ids"], "request_seconds"]
    expected = expected.sort_index()
    completed = epochs.groupby(["algorithm", "scenario_id"]).requests.sum()
    if meta["epoch_limit"] is None and any(
        not np.array_equal(completed.loc[method].reindex(expected.index), expected) for method in methods
    ):
        raise RuntimeError("Experiment request counts disagree with the scenario manifest")
    if len(random) != len(feasible) * cfg["random_trials"]:
        raise RuntimeError("Random trial count disagrees with the experiment settings")
    paired_keys = feasible.index[feasible.all(axis=1)]
    paired_rows = epochs[epochs.set_index(keys).index.isin(paired_keys)]
    all_rows = pd.DataFrame([dict(algorithm=m, **summarize(epochs[epochs.algorithm == m])) for m in methods])
    paired = pd.DataFrame([dict(algorithm=m, **summarize(paired_rows[paired_rows.algorithm == m])) for m in methods])
    scenario = pd.DataFrame([dict(scenario_id=s, algorithm=m, **summarize(d))
                             for (s, m), d in epochs.groupby(["scenario_id", "algorithm"])])
    config_groups = epochs.assign(scenario_config=epochs.scenario_id.str.split("__").str[0])
    per_config = pd.DataFrame([dict(scenario_config=c, algorithm=m, **summarize(d))
                               for (c, m), d in config_groups.groupby(["scenario_config", "algorithm"])])
    exact = epochs.loc[epochs.algorithm == "Exact"].copy()
    exact["proven_optimal"] = exact.proven_optimal.eq(True)
    status = exact.groupby(["stage1_status", "stage2_status", "proven_optimal"], dropna=False).size().rename("decisions").reset_index()
    random_valid = random[random.violations.eq(0)]
    random_paired = random_valid[random_valid.set_index(keys).index.isin(paired_keys)]
    trial_stats = random_valid.groupby(keys)[["cost", "rate"]].agg(["count", "min", "mean", "median", "max", "std"])
    trial_stats.columns = ["_".join(v) for v in trial_stats.columns]
    trial_stats = random.groupby(keys).size().rename("attempts").to_frame().join(trial_stats)
    trial_stats["cost_count"] = trial_stats.cost_count.fillna(0).astype(int)
    trial_stats["rate_count"] = trial_stats.rate_count.fillna(0).astype(int)
    variability = random_valid[["cost", "rate"]].agg(["count", "min", "mean", "median", "max", "std"]).T
    variability["mean_within_epoch_std"] = [trial_stats[f"{m}_std"].mean() for m in variability.index]
    random_selected = random_valid[random_valid.selected_trial]
    variability["selected_mean"] = [random_selected[m].mean() for m in variability.index]
    variability["selected_best"] = [random_selected.cost.min(), random_selected.rate.max()]
    variability["selected_worst"] = [random_selected.cost.max(), random_selected.rate.min()]
    matched_exact = exact.set_index(keys).loc[paired_keys]
    other = paired_rows.pivot(index=keys, columns="algorithm", values=["cost", "rate"])
    expense_margin = other["cost"].subtract(matched_exact.stage1_cost, axis=0)
    within_band = expense_margin.le(cfg["epsilon"] + 1e-9)
    cost_errors = int(expense_margin.lt(-1e-9).sum().sum())
    rate_errors = int((within_band & other["rate"].gt(matched_exact.rate + 1e-6, axis=0)).sum().sum())
    audit_only = [name for name in epochs if name.startswith("violations_") and name != "violations_demand"]
    validation = {
        "scenario_count": int(epochs.scenario_id.nunique()),
        "decisions_per_algorithm": int(len(feasible)),
        "common_feasible_decisions": int(len(paired_keys)),
        "paired_minimum_cost_mismatches": cost_errors,
        "paired_rate_bound_mismatches": rate_errors,
        "non_demand_violations": epochs.groupby("algorithm")[audit_only].sum().to_dict(orient="index"),
        "exact_proven_optimal": int(exact.proven_optimal.sum()),
        "exact_proven_infeasible": int(exact.stage1_status.eq(2).sum()),
        "exact_optimal_status_both_stages": int((exact.stage1_status.eq(0) & exact.stage2_status.eq(0)).sum()),
        "exact_solver_limits_or_errors": int((~exact.stage1_status.isin([0, 2]) |
                                            (exact.stage1_status.eq(0) & ~exact.stage2_status.eq(0))).sum()),
        "trial_count": int(len(random)),
        "feasible_trial_count": int(len(random_valid)),
        "sources_sha256": {str(path): sha256_file(path) for path in sorted(Path("objective_comparison").glob("*.py"))},
    }
    (dest / "validation.json").write_text(json.dumps(validation, indent=2))
    epochs.to_csv(dest / "raw_results.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 3})
    random.to_csv(dest / "random_trials.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 3})
    trial_stats.to_csv(dest / "random_trial_statistics.csv")
    variability.to_csv(dest / "random_summary.csv")
    scenario.to_csv(dest / "scenario_results.csv", index=False)
    per_config.to_csv(dest / "scenario_config_results.csv", index=False)
    all_rows.to_csv(dest / "all_epoch_summary.csv", index=False)
    paired.to_csv(dest / "paired_summary.csv", index=False)
    status.to_csv(dest / "solver_status.csv", index=False)
    figures(dest / "figures", paired, all_rows, random_paired, exact)
    comparison = paired[["algorithm", "eta", "total_cost", "mean_cost_per_consumer", "cost_per_gb",
                         "mean_snapshot_rate_mbps", "mean_rate_per_consumer", "pue_pct"]].copy()
    comparison["stage1_min_total"] = np.where(comparison.algorithm.eq("Exact"), matched_exact.stage1_cost.sum(), np.nan)
    components = paired[["algorithm", "pue_cost", "gnb_cost", "mec_fee", "pue_rate_sum_mbps",
                         "gnb_rate_sum_mbps", "sum_snapshot_rate_mbps", "pue_gb", "gnb_gb"]]
    outcomes = all_rows[["algorithm", "feasible_decisions", "feasibility_pct", "served", "partial",
                         "unserved", "violations", "mean_runtime_ms", "total_runtime_s"]]
    lstm = pd.DataFrame(meta["lstm"]).T.rename_axis("SUMO run").reset_index()
    probe = epochs[epochs.algorithm == "Exact"]
    peak = paired.loc[paired.mean_snapshot_rate_mbps.idxmax(), "algorithm"]
    low = paired.loc[paired.total_cost.idxmin(), "algorithm"]
    fast = all_rows.loc[all_rows.mean_runtime_ms.idxmin(), "algorithm"]
    demand = probe.demand_mbps.describe()
    epsilon_share = 100 * divide(cfg["epsilon"], cfg["eta"])
    epsilon_note = f" ({epsilon_share:g}% of one fee)" if cfg["eta"] else ""
    best_feasibility = all_rows.feasible_decisions.max()
    feasible_best = ", ".join(all_rows.loc[all_rows.feasible_decisions.eq(best_feasibility), "algorithm"])
    artifact_meta = read_json(meta["phase3_config"]["reliability"]["metadata"])
    disjoint = set(meta["lstm"]).isdisjoint(artifact_meta["split"]["train_runs"])
    expected_split = set(meta["lstm"]).issubset(artifact_meta["split"][f"{cfg['split']}_runs"])
    if not disjoint or not expected_split:
        raise RuntimeError("Experiment mobility runs disagree with the model data split")
    written = f"""# MEC marketplace experiment

## Findings

Four algorithms ran on {len(meta["scenario_ids"])} existing held-out scenarios from {len(lstm)} SUMO runs, using the Phase 3 geometry channel `{cfg["channel"]}`. Each method evaluated {len(feasible):,} active one-second decisions and {int(probe.requests.sum()):,} request-seconds. The demand per active snapshot had a median of {demand["50%"]:.3f} Mbps and a range of {demand["min"]:.3f}–{demand["max"]:.3f} Mbps. All four methods returned fully feasible allocations on {len(paired_keys):,} common decisions.

{low} had the lowest summed cost among the returned paired allocations. Exact's Stage 1 separately establishes the minimum cost before the allowed Stage 2 increase. {peak} had the highest mean achievable radio rate in the paired comparison. {fast} had the lowest mean runtime. The largest number of fully feasible decisions was {best_feasibility:,}, achieved by {feasible_best}. The rates in the table are scheduled radio capacities. The demand actually delivered can be reconstructed from the allocated GB or read in the raw results. An epoch with failed full service has no valid objective cost.

## Paired cost and rate

{markdown_table(comparison)}

### Paired cost, capacity and volume components

{markdown_table(components)}

The capacity-sum columns add the instantaneous Mbps totals across one-second snapshots. Use the mean rate for a time-average comparison.

The stage-one cost column is the sum of the per-epoch proven minima. Stage two may increase cost by up to epsilon in each epoch. A heuristic outside an epoch's cost band can achieve a higher rate at a higher price. The paired solver audit found {cost_errors} costs below the Stage 1 bound and {rate_errors} rates above the Stage 2 bound within the band, using numerical comparison tolerances of 10⁻⁹ price units and 10⁻⁶ Mbps.

## All-epoch service and feasibility

{markdown_table(outcomes)}

Counts are request-seconds and include repeated service to the same physical consumer. The solver records an empty diagnostic output for a proven-infeasible epoch. Its unmet-demand count reflects the absence of a full-service solution. Heuristics can report partial progress from fully served individual requests in the same epoch. See the per-family violations in the CSV. The number of violations does not measure the relative quality of these infeasible outputs. 

## Model and protocol

See [repository and variable audit](../AUDIT.md) for the complete mapping. Costs include PUE GB at the posted price, remaining GB at the gNodeB price and the MEC fee for a PUE acquisition. The fixed, assumed fee was **η = {cfg["eta"]} price units per PUE acquisition** in every epoch and method. Epsilon was fixed at **{cfg["epsilon"]} price units per epoch**{epsilon_note} to allow a small absolute cost tolerance. The method set uses the same budget, coverage, provider access and backhaul resources, cell capacity, legacy link peak, one-PUE rule, and radio resource limits. The positivity floor for an active allocation in the solver was {cfg["min_selected_mbps"]} Mbps.

- Greedy processes requests by descending priority, ascending arrival and request ID. It examines the gNodeB-only option and the cost-minimizing endpoint of each feasible relay split interval. Within a local epsilon tie it prefers higher PUE spectral efficiency. It then assigns spare bandwidth in descending efficiency order to selected links. The procedure has no global cost guarantee.
- Random generates {cfg["random_trials"]} seeded attempts per epoch by shuffling request order, uniformly sampling a feasible split interval per option and sampling an option. Incomplete attempts remain in the log. Among feasible attempts it finds the minimum cost and then picks the highest rate within the epsilon band. When all attempts fail it reports the attempt serving the most requests, with delivered traffic as a tiebreaker. The master seed was {cfg["seed"]} and each scenario and timestamp derived a stable seed from SHA-256.
- Best-SINR uses the same sequential order as Greedy and gives first preference to the feasible relay with highest instantaneous SINR. The splitter uses the largest feasible relay share. It uses gNodeB alone only when no feasible relay split exists. The heuristic's cost and rate are evaluated independently.
- Exact uses two HiGHS MILPs. Stage 1 minimizes price and fees. Stage 2 maximizes radio rate with cost bounded by Stage 1 plus epsilon. Fixed SINR makes each log coefficient constant. Positive traffic and bandwidth are enabled by binary selection, which removes the selection–volume product. Bandwidth can be capped at the point at which a PUE's link limit saturates, so the minimum operator becomes linear without loss under this model. Solver time limit was {cfg["solver_time_limit_s"]} s per stage with requested relative gap {cfg["solver_relative_gap"]}. The raw results store both stages and are independently checked against the evaluator.

## Radio and inference

SINR is dynamically derived from the existing scenario and mobility data using the configured propagation and interference model. It is a simulation-derived quantity, not a directly measured SINR value from the original dataset.

The PC5 link uses the repository TR 37.885 urban V2V path loss at 6 GHz, 23 dBm, 20 MHz, 3 dBi at both ends, and 9 dB receiver noise figure. Candidate path states come from Phase 3 geometry realizations. The calculation uses median NLOSv blockage and zero additional shadow-fading draws. Out-of-range interfering paths without a state use LOS. {cfg["pc5_interference"]} {cfg["gnb_interference"]} The gNodeB surrogate uses the same LOS propagation equations with the configuration's 3.5 GHz, 46 dBm and antenna overrides. The surrogate is an explicit assumption because the repository provides site positions and Mbps capacities without gNodeB RF measurements. Thermal noise is -174 dBm/Hz plus full-carrier bandwidth and receiver noise figure. Desired and interfering powers are converted into mW before forming the ratio. Integer channel choice and changing mobility affect the derived ratios. The per-scenario link tables include the intermediate radio quantities and eligibility.

The checkpoint metadata records a mobility-run-disjoint split. Every experiment trace belonged to the selected {cfg["split"]} split and was absent from training. The existing checkpoint, model configuration and scaler matched their recorded hashes before and after the run. The existing online features and fitted scaler produced fresh inference probabilities in [0, 1]. Inference used evaluation mode and the model's parameters were read from disk. The applied threshold for all methods was **{cfg["reliability_threshold"]}** from corrected metadata. Its range and horizon describe link survival independently of the radio SINR. Fresh probabilities matched the existing Phase 3 cache to the validation tolerance with the same eligibility decisions.

{markdown_table(lstm)}

## Random attempts and solver validation

Random generated {len(random):,} attempts, of which {len(random_valid):,} passed the full evaluation. The statistics pool feasible trials over all epochs. Per-epoch statistics are also available. Lower cost and higher rate define the favorable extremes of their respective metrics.

{markdown_table(variability.rename_axis("metric").reset_index())}

{markdown_table(status)}

HiGHS statuses are 0 for optimal, 1 for a resource or iteration limit, 2 for infeasible, 3 for unbounded and 4 for solver error. An empty Stage 2 status means the stage was skipped. The strict proof flag also requires both reported gaps to equal exactly zero. A solver status of 0 with a tiny floating-point gap can therefore appear with the flag unset. The maximum reported Stage 1 and Stage 2 gaps were {exact.stage1_gap.max():.6g} and {exact.stage2_gap.max():.6g}. Runtime includes the objective evaluator. Shared snapshot time is logged separately. Full-run wall time was {meta["wall_time_s"]:.1f} seconds.

Small-case validation checks distance, path loss, received power, thermal noise, multiple interferers, isolated transmitters, movement, reproducibility and invalid coordinates. A hand-computed 8 Mbps example uses 6 Mbps PUE plus 2 Mbps gNodeB, costs 0.00130 price units including the fee and provides 14 Mbps achievable radio capacity. All four methods were exercised. The audit tests also inject invalid demands, excessive resource use and a reliability failure.

## Limitations

The full-demand equation is infeasible when the fixed budgets, radio links and network capacity cannot support all requests simultaneously. The feasible subset defines the paired objective comparison and can favor easier epochs. The results describe an independent snapshot experiment using held-out SUMO mobility and synthetic demand, budgets and prices. Channel reuse was held fixed throughout each decision. A scheduler with allocation-dependent interference could change the ranking and would require a different optimization model. A link carrying a small positive volume can contribute spare scheduled capacity to the rate objective. The delivered traffic metric remains capped by demand, backhaul and cell throughput. Only the geometry channel and the primary threshold were evaluated.

## Reproduction and figures

Run from the repository root with the project's existing Python environment.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest objective_comparison.test_objectives
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m objective_comparison.run --output objective_comparison/reproduction
MPLCONFIGDIR=/tmp/mec-matplotlib .venv/bin/python -m objective_comparison.report objective_comparison/reproduction
```

- [Cost, rate, service, feasibility and runtime](figures/comparison.png)
- [Traffic contributions, tradeoff and Random variability](figures/traffic_tradeoff_random.png)
- [Solver gaps](figures/solver_gaps.png)
- [Epoch results](raw_results.csv.gz), [paired summary](paired_summary.csv), [all-epoch summary](all_epoch_summary.csv) and [scenario results](scenario_results.csv) and [scenario configuration results](scenario_config_results.csv)
- [Random trials](random_trials.csv.gz), [per-epoch Random statistics](random_trial_statistics.csv), [solver status](solver_status.csv), [validation](validation.json) and [experiment metadata](metadata.json)
"""
    (dest / "experiment_report.md").write_text(written)
    print(comparison.to_string(index=False))
    print(outcomes.to_string(index=False))
    if cost_errors or rate_errors:
        raise RuntimeError("Paired optimum audit detected violations")


if __name__ == "__main__":
    main()
