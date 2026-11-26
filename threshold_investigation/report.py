"""Build the report from the validation-only sweep and existing validation anchors."""
import argparse
import csv
from itertools import groupby
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from objective_comparison.report import markdown_table
from src.utils.integrity import sha256_file
from threshold_investigation.run import CONFIG, GRID, SOURCE, read


def aggregate(rows):
    keys = ["allocator", "threshold"]
    counts = [
        "requests", "admitted", "blocked", "completed", "dropped", "censored", "cue_departed",
        "eligible_links", "eligible_providers", "eligible_request_seconds", "offered_request_seconds",
        "interruption_episodes", "interrupted_seconds", "sessions_interrupted", "recoveries",
        "reallocations", "rebalances", "proactive_switches", "active_seconds", "delivered_mbit",
        "window_required_mbit", "pue_mbit", "gnb_mbit", "payment", "audit_violations", "ledger_problems", "errors",
    ]
    s = rows.groupby(keys)[[col for col in counts if col in rows]].sum().reset_index()
    s["scenarios"] = rows.groupby(keys).scenario_id.nunique().to_numpy()
    s["admission_pct"] = 100 * s.admitted / s.requests
    s["pue_traffic_pct"] = 100 * s.pue_mbit / s.delivered_mbit
    s["gnb_traffic_pct"] = 100 * s.gnb_mbit / s.delivered_mbit
    s["cost_per_gb"] = 8000 * s.payment / s.delivered_mbit
    s["delivered_demand_pct"] = 100 * s.delivered_mbit / s.window_required_mbit
    s["interruptions_per_100_admitted"] = 100 * s.interruption_episodes / s.admitted
    return s


def charts(frame, columns, path, intro, ylabel, legend=False):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, (column, label) in zip(axes.flat, columns):
        ax.plot(frame.threshold, frame[column], marker="o", markersize=3, color="#245b76")
        ax.axvline(.62, color="#bd6634", ls="--", alpha=.8, label="Primary 0.62")
        ax.axvline(.65, color="#5b8777", ls=":", alpha=.8, label="Sensitivity 0.65")
        if legend and column == "precision_raw":
            ax.axhline(.90, color="#777777", linestyle="-.", label="Precision floor")
            ax.plot(frame.threshold, frame.precision_lb95_onesided_pair, color="#7d507e",
                    label="Pair-cluster lower bound")
            ax.legend(fontsize=7)
        ax.set(title=label, xlabel="Threshold θ", ylabel=ylabel)
        ax.grid(alpha=.2)
    fig.suptitle(intro, fontsize=15)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def read_runs(path):
    """Keep the stable metrics and ground-truth fields from each scenario batch."""
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        audited = [key for key in header if key.startswith("gt_")]
        stable = header[:next(i for i, key in enumerate(header) if key.startswith(("sc_", "gt_")))]
        records = []
        for sid, batch in groupby(reader, key=lambda row: row[0]):
            batch = list(batch)
            # The initial capture emitted optional switch counters before the audit fields
            # if present at the first threshold, or after them if first seen at a later θ.
            # In the first row only the latter cells are empty, so the audit block ends
            # at the last populated cell. The coordinator's audit block is always numeric.
            audit_end = len(batch[0])
            while batch[0][audit_end - 1] == "":
                audit_end -= 1
            audit_start = audit_end - len(audited)
            if audit_start < len(stable) or any(len(row) != len(batch[0]) for row in batch):
                raise ValueError(f"Invalid scenario result batch: {sid}")
            records.extend(row[:len(stable)] + row[audit_start:audit_end] for row in batch)
    result = pd.DataFrame(records, columns=stable + audited)
    for key in result.columns.difference(["scenario_id", "run_id", "allocator"]):
        result[key] = pd.to_numeric(result[key], errors="raise")
    return result


def verify_scenario_inputs(scenario_ids):
    roots = read(CONFIG)["inputs"]
    phase3 = Path(roots["phase3_root"]) / "val"
    phase2 = Path(roots["phase2_root"]) / "val"
    specs = [
        (phase3, "phase3_manifest.csv", ("request_seconds", "candidate_links")),
        (phase2, "marketplace_scenarios.csv",
         ("cue_requests", "pue_resources", "gnodeb_resources", "pue_withdrawals")),
    ]
    checked = 0
    for root, filename, artifacts in specs:
        rows = pd.read_csv(root / filename).set_index("scenario_id").loc[scenario_ids]
        if not rows.validation_passed.all():
            raise ValueError(f"Recorded validation failed in {filename}")
        for sid, row in rows.iterrows():
            for artifact in artifacts:
                suffix = ".csv" if artifact == "pue_withdrawals" else ".csv.gz"
                path = root / sid / f"{artifact}{suffix}"
                if sha256_file(path) != row[f"sha256_{artifact}"]:
                    raise ValueError(f"Scenario artifact differs from its manifest: {path}")
                checked += 1
    return checked


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    dest = cli.parse_args().output
    proto = read(dest / "protocol.json")
    raw = read_runs(dest / "scenario_results.csv")
    observed = pd.MultiIndex.from_frame(raw[["scenario_id", "threshold"]])
    expected = pd.MultiIndex.from_product([proto["scenarios"], GRID], names=observed.names)
    if observed.has_duplicates or not observed.sort_values().equals(expected.sort_values()):
        raise ValueError("Incomplete or duplicate scenario results")
    if not raw.allocator.eq(proto["allocator"]).all():
        raise ValueError("Unexpected allocation method in replay results")
    if raw[["audit_violations", "ledger_problems", "errors"]].sum().any():
        raise ValueError("Marketplace checks failed")
    truth = [
        ("departure_events", "departure_agree"), ("out_of_range_events", "out_of_range_agree"),
        ("censored_sessions", "censored_consistent"), ("cue_departed_sessions", "cue_departed_consistent"),
    ]
    if (any((raw[f"gt_{a}"] != raw[f"gt_{b}"]).any() for a, b in truth)
            or raw.gt_departure_on_censored_vehicle.any()
            or not raw.gt_cue_departed_sessions.eq(raw.cue_departed).all()
            or not raw.gt_censored_sessions.eq(raw.censored).all()):
        raise ValueError("Session ground-truth validation failed")
    for filename, digest in proto["input_hashes"].items():
        if sha256_file(filename) != digest:
            raise ValueError(f"Input changed since sweep start: {filename}")
    loc = Path("data/mec/phase5/val/grid")
    recorded = read(loc / "summary.json")
    if recorded["split"] != "val" or sha256_file(loc / "runs.csv") != recorded["runs_csv_sha256"]:
        raise ValueError("Existing Phase 5 validation grid provenance failed")
    anchor = pd.read_csv(loc / "runs.csv")
    anchor = anchor[(anchor.channel == proto["channel"]) & anchor.proactive &
                    anchor.theta_tag.isin(["primary", "sensitivity"]) &
                    anchor.scenario_id.isin(proto["scenarios"])].copy()
    if not anchor.max_interruption_s.eq(proto["interruption_tolerance_s"]).all():
        raise ValueError("Phase 5 reference uses a different interruption tolerance")
    anchor["threshold"] = anchor.theta_tag.map({"primary": .62, "sensitivity": .65})
    compare = raw.merge(anchor, on=["allocator", "threshold", "scenario_id"], suffixes=("", "_recorded"),
                        validate="one_to_one")
    audit_fields = ["admitted", "dropped", "recoveries", "reallocations", "interruption_episodes",
                    "active_seconds", "delivered_mbit", "payment"]
    if len(compare) != 2 * len(proto["scenarios"]):
        raise ValueError("Missing paired reference rows")
    discrepancies = {name: int((abs(compare[name] - compare[f"{name}_recorded"]) > 1e-8).sum())
                     for name in audit_fields}
    if any(discrepancies.values()):
        raise ValueError(f"Reference thresholds differ from recorded validation results: {discrepancies}")

    inputs = read(CONFIG)["inputs"]
    phase2 = pd.read_csv(Path(inputs["phase2_root"]) / "val/marketplace_scenarios.csv")
    base_seeds = sorted(phase2.loc[phase2.scenario_id.isin(proto["scenarios"]), "base_seed"].unique())
    radio_seed = read(Path(inputs["phase3_root"]) / "val/experiment_config.json")["config"]["channel_seed"]
    c = pd.read_csv(dest / "classification.csv")
    eligible_primary = c[c.precision_raw.ge(.90) & c.recall_raw.ge(.50)]
    eligible_sensitivity = c[c.precision_lb95_onesided_pair.ge(.90) & c.recall_lb95_onesided_pair.ge(.50)]
    if (proto["rule"] != "precision_constrained" or proto["selected"] != eligible_primary.threshold.min()
            or proto["selected"] != .62 or eligible_sensitivity.threshold.min() != .65):
        raise ValueError("Reconstructed threshold decisions disagree with recorded validation rules")
    selected = c[c.threshold.isin(GRID)]
    checks = {
        "scenario_threshold_runs": len(raw),
        "scenario_files_verified": verify_scenario_inputs(proto["scenarios"]),
        "matched_recorded_anchor_runs": len(compare),
        "maximum_outcome_differences_above_tolerance": discrepancies,
        "audit_violations": int(raw.audit_violations.sum()),
        "ledger_problems": int(raw.ledger_problems.sum()),
        "errors": int(raw.errors.sum()),
        "evaluated_ground_truth_events": {a: int(raw[f"gt_{a}"].sum()) for a, _ in truth},
        "primary": float(proto["selected"]),
        "sensitivity": float(eligible_sensitivity.threshold.min()),
        "phase5_reference_summary_sha256": sha256_file(loc / "summary.json"),
        "code_sha256": {str(p): sha256_file(p) for p in Path("threshold_investigation").glob("*.py")},
    }
    (dest / "validation_checks.json").write_text(json.dumps(checks, indent=2))
    m = aggregate(raw)
    anchors = aggregate(anchor)
    raw.to_csv(dest / "marketplace_runs.csv", index=False)
    m.to_csv(dest / "marketplace_summary.csv", index=False)
    anchors.to_csv(dest / "reference_summary.csv", index=False)
    by_run = pd.concat([aggregate(group).assign(run_id=run_id)
                        for run_id, group in raw.groupby("run_id")], ignore_index=True)
    by_run.to_csv(dest / "marketplace_by_mobility_run.csv", index=False)
    tables = dest / "figures"
    tables.mkdir(exist_ok=True)
    charts(c, [
        ("precision_raw", "Precision"), ("recall_raw", "Recall"), ("f1_score", "F1"),
        ("balanced_accuracy_raw", "Balanced accuracy"), ("false_positive_rate", "False-positive rate"),
        ("false_negative_rate", "False-negative rate"),
    ], tables / "classification.png", "Validation classification across the full threshold grid", "Fraction", True)
    charts(m, [
        ("eligible_links", "Eligible PUE–CUE link-seconds"), ("admission_pct", "Session admission (%)"),
        ("pue_traffic_pct", "PUE traffic share (%)"), ("cost_per_gb", "Payment per GB"),
        ("interruptions_per_100_admitted", "Interruption episodes per 100 admitted"),
        ("dropped", "Dropped sessions"),
    ], tables / "marketplace.png", "Validation marketplace sensitivity", "")

    def cls(theta):
        return c[c.threshold.eq(theta)].iloc[0]

    def market(theta):
        return m[m.threshold.eq(theta)].iloc[0]

    base = market(.62)
    baseline = cls(.62)
    comparisons = []
    for theta in GRID:
        v, p = market(theta), cls(theta)
        comparisons.append({
            "θ": theta, "Δ precision (pp)": 100 * (p.precision_raw - baseline.precision_raw),
            "Δ recall (pp)": 100 * (p.recall_raw - baseline.recall_raw),
            "Δ eligible links (%)": 100 * (v.eligible_links / base.eligible_links - 1),
            "Δ sessions admitted": v.admitted - base.admitted,
            "Δ PUE traffic (pp)": v.pue_traffic_pct - base.pue_traffic_pct,
            "Δ cost/GB": v.cost_per_gb - base.cost_per_gb,
            "Δ interruption episodes": v.interruption_episodes - base.interruption_episodes,
        })
    cls_cols = ["threshold", "precision_raw", "recall_raw", "specificity", "f1_score",
                "balanced_accuracy_raw", "false_positive_rate", "false_negative_rate",
                "predicted_positive_count", "predicted_ineligible_count", "precision_lb95_onesided_pair"]
    market_cols = ["threshold", "eligible_links", "eligible_providers", "eligible_request_seconds",
                   "admitted", "admission_pct", "active_seconds", "pue_traffic_pct", "gnb_traffic_pct",
                   "cost_per_gb", "delivered_demand_pct"]
    session_cols = ["threshold", "interruption_episodes", "interrupted_seconds", "dropped",
                    "recoveries", "reallocations", "proactive_switches", "interruptions_per_100_admitted"]
    robust = cls(.65)
    f1 = c.loc[c.f1_score.idxmax()]
    bacc = c.loc[c.balanced_accuracy_raw.idxmax()]
    history = read(SOURCE / "reliability_threshold/selected_threshold.json")
    r60, r61, r67 = cls(.60), cls(.61), cls(.67)
    near = m[m.threshold.isin([.60, .62, .65, .67])]
    span = lambda metric: near[metric].max() - near[metric].min()
    lower, stricter, high = market(.60), market(.65), market(.67)
    report = f"""# Validation-only LSTM reliability threshold investigation

## Recommendation

Retain **0.62 as the primary threshold** under the existing point-estimate policy. It is the smallest cutoff on the predeclared 0.50–0.90 grid (step 0.01) with at least 90% precision and 50% recall. It yielded **{baseline.precision_raw:.3%} precision**, **{baseline.recall_raw:.3%} recall** and approved **{baseline.predicted_positive_count:,}** labeled link windows. Keep **0.65** as the stricter sensitivity setting. Its estimated one-sided 95% pair-cluster precision lower bound was **{robust.precision_lb95_onesided_pair:.2%}**, compared with **{baseline.precision_lb95_onesided_pair:.2%}** at 0.62. The corresponding recall lower bound at 0.65 was **{robust.recall_lb95_onesided_pair:.2%}**, above 50%. At 0.62 the worst validation run had **{baseline.precision_min_over_val_runs:.2%}** precision. The point criterion and the lower-bound criterion answer different questions. The recommendation assumes the original point criterion. The immediately lower grid point, 0.61, had {r61.precision_raw:.3%} precision. Selecting the smallest qualifying threshold retains the largest candidate set among the qualifying grid points.

## Repository audit

- Phase 1's [selection script](../../scripts/sweep_reliability_threshold.py) tried 41 thresholds and applied both floors to unrounded validation rates. Its predeclared fallback was maximum balanced accuracy, which was unnecessary. The historical [selection record](../../outputs/large_dataset_experiment/reliability_threshold/selected_threshold.json) lists qualifying thresholds {history["thresholds_meeting_both_constraints"][0]:.2f}–{history["thresholds_meeting_both_constraints"][-1]:.2f}. Marketplace outcomes were outside the selection rule. On this grid, maximum F1 selects {f1.threshold:.2f}, and maximum balanced accuracy selects {bacc.threshold:.2f}. These objectives trade false approvals and missed survivals differently and do not enforce the precision floor.
- [Checkpoint metadata](../../outputs/large_dataset_experiment/checkpoints/checkpoint_metadata.json) records 0.62 for eligibility and the earlier 0.53 classifier operating point separately. The checkpoint's stored default is 0.50. [Phase 3 configuration](../../config/simulation/phase3.json) resolves primary and sensitivity metadata. [Recorded validation settings](../../data/simulation/phase3/val/experiment_config.json) confirm 0.62 and 0.65. [Phase 5](../../config/mec/phase5.json) uses policy tags for these cutoffs, and the [four-algorithm experiment](../../objective_comparison/config.json) explicitly pins 0.62.
- [Grouped confidence analysis](../../scripts/threshold_grouped_ci.py) chose 0.65 as the smallest threshold meeting its lower-bound rule. It used 2,000 bootstrap resamples (seed 42), with 594 validation pair clusters, 108 consumer clusters and four mobility runs. Shared mobility conditions make pair-cluster uncertainty estimates potentially optimistic.

## Method

The fixed LSTM scores a 10-second mobility history for 30-second distance survival within 1,000 m. The rule approves scores greater than or equal to θ. Changing θ changes which links pass the filter and leaves the network weights and scores intact. Classification metrics were reconstructed from the original validation confusion counts for 25,602 labeled windows across runs `run_000`, `run_001`, `run_006`, and `run_013`. Existing validation confidence bounds were reused.

The marketplace sweep replayed the same {len(proto["scenarios"])} validation scenarios for all nine thresholds, using the existing Phase 5 MEC coordinator and the frozen Phase 4 **{proto["allocator"]}** allocator. Channel {proto["channel"]}, proactive switching, the {proto["interruption_tolerance_s"]}-second interruption timeout, the arrival policy and scenario replicates were fixed. Existing marketplace base seeds were {", ".join(map(str, base_seeds))} and the existing channel seed was {radio_seed}. Only the reliability gate was recomputed in memory from the recorded probabilities. The physical availability and radio/resource gates were unchanged. A working link can continue below θ, but a sustained low score can trigger a proactive switch. Candidate link counts are link-seconds. Eligible provider counts sum the distinct providers in each scenario. Active request-seconds come from the session ledger. Cost per GB pools Phase 5 payments and delivered traffic. This protocol measures the frozen marketplace method's threshold sensitivity.

The sweep verified the reproduced 0.62 and 0.65 candidate flags against the recorded Phase 3 flags. For both thresholds, all eight checked outcome fields matched the corresponding existing Phase 5 validation runs. Solver/ledger audits and independent event checks passed. Guard checks used the protected artifact manifest, Phase 4 dependency hashes and the Phase 2/3 validation input manifests. The analysis used validation scenario files and recorded validation threshold tables. The protected artifact guard also checks file hashes without parsing held-out test metrics. Results were written to a separate directory. The scenario table uses the common outcome and audit fields across scenarios. Optional coordinator diagnostics have varying keys, so their counts are excluded from the common table. The standard proactive switch and reallocation counts are included.

## Validation classification

{markdown_table(selected[cls_cols])}

False-positive rate is the fraction of truly non-surviving windows that pass. The fraction of approved windows that later fail equals one minus precision. These metrics have different denominators.

[Classification plots](figures/classification.png) · [All 41 thresholds and confidence bounds](classification.csv)

## Validation marketplace

{markdown_table(m[market_cols])}

{markdown_table(m[session_cols])}

[Marketplace plots](figures/marketplace.png) · [Scenario-level results](marketplace_runs.csv) · [Pooled marketplace metrics](marketplace_summary.csv) · [Metrics by mobility run](marketplace_by_mobility_run.csv)

## Sensitivity relative to 0.62

{markdown_table(pd.DataFrame(comparisons))}

At 0.60, precision falls to {r60.precision_raw:.2%} and recall increases to {r60.recall_raw:.2%}. At 0.67, precision increases to {r67.precision_raw:.2%} and recall falls to {r67.recall_raw:.2%}. Moving from 0.62 to 0.67 excludes {baseline.fp - r67.fp:,} additional false approvals and {baseline.tp - r67.tp:,} additional true survivals. Moving from 0.62 down to 0.60 adds {r60.fp - baseline.fp:,} false approvals and {r60.tp - baseline.tp:,} true survivals. Relative to 0.62, 0.60 exposed {lower.eligible_links - base.eligible_links:+,} eligible link-seconds, 0.65 exposed {stricter.eligible_links - base.eligible_links:+,}, and 0.67 exposed {high.eligible_links - base.eligible_links:+,}. Across these four settings the admission-rate spread was {span("admission_pct"):.3f} percentage points, the PUE traffic share spread was {span("pue_traffic_pct"):.3f} percentage points, the cost spread was {span("cost_per_gb"):.6f} price units/GB, and the range in interruption counts was {span("interruption_episodes"):,} episodes. These are descriptive sensitivity effects for this scenario set. No operational materiality margin was specified. Marketplace interruptions also depend on alternate suppliers, gNodeB resources, radio outages, mobility departures and the coordinator's switching policy. Compare normalized interruption counts alongside raw counts because the set of admitted requests may change.

### Existing allocator cross-check at 0.62 and 0.65

{markdown_table(anchors[["allocator", "threshold", "admission_pct", "pue_traffic_pct", "cost_per_gb",
                        "interruption_episodes", "dropped", "recoveries", "reallocations"]])}

These rows come from the recorded Phase 5 validation grid. The complete nine-threshold replay covers the configured allocator shown above. [Reference summary](reference_summary.csv)

## Limitations

The 90% requirement represents an operational target for approved, labeled validation link windows. Windows without a defined 30-second label were outside the classification sample. The session replay separately counts mobility departures and interruptions. It does not automatically guarantee reliable sessions or generalization to new mobility environments. Selection used the same four validation mobility runs as the threshold assessment. Temporally adjacent windows and pairs can share vehicles, and marketplace replicates reuse each of the four mobility traces. The small number of independent traces limits uncertainty claims. This study holds the radio setting and interruption policy fixed. A broader sweep across the other channel settings, additional independent mobility runs and an all-threshold MILP comparison could strengthen generalization. The test split remained untouched in this investigation.

## Viva answer

“We tried 41 cutoffs on validation data. We wanted at least 90% of approved links to survive while retaining at least half of the survivable links. At 0.60 we missed the precision target. At 0.67 we gained precision but discarded more usable links. 0.62 was the lowest tested cutoff that met both targets, and we tested 0.65 separately to check a stricter confidence-based setting.”

## Reproduction

From the repository root with the existing environment and validation inputs:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m threshold_investigation.run \
  threshold_investigation/reproduction --workers 8
MPLCONFIGDIR=/tmp/mec-matplotlib .venv/bin/python -m threshold_investigation.report \
  threshold_investigation/reproduction
```

[Protocol and input hashes](protocol.json) · [Validation checks](validation_checks.json)
"""
    (dest / "report.md").write_text(report)
    print(m[market_cols].to_string(index=False))


if __name__ == "__main__":
    main()
