#!/usr/bin/env python
"""
Phase 5 MEC session management runner.

  demo                      narrated end-to-end lifecycle on a SYNTHETIC scenario (tests/mec_fixtures.py):
                            request -> discovery -> LSTM/wireless checks -> Phase 4 allocation -> session ->
                            PUE departure -> detection -> replacement PUE -> link outage -> gNodeB fallback ->
                            accounting / audit checks. Writes nothing.
  run --scenario SID        one REAL validation scenario through the MEC control loop; write-once outputs in
                            <output_root>/runs/<label>/<SID>/ (ledger, sessions, events, audit, timeline, report)

The interruption tolerance is still an open decision (selected on validation, Revision 2 section 9), so
`run` requires --max-interruption-s explicitly and marks the run as provisional. The test split is refused:
Phase 5 is not frozen. Phase 4 dependency pins are verified before anything runs.
"""
import argparse
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.marketplace.greedy import GreedyAllocator  # noqa: E402
from src.marketplace.milp import MilpAllocator  # noqa: E402
from src.mec.config import Phase5Config  # noqa: E402
from src.mec.coordinator import MecCoordinator  # noqa: E402
from src.mec.inputs import Phase5ScenarioInputs  # noqa: E402
from src.mec.state import StateManager  # noqa: E402
from src.utils.integrity import verify_manifest  # noqa: E402

CONFIG = "config/mec/phase5.json"


def allocator(name, cfg4):
    m = cfg4["milp"]
    return MilpAllocator(cfg4["rate_eps_mbps"], m["mip_rel_gap"], m["time_limit_s"], m["presolve"]) \
        if name == "milp" else GreedyAllocator(cfg4["rate_eps_mbps"])


def guard(cfg):
    ch, mi = verify_manifest("outputs/PROTECTED_ARTIFACTS.sha256")
    if ch or mi:
        raise SystemExit(f"ABORT: protected artifacts changed={ch} missing={mi}")
    return cfg.verify_phase4_dependency()


def cmd_demo(cfg):
    from tests import mec_fixtures as fx
    print("SYNTHETIC demonstration scenario (tests/mec_fixtures.py); channel G, aware theta=0.62, Phase 4 MILP\n")
    rng = lambda r, p, t: (r == "R1" and (p == 2 or (p == 4 and t >= 11))) or (r == "R2" and p == 1)
    d = tempfile.mkdtemp()
    p3, p2 = fx.write(d, range_fn=rng, outage=[("R2", 1, 15)])
    inp = Phase5ScenarioInputs("FIX__fixture_run__r0", p3, p2)
    st = StateManager(inp, fx.mobility())
    c = MecCoordinator(inp, st, "G", "primary", MilpAllocator(), max_interruption_s=5).run()
    res = c.results()
    L, ev = res["ledger"], res["events"]
    cl = inp.cl[(inp.cl.channel_scenario == "G")]

    def links(rid, t):
        x = cl[(cl.request_id == rid) & (cl.time_s == t)]
        return "\n".join(f"      PUE {int(r.pue_id)}: status={r.provider_status}, p_survive={r.p_survive:.2f} "
                         f"(reliable@0.62={bool(r.reliable_primary)}), realized C3={r.realized_link_capacity_mbps:.1f} Mbps, "
                         f"wireless_ok={bool(r.wireless_ok)} -> policy-eligible={bool(r.feasible_primary)}"
                         for r in x.itertuples()) or "      (no provider in range)"

    def row(rid, t):
        r = L[(L.request_id == rid) & (L.time_s == t)].iloc[0]
        pue = "none" if r.pue_id != r.pue_id or r.pue_id is None else int(r.pue_id)
        return (f"PUE={pue} r={r.pue_rate_mbps:.2f} Mbps, gNodeB={r.gnb_rate_mbps:.2f} Mbps, "
                f"payment={r.payment:.6f} (PUE {r.pue_revenue:.6f} + gNodeB {r.gnb_revenue:.6f})")

    w = st.windows["R1"]
    print(f"1  CUE {w.cue_id} submits request R1 at t={w.start_s:.0f}s (5 Mbps, window to t={w.nominal_end_s:.0f}s)")
    print(f"2  MEC discovers in-range PUEs at t=2:\n{links('R1', 2.0)}")
    print("3  LSTM reliability (Phase 3 p_survive vs theta) and REALIZED wireless capacity checked (above)")
    print(f"4  Phase 4 MILP selects resources:  t=2  {row('R1', 2.0)}")
    print(f"5  Session R1 ACTIVE; ledger row recorded each second (t=3: {row('R1', 3.0)})")
    e = ev[(ev.request_id == "R1") & (ev.time_s == 11.0)]
    print(f"6  PUE 2 leaves the SUMO trace after t=10")
    print(f"7  MEC detects at t=11: {', '.join(e.event)}")
    print(f"   candidate links at t=11:\n{links('R1', 11.0)}")
    print(f"8  Replacement PUE found in the same second:  t=11  {row('R1', 11.0)}")
    print(f"9  Service resumed without interruption (interruptions={c.sm.sessions['R1'].n_interruptions}, "
          f"reallocations={c.sm.sessions['R1'].n_reallocations}); R1 ends: {c.sm.sessions['R1'].state.value}")
    e2 = ev[(ev.request_id == "R2") & (ev.time_s == 15.0)]
    print(f"   Link failure case: R2 on PUE 1, t=15 event {', '.join(e2.event)} -> gNodeB fallback: t=15 {row('R2', 15.0)}")
    rep = c.report()
    print(f"10 Accounting / audit: ledger checks {rep['ledger_check']}; per-second Phase 4 audit violations "
          f"= {rep['audit_violations']} over {rep['audit_seconds']} s; total payment {rep['payment_total']:.6f}")
    print("\nSession timeline R1:")
    for t, a, dtl in c.sm.sessions["R1"].timeline:
        print(f"   t={t:>4.0f}  {a:<12} {dtl}")
    ok = rep["audit_violations"] == 0 and not any(rep["ledger_check"].values()) and not rep["errors"]
    print("\nDEMO", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cmd_run(cfg, a):
    if a.split != "val":
        raise SystemExit("ABORT: Phase 5 is not frozen; only the validation split may be run")
    cfg4 = json.load(open(cfg["inputs"]["phase4_config"]))
    pol = cfg.policy(a.policy)
    if a.channel not in cfg.channel_scenarios:
        raise SystemExit(f"unknown channel {a.channel}")
    from src.mobility.trace import load_run
    sid = a.scenario
    inp = Phase5ScenarioInputs(sid, cfg.phase3_dir(a.split, sid), cfg.phase2_dir(a.split, sid), cfg["dt_s"])
    st = StateManager(inp, load_run(sid.split("__")[1], cfg["inputs"]["raw_mobility_dir"]))
    c = MecCoordinator(inp, st, a.channel, pol.theta_tag, allocator(a.allocator, cfg4), a.max_interruption_s,
                       cfg["rate_eps_mbps"], cfg["audit_tolerance_mbps"], cfg["continuity_weight"]).run()
    label = f"{a.split}_{a.channel}_{a.policy}_{a.allocator}_mi{a.max_interruption_s}"
    out = os.path.join(cfg["output_root"], "runs", label, sid)
    if os.path.exists(out):
        raise SystemExit(f"ABORT: {out} exists (write-once)")
    os.makedirs(out)
    for name, df in c.results().items():
        df.to_csv(os.path.join(out, f"{name}.csv.gz"), index=False, compression={"method": "gzip", "mtime": 0})
    rep = c.report() | {"provisional": True, "note": "max_interruption_s not yet selected on validation",
                        "policy": a.policy, "split": a.split}
    with open(os.path.join(out, "report.json"), "x") as f:
        json.dump(rep, f, indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))
    print(f"outputs: {out}")
    ok = rep["audit_violations"] == 0 and not any(rep["ledger_check"].values()) and not rep["errors"]
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["demo", "run"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--scenario")
    ap.add_argument("--channel", default="G")
    ap.add_argument("--policy", default="aware_primary")
    ap.add_argument("--allocator", choices=["milp", "greedy"], default="milp")
    ap.add_argument("--max-interruption-s", type=int)
    a = ap.parse_args()
    cfg = Phase5Config.load(CONFIG)
    print("Phase 4 dependency verified:", guard(cfg))
    if a.cmd == "demo":
        sys.exit(cmd_demo(cfg))
    if not a.scenario or a.max_interruption_s is None:
        ap.error("run requires --scenario and --max-interruption-s")
    sys.exit(cmd_run(cfg, a))


if __name__ == "__main__":
    main()
