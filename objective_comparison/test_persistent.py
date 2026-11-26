import hashlib
from dataclasses import replace

import numpy as np
import pytest

from objective_comparison.allocation import choose
from objective_comparison.persistent import PolicyState, initial, pending_snapshot
from objective_comparison.run import read_json
from objective_comparison.test_objectives import example

CFG = read_json("objective_comparison/config.json")
METHODS = CFG["algorithms"]


@pytest.mark.parametrize("method", METHODS)
def test_initial_policy_parity(method):
    snap = example()
    sub, _ = pending_snapshot(snap, snap.empty(), [0])
    plan, notes = initial(sub, method, CFG)
    if method == "Random":
        seed = hashlib.sha256(f"{CFG['seed']}:manual:10".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(seed[:8], "little"))
        trials = [snap.constructive(method, rng) for _ in range(16)]
        original = trials[choose([snap.evaluate(p) for p in trials], snap.epsilon)]
    elif method == "Exact":
        original, baseline = snap.exact(3., 0.)
        assert notes == baseline
    else:
        original = snap.constructive(method)
    for a, b in zip(original, plan):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    assert snap.evaluate(original) == sub.evaluate(plan)


@pytest.mark.parametrize("method", METHODS)
def test_keep_then_fallback_without_second_policy_call(monkeypatch, method):
    import objective_comparison.persistent as module

    calls = []
    original = module.initial

    def tracked(s, method, cfg):
        calls.append(s.problem.time_s)
        return original(s, method, cfg)

    monkeypatch.setattr(module, "initial", tracked)
    state = PolicyState(method, CFG)
    first = example()
    first.problem.gnb_price_per_gb = 1.5
    before = state.step(first)
    assert before["initial_pue"] == 1
    provider = state.active["r"].pue
    second = replace(first, problem=replace(first.problem, time_s=11))
    retained = state.step(second)
    assert retained["retained_pue"] == 1
    assert state.active["r"].pue == provider
    assert retained["mec_fee"] == before["mec_fee"]
    assert retained["cost"] == before["cost"]
    third = replace(first, problem=replace(first.problem, time_s=12, link_p_survive=np.array([.5])))
    after = state.step(third)
    assert after["failures"] == after["immediate_fallbacks"] == 1
    assert after["pue_gb"] == 0
    assert after["mec_fee"] == 0
    assert after["delivered_mbps"] == 8
    assert after["cost"] == pytest.approx(1.5 * 8 / 8000)
    assert after["rate"] == pytest.approx(8)
    fourth = replace(first, problem=replace(first.problem, time_s=13))
    recovered_radio = state.step(fourth)
    assert recovered_radio["pue_gb"] == 0
    assert state.active["r"].mode == "gnb"
    assert calls == [10]
    assert state.active["r"].requested_seconds == 4


def test_distinct_residual_bandwidth_rhs():
    s = example()
    relay, cell, rb, gb = s.empty()
    relay[0], cell[0], rb[0], gb[0] = 1., 2., .5, 5.
    sub, _ = pending_snapshot(s, (relay, cell, rb, gb), [0])
    assert sub.provider_cap[0] == 5
    assert sub.problem.cell_capacity_mbps[0] == 8
    assert sub.pc5_remaining[0] == 4.5
    assert sub.gnb_remaining[0] == 3
    assert s.provider_cap[0] == 6
    assert s.gnb_mhz == 8


def test_denied_fallback_retries_only_gnb():
    cfg = dict(CFG)
    state = PolicyState("Greedy", cfg)
    s = example()
    assert state.step(s)["initial_pue"] == 1
    outage = replace(s, problem=replace(s.problem, time_s=11, cell_capacity_mbps=np.array([1.]),
                                         link_p_survive=np.array([.5])))
    result = state.step(outage)
    assert result["failures"] == 1
    assert result["immediate_fallbacks"] == 0
    assert result["unserved"] == 1
    assert state.active["r"].mode == "gnb"
    returned = state.step(replace(s, problem=replace(s.problem, time_s=12)))
    assert returned["arrivals"] == 0
    assert returned["gnb_gb"] == pytest.approx(.001)


def test_shared_real_snapshot_parity(tmp_path, monkeypatch):
    import pandas as pd
    import objective_comparison.run as old
    import objective_comparison.persistent_run as new

    cfg = dict(CFG, split="val")
    phase = old.read_json(cfg["phase3_config"])
    rows = pd.read_csv(f"{phase['output_root']}/val/phase3_manifest.csv")
    rows = rows[rows.scenario_id.eq("C00_baseline__run_000__r1")]
    run_id = rows.iloc[0].run_id
    originals = []
    persistent = []

    def capture(module, target):
        constructor = module.Snapshot

        def wrapped(*args):
            snapshot = constructor(*args)
            target.append(snapshot)
            return snapshot

        monkeypatch.setattr(module, "Snapshot", wrapped)

    capture(old, originals)
    capture(new, persistent)
    for name, runner in (("old", old), ("new", new)):
        path = tmp_path / name
        path.mkdir()
        runner.run_group((cfg, phase, run_id, rows, path, 70))
    assert len(originals) == len(persistent) == 70
    assert any(s.problem.n_links for s in originals)
    for a, b in zip(originals, persistent):
        assert a.problem.request_ids == b.problem.request_ids
        for key in ("link_p_survive", "link_request", "link_pue", "link_capacity_mbps",
                    "cell_index", "covered", "access_mbps", "backhaul_mbps", "cell_capacity_mbps"):
            np.testing.assert_array_equal(getattr(a.problem, key), getattr(b.problem, key))
        np.testing.assert_array_equal(a.link_se, b.link_se)
        np.testing.assert_array_equal(a.cell_se, b.cell_se)


def test_competing_arrival_cannot_use_retained_bandwidth():
    from objective_comparison.persistent import PersistentScenario

    cfg = dict(CFG, algorithms=["Greedy", "Best-SINR"])
    scenario = PersistentScenario(cfg)
    s = example()
    s.problem.gnb_price_per_gb = 1.5
    scenario.step(s)
    left, right = scenario.states
    assert left.active["r"] is not right.active["r"]
    p = s.problem
    p2 = replace(p, time_s=11, request_ids=["r", "new"], cue_ids=np.array([1, 2]),
                 demand_mbps=np.array([8., 1.]), weight=np.ones(2), max_price_per_gb=np.full(2, 2.),
                 arrival_s=np.array([0., 11.]), covered=np.ones(2, dtype=bool),
                 cell_index=np.zeros(2, dtype=int), link_request=np.array([0, 1]),
                 link_pue=np.zeros(2, dtype=int), link_capacity_mbps=np.full(2, 6.),
                 link_p_survive=np.full(2, .85))
    t = replace(s, problem=p2, link_se=np.full(2, 2.), cell_se=np.ones(2))
    for _, metrics in scenario.step(t):
        assert metrics["retained_pue"] == 1
        assert metrics["arrivals"] == 1
        assert metrics["admitted"] == 0
        assert metrics["unserved"] == 1
    ending = replace(s, problem=replace(p, time_s=12, request_ids=["fresh"]))
    for _, metrics in scenario.step(ending):
        assert metrics["admitted"] == 1
