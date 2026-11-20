import medgemma_grounding.analysis.attention_layers as routing_analysis
import medgemma_grounding.experiments.run_attention_layers as run_attention_layers
import medgemma_grounding.interventions.attention_runtime as attention_runtime
import numpy as np
import pytest


def row(patient, outcome, score, delta, condition="B_00_00", case=None):
    label = int(outcome in ("TP", "FN"))
    correct = outcome.startswith("T")
    changed_correct = (score + delta > 0) == bool(label)
    return {"case_id": case or patient, "patient": patient, "outcome": outcome, "label": label,
            "id": condition, "S": score, "changed_S": score + delta, "delta_S": delta,
            "delta_M": (2 * label - 1) * delta, "kl_from_baseline": 0.,
            "corrected": changed_correct and not correct, "introduced": correct and not changed_correct,
            "answer_format_valid": True, "start": 0, "end": 0, "route": "B"}


def cohort():
    rows = []
    for j, outcome in enumerate(routing_analysis.OUTCOMES):
        for i in range(10):
            s = (i + 1) * (1 if outcome.endswith("P") else -1)
            rows.append(row(f"{outcome}{i:02d}", outcome, s, .2*s + .3))
    return rows


def test_patient_weighting_and_fixed_groups():
    rows = [row("a", "FP", 2, -3, case="a1"), row("a", "FP", 4, -2, case="a2"), row("b", "TN", -1, 0)]
    patients = routing_analysis.patient_rows(rows)
    assert len(patients) == 2 and patients[0]["delta_S"] == -2.5
    assert patients[0]["outcome"] == "FP"  # not reclassified after removal
    with pytest.raises(ValueError, match="Ambiguous"):
        routing_analysis.patient_rows([*rows, row("a", "TN", -1, 0, case="a3")])


def test_direct_contrast_ci_and_degenerate_welch():
    result = routing_analysis.contrast([2, 2, 2], [0, 0, 0])
    assert result["effect"] == 2 and result["ci95"] == [2, 2]
    assert result["p"] == 0 and result["welch_t"] is None
    assert routing_analysis.contrast([0, 0], [0, 0])["p"] == 1
    result = routing_analysis.contrast([1, 2, 3], [-1, 0, 1])
    assert result["effect"] == 2 and result["ci95"][0] < 2 < result["ci95"][1]
    np.testing.assert_allclose(routing_analysis.bh_adjust([.01, .04, .03, .2]), [.04, .053333333333, .053333333333, .2])


def test_folds_patient_disjoint_stratified_deterministic():
    patients = routing_analysis.patient_rows(cohort())
    first = routing_analysis.fixed_folds(patients)
    assert first == routing_analysis.fixed_folds(list(reversed(patients)))
    assert len(first) == 40
    for fold in range(5):
        assert sum(first[r["patient"]] == fold and r["label"] == 1 for r in patients) == 4
        assert sum(first[r["patient"]] == fold and r["label"] == 0 for r in patients) == 4


def test_affine_and_shift_controls_no_outcome_fits():
    patients = routing_analysis.patient_rows(cohort())
    pred, fits, summaries = routing_analysis.score_controls(patients, routing_analysis.fixed_folds(patients))
    assert len(pred) == 80 and len(fits) == 10
    assert next(r for r in summaries if r["control"] == "affine")["rmse"] < 1e-12
    assert all(r["training_patients"] == 32 for r in fits)
    assert all("outcome" not in r for r in fits)
    assert routing_analysis.fit_score_control([1, 1], [2, 3], "affine")["status"] == "constant_x_shift_fallback"
    fitted = routing_analysis.fit_score_control([1, 2], [3, 2], "affine")
    assert fitted["slope"] == 1e-8 and fitted["status"] == "positive_slope_clamped"
    fitted = routing_analysis.fit_score_control([1, 2], [3, 4], "shift")
    assert fitted["intercept"] == 2 and fitted["slope"] == 1


def test_held_out_targets_cannot_change_their_fold_fit():
    patients = routing_analysis.patient_rows(cohort())
    folds = routing_analysis.fixed_folds(patients)
    _, fits1, _ = routing_analysis.score_controls(patients, folds)
    changed = [{**r, "changed_S": r["changed_S"] + 100} if folds[r["patient"]] == 0 else r for r in patients]
    _, fits2, _ = routing_analysis.score_controls(changed, folds)
    assert [r for r in fits1 if r["fold"] == 0] == [r for r in fits2 if r["fold"] == 0]


def test_tie_rule_threshold_envelope_and_ranking():
    rows = [row("a", "TN", 0, 0), row("b", "TP", 3, 0), row("c", "FP", 2, 0), row("d", "FN", -1, 0)]
    assert routing_analysis.classification(rows, "S")["TN"] == 1
    assert routing_analysis.classification(rows, "S", 2)["FP"] == 0
    assert routing_analysis.classification(rows, "S", 2)["TP"] == 1
    assert routing_analysis.ranking(rows, "S") == .5
    assert routing_analysis.roc_thresholds(rows)[0]["FN"] == 0


def test_all_effects_including_zero_and_loo(tmp_path):
    rows = cohort()
    rows = [{**r, "delta_S": 0., "changed_S": r["S"], "delta_M": 0.} for r in rows]
    result = routing_analysis.analyze(rows, tmp_path, routing_analysis.fixed_folds(routing_analysis.patient_rows(rows)))
    assert all(r["effect"] == 0 for r in result["contrasts"])
    assert len((tmp_path / "leave_one_patient_out.csv").read_text().splitlines()) == 21
    assert len((tmp_path / "score_predictions.csv").read_text().splitlines()) == 81
    with pytest.raises(ValueError, match="all 34"):
        routing_analysis.analyze(rows, tmp_path, routing_analysis.fixed_folds(routing_analysis.patient_rows(rows)), primary_family=True)


def test_joint_comparison_orientation_and_compatibility():
    single = [{**row("p", "FP", 2, 1, f"B_{i:02d}_{i:02d}"), "start": i, "end": i} for i in range(6)]
    joint = [{**row("p", "FP", 2, 4, "B_00_05"), "start": 0, "end": 5}]
    result = routing_analysis.joint_comparison(single, joint)
    assert result[0]["sum_single_delta_S"] == 6 and result[0]["joint_minus_sum"] == -2
    single[0]["S"] += 1
    with pytest.raises(ValueError, match="Unmatched"):
        routing_analysis.joint_comparison(single, joint)


def test_full_coverage_and_checksum_resume_guards(tmp_path):
    assert len(run_attention_layers.conditions()) == 34 and len(run_attention_layers.conditions())*142 == 4828
    p = {"protocol_hash": "x", "case_ids": ["c"], "conditions": run_attention_layers.conditions()}
    directory = tmp_path / "cases/c"
    directory.mkdir(parents=True)
    (directory / "baseline.pt").write_bytes(b"bytes")
    attention_runtime.write_record(directory / "baseline.json", {"protocol_hash": "x", "case_id": "c", "tensor_sha256": run_attention_layers.file_sha256(directory / "baseline.pt")})
    assert not run_attention_layers.complete_case(tmp_path, p, "c")
    (directory / "baseline.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        run_attention_layers.complete_case(tmp_path, p, "c")


def test_completed_resume_never_loads_weights(tmp_path, monkeypatch):
    p = {"case_ids": ["a", "b"]}
    monkeypatch.setattr(run_attention_layers, "prepare", lambda *args: p)
    monkeypatch.setattr(run_attention_layers, "validate", lambda *args: {})
    monkeypatch.setattr(run_attention_layers, "complete_case", lambda *args: True)
    monkeypatch.setattr(attention_runtime, "runtime", lambda *args: pytest.fail("Model loading"))
    monkeypatch.setattr(run_attention_layers, "report", lambda *args: {"complete": True})
    assert run_attention_layers.run("run", tmp_path)["complete"]


def test_interrupted_legacy_benchmark_cannot_approve_new_cohort():
    with pytest.raises(ValueError, match="Incomplete E-010"):
        run_attention_layers.validate_benchmark({"backend_approved": True, "forecast_forwards_upper": 2982,
                                  "controls_passed": True}, {"expected_cohort_forwards": 5112})
    completed = {"forecast_forwards_upper": 5112, "single_layer_probability_checks": [0,8,13,17,33],
                 "historical_checks": [{}, {}, {}, {}], "controls_passed": True,
                 "full_depth_B_image_independence": True}
    run_attention_layers.validate_benchmark(completed, {"expected_cohort_forwards": 5112})


def test_diagnostic_audit_excludes_core_benchmark_summary(tmp_path):
    core = tmp_path / "engineering/core/engineering/case"
    core.mkdir(parents=True)
    (core.parent.parent / "benchmark.json").write_text('{}')
    (core / "B_00_33.json").write_text('{}')
    target = tmp_path / "engineering/case"
    target.mkdir()
    (target / "B_08_08.json").write_text('{}')
    assert [p.name for p in run_attention_layers.diagnostic_paths(tmp_path)] == ["B_08_08.json", "B_00_33.json"]


def test_metadata_only_known_cases(tmp_path):
    path = tmp_path / "CheXpert"
    path.mkdir()
    (path / "val_labels.csv").write_text("Path,Sex,Age,Frontal/Lateral,AP/PA,Pleural Effusion,Edema\n"
        "old/valid/p/study1/view1.jpg,Female,60,Frontal,AP,0,1\n"
        "old/valid/reserved/study1/view1.jpg,Male,50,Frontal,PA,1,0\n")
    manifest = {"cases": [{"case_id": "p_study1_view1", "patient": "p", "label": 0, "image_size": [100, 200], "annotation_status": "missing"}]}
    records, _ = run_attention_layers.metadata(manifest, tmp_path)
    assert len(records) == 1 and records[0]["patient"] == "p"
    assert records[0]["Edema"] == 1


def test_full_saved_record_audit_schema_without_model(tmp_path, monkeypatch):
    import torch
    p = {"protocol_hash": "p", "case_ids": [f"c{i:03d}" for i in range(142)],
         "conditions": run_attention_layers.conditions(), "sliding_window": 1024, "device": "mps",
         "scoring": {"ids": {"Yes": 0, "No": 1}}}
    groups = {"image": [1,2], "post_image": [3,4,5], "final": [5], "pre_image": [0], "valid": list(range(6))}
    manifest = {"cases": [{"case_id": cid, "patient": f"p{i%140}", "label": 0,
                           "image_sha256": "image", "historical_outcome": "TN",
                           "historical_answer_score": {"score": -2.}} for i, cid in enumerate(p["case_ids"])]}
    original = torch.tensor([[[-1., 1.]]])
    for case in manifest["cases"]:
        directory = tmp_path / "cases" / case["case_id"]
        directory.mkdir(parents=True)
        torch.save(original, directory / "baseline.pt")
        attention_runtime.write_record(directory / "baseline.json", {"protocol_hash": "p", "case_id": case["case_id"],
            "tensor_sha256": run_attention_layers.file_sha256(directory / "baseline.pt"), "input_sha256": "input"})
        for c in p["conditions"]:
            r = attention_runtime.measurement(case, p, original, original, c, {"seconds": .1, "blocked_edges": 6,
                "layers": {str(c["start"]): {"non_target_unchanged": True, "no_fully_blocked_rows": True}}}, "input")
            attention_runtime.write_record(directory / f"{c['id']}.json", {**r, "token_groups": groups})
    attention_runtime.write_record(tmp_path / "runtime_checks.json", {"protocol_hash": "p",
        "frozen_unchanged_no_gradients": True, "no_hooks_remain": True})
    run_attention_layers.atomic_json(tmp_path / "report.json", {"artifacts_sha256": {}})
    core = tmp_path / "engineering/core"
    core.mkdir(parents=True)
    attention_runtime.write_record(core / "benchmark.json", {"protocol_hash": "p", "controls_passed": True})
    attention_runtime.write_record(core / "B_00_33.json", {"protocol_hash": "p", "layers": {"0": {"target_max_probability": 0}}})
    monkeypatch.setattr(run_attention_layers, "validate", lambda *args: manifest)
    monkeypatch.setattr(run_attention_layers, "run", lambda *args: {"complete": True})
    monkeypatch.setattr(attention_runtime, "runtime", lambda *args: pytest.fail("Model loading"))
    result = run_attention_layers.verify(tmp_path, p, manifest)
    assert result["knockouts"] == 4828 and result["model_free_byte_identical_resume"]
