"""Discovery protocol checks without model weights or confirmation inference."""

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask
import pytest
import torch
from scipy.stats import mannwhitneyu, spearmanr
from streamlit.testing.v1 import AppTest

from medgemma_grounding.chexlocalize import expert_mask, validation_cohort, validate_partition
from medgemma_grounding.feature_analysis import (
    aggregate_patients, feature_rankings, select_candidates, pooled_fidelity, audit_masks, run,
)
from medgemma_grounding.medgemma_features import (
    atomic_json, identity_hash, file_sha256, answer_candidates, answer_score, load_discovery_case,
)
from medgemma_grounding.sae_reconstruction import position_groups, masked_replacement, prompt_positions


def encode_mask(binary):
    encoded = coco_mask.encode(np.asfortranarray(binary.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode()
    return encoded


def test_original_coordinate_masks_and_statuses():
    case = {"case_id": "case", "image_size": [7, 13]}
    polygon = [[8, 1], [11, 1], [11, 3], [8, 3]]
    original = Image.new("1", (13, 7))
    ImageDraw.Draw(original).polygon([tuple(p) for p in polygon], fill=1, outline=1)
    binary = np.asarray(original)
    raw = {"case": {"img_size": [7, 13], "Pleural Effusion": [polygon]}}
    released = {"case": {"Pleural Effusion": encode_mask(binary)}}
    status, decoded, _ = expert_mask(case, raw, released)
    assert status == "present nonempty"
    assert decoded.shape == (7, 13) and decoded[2, 10] and not decoded[5, 2]
    assert expert_mask(case, {}, {})[0] == "missing"
    empty = {"case": {"img_size": [7, 13]}}
    zero = {"case": {"Pleural Effusion": encode_mask(np.zeros((7, 13)))}}
    assert expert_mask(case, empty, zero)[0] == "present empty"
    assert expert_mask(case, raw, zero)[0] == "invalid/mismatched"
    assert expert_mask(case, raw, {})[0] == "invalid/mismatched"
    assert expert_mask({**case, "image_size": [13, 7]}, raw, released)[0] == "invalid/mismatched"


def test_deterministic_patient_split_pilot_and_exclusions(tmp_path):
    root = tmp_path
    (root / "CheXpert").mkdir()
    (root / "CheXlocalize").mkdir()
    rows = []
    for i in range(8):
        path = root / "CheXpert" / "val" / f"patient{i}" / "study1" / "view1_frontal.jpg"
        path.parent.mkdir(parents=True)
        Image.new("RGB", (13, 7), (i * 25, 0, 0)).save(path)
        rows.append({"Path": f"CheXpert-v1.0/valid/patient{i}/study1/view1_frontal.jpg",
                     "Pleural Effusion": str(1 - i % 2), "Frontal/Lateral": "Frontal"})
    rows.extend([{"Path": "uncertain", "Pleural Effusion": "-1", "Frontal/Lateral": "Frontal"},
                 {"Path": "lateral", "Pleural Effusion": "1", "Frontal/Lateral": "Lateral"}])
    label_path = root / "CheXpert" / "val_labels.csv"
    def write_rows(data):
        with label_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(data)
    write_rows(rows)
    atomic_json(root / "CheXlocalize" / "gt_annotations_val.json", {})
    atomic_json(root / "CheXlocalize" / "gt_segmentations_val.json", {})
    kwargs = {"expected": (8, 8, 4), "targets": (3, 2)}
    cohort = validation_cohort(root, {"patient0", "patient1"}, **kwargs)
    selected = validate_partition(cohort, {"patient0", "patient1"}, **kwargs)
    assert len(selected) == 5 and {"patient0", "patient1"} <= {c["patient"] for c in selected}
    assert len(cohort["exclusions"]) == 2 and all(c["annotation_status"] == "missing" for c in cohort["cases"])
    write_rows(list(reversed(rows)))
    reordered = validation_cohort(root, {"patient0", "patient1"}, **kwargs)
    assert {c["patient"]: c["partition"] for c in cohort["cases"]} == {c["patient"]: c["partition"] for c in reordered["cases"]}
    broken = copy.deepcopy(cohort)
    broken["cases"][0]["partition"] = "confirmation"
    with pytest.raises(ValueError, match="Partition"):
        validate_partition(broken, {"patient0", "patient1"}, **kwargs)
    broken = copy.deepcopy(cohort)
    a = next(c for c in broken["cases"] if c["partition"] == "discovery")
    b = next(c for c in broken["cases"] if c["partition"] == "confirmation")
    b["image_sha256"] = a["image_sha256"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_partition(broken, {"patient0", "patient1"}, **kwargs)
    with pytest.raises(ValueError, match="counts"):
        validation_cohort(root, {"patient0"})


def test_position_masks_padding_and_selected_changes():
    ids = torch.tensor([[0, 99, 99, 3, 8]])
    groups = position_groups(ids, torch.tensor([[0, 1, 1, 1, 1]]), 99, [0, 8])
    assert groups["image"].tolist() == [False, True, True, False, False]
    assert groups["text"].tolist() == [False, False, False, True, False]
    assert groups["non_image"].tolist() == [False, False, False, True, True]
    assert groups["final"].tolist() == [False, False, False, False, True]
    original = torch.arange(10).reshape(1, 5, 2).float()
    replaced = masked_replacement(original, original + 100, groups["image"])
    torch.testing.assert_close(replaced[:, ~groups["image"]], original[:, ~groups["image"]])
    torch.testing.assert_close(replaced[:, groups["image"]], original[:, groups["image"]] + 100)
    torch.testing.assert_close(masked_replacement(original, original, groups["all"]), original)
    with pytest.raises(ValueError, match="padding"):
        position_groups(ids, torch.tensor([[1, 1, 1, 1, 0]]), 99, [])
    assert prompt_positions(ids, [3]).tolist() == [False, False, False, True, False]
    with pytest.raises(ValueError, match="one exact"):
        prompt_positions(ids, [99])


def test_boundary_candidates_and_exact_score():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return { (1, 2): "model\n", (3,): "Yes", (4,): "No"}[tuple(ids)]
        def encode(self, text, **kwargs):
            return {"model\n": [1, 2], "model\nYes": [1, 2, 3], "model\nNo": [1, 2, 4]}[text]
    candidates = answer_candidates(Tokenizer(), torch.tensor([[1, 2]]))
    logits = torch.tensor([[[0., 0., 0., 3., 1.]]])
    scored = answer_score(logits, candidates)
    assert scored["score"] == 2 and scored["decision"] == "Yes"
    assert scored["candidate_probability_mass"] < 1
    assert scored["yes_probability"] == pytest.approx(logits.double().softmax(-1)[0, 0, 3])
    tied = answer_score(torch.zeros_like(logits), candidates)
    assert tied["tie"] and tied["decision"] == "No"
    bad = Tokenizer()
    bad.encode = lambda *a, **k: [1, 2, 5, 6]
    with pytest.raises(ValueError):
        answer_candidates(bad, torch.tensor([[1, 2]]))


def test_patient_aggregation_ranking_direction_and_rare_exclusion():
    records = [{"patient": f"p{i}", "case_id": str(i), "label": int(i >= 6),
                "answer_score": {"score": float(i)}} for i in range(12)]
    x = np.column_stack([np.arange(12), 11 - np.arange(12), np.ones(12), [1] + [0] * 11])
    patients, arrays = aggregate_patients(records + [records[-1]], {k: np.vstack([x, x[-1]]) for k in ("image_mean", "final")})
    assert len(patients) == 12 and next(p for p in patients if p["patient"] == "p11")["images"] == 2
    rows = feature_rankings(patients, arrays)
    by_id = {r["feature"]: r for r in rows if r["representation"] == "image_mean" and r["target"] == "label"}
    assert by_id[0]["label_auc"] == 1 and by_id[1]["label_auc"] == 0
    assert by_id[1]["direction"] == "negative"
    assert not by_id[2]["eligible"] and not by_id[3]["eligible"]
    assert by_id[0]["answer_spearman"] == pytest.approx(spearmanr(arrays["image_mean"][:, 0], [p["answer_score"] for p in patients]).statistic)
    shortlist = select_candidates(rows)
    assert [r["feature"] for r in shortlist] == [0, 1]
    assert all(r["selection_track"] == "image_mean:label" for r in shortlist)
    mixed = copy.deepcopy(records[-1]); mixed["label"] = 0
    metadata, _ = aggregate_patients(records + [mixed], {k: np.vstack([x, x[-1]]) for k in ("image_mean", "final")})
    assert next(p for p in metadata if p["patient"] == "p11")["mixed_label"]


def test_pooled_final_fvu_is_centered_across_cases():
    xs = [torch.tensor([[1., 2.]]), torch.tensor([[3., 4.]])]
    records, sufficient = [], []
    for x in xs:
        records.append({"fidelity": {"final": {"tokens": 1, "sse": 2., "energy": x.square().sum().item(), "active_features": 3}}})
        sufficient.append({"final": {"sum_x": x.sum(0), "sum_x2": x.square().sum(0)}})
    result = pooled_fidelity(records, sufficient)["final"]
    assert result["centered_sst"] == 4 and result["sse"] == 4
    assert result["fvu"] == 1 and result["observed_l0"] == 3


def test_tie_heavy_rankings_against_independent_statistics_and_full_selection():
    rng = np.random.default_rng(42)
    x = rng.integers(0, 5, (140, 40)).astype(float)
    x[:, 0] = 1
    x[:, 1] = 0
    x[:4, 1] = 1
    labels = np.array([1] * 45 + [0] * 95)
    scores = rng.normal(size=140)
    patients = [{"label": int(label), "answer_score": float(score)} for label, score in zip(labels, scores)]
    rows = feature_rankings(patients, {"image_mean": x, "final": x})
    for row in rows:
        i = row["feature"]
        expected_auc = mannwhitneyu(x[labels == 1, i], x[labels == 0, i]).statistic / (45 * 95)
        assert row["label_auc"] == pytest.approx(expected_auc, abs=1e-12)
        if np.ptp(x[:, i]):
            assert row["answer_spearman"] == pytest.approx(spearmanr(scores, x[:, i]).statistic, abs=1e-12)
    selected = select_candidates(rows)
    assert len(selected) == len({r["feature"] for r in selected}) == 8
    assert all(r["feature"] not in (0, 1) for r in selected)


@pytest.fixture
def discovery_cache(tmp_path):
    directory = tmp_path / "experiment"
    root = tmp_path / "data"
    annotation_dir = root / "CheXlocalize"
    annotation_dir.mkdir(parents=True)
    atomic_json(annotation_dir / "gt_annotations_val.json", {})
    atomic_json(annotation_dir / "gt_segmentations_val.json", {})
    cases = []
    for i in range(12):
        path = root / f"case{i}.png"
        Image.new("RGB", (13, 7), (i * 20, 10, 30)).save(path)
        cases.append({"case_id": f"case{i}", "patient": f"patient{i}", "label": int(i >= 6),
                      "image_path": str(path), "image_sha256": file_sha256(path),
                      "image_size": [7, 13], "partition": "discovery"})
    cohort = {"data_root": str(root), "cases": cases, "metadata_sha256": {
        str(p): file_sha256(p) for p in annotation_dir.glob("*.json")}}
    atomic_json(directory / "cohort.json", cohort)
    protocol = {"schema_version": 2, "discovery_cases": cases, "sae_width": 4,
                "cohort_sha256": file_sha256(directory / "cohort.json"),
                "analysis": {"confirmation_enabled": False, "min_active_patients": 5}}
    protocol["protocol_hash"] = identity_hash(protocol)
    atomic_json(directory / "protocol.json", protocol)
    for i, case in enumerate(cases):
        features = torch.zeros(6, 4)
        features[:, int(i < 6)] = 1 + i / 10
        features[:, 2] = 1
        if i == 0:
            features[:, 3] = 2
        tokens = [{"index": j, "id": j, "piece": str(j), "text": str(j),
                   "group": "special" if j == 0 else "text" if j == 5 else "image"} for j in range(6)]
        positions = {"image": [1, 2, 3, 4], "text": [5], "non_image": [0, 5], "final": [5], "all": list(range(6))}
        fidelity, sufficient = {}, {}
        for group, indices in positions.items():
            x = torch.full((len(indices), 2), float(i + 1))
            fidelity[group] = {"tokens": len(indices), "sse": 0., "energy": x.square().sum().item(),
                               "relative_squared_error": 0., "active_features": int(features[indices].count_nonzero())}
            sufficient[group] = {"sum_x": x.sum(0), "sum_x2": x.square().sum(0)}
        record = {**case, "answer": "Yes" if i >= 6 else "No", "parsed_answer": "Yes" if i >= 6 else "No",
                  "answer_score": {"score": float(i - 5), "tie": i == 5}, "parsed_score_disagreement": False,
                  "feature_shape": [6, 4], "tokens": tokens, "positions": positions, "fidelity": fidelity,
                  "runtime_seconds": 1., "peak_rss_bytes": 100,
                  "spatial_layout": {"verified": True, "mapping": "direct_resize_row_major", "grid": [2, 2]},
                  "protocol_hash": protocol["protocol_hash"]}
        tensor_path = directory / "cases" / f"{case['case_id']}.pt"
        tensor_path.parent.mkdir(exist_ok=True)
        torch.save({"features": features.to_sparse().coalesce(), "sufficient_statistics": sufficient,
                    "summaries": {"image_mean": features[1:5].mean(0), "final": features[5],
                                  "image_max": features[1:5].amax(0), "image_active_fraction": (features[1:5] > 0).float().mean(0)}}, tensor_path)
        record["tensor_sha256"] = file_sha256(tensor_path)
        atomic_json(tensor_path.with_suffix(".json"), record)
    return directory


def test_shards_reject_confirmation_corruption_and_bad_protocol(discovery_cache):
    directory = discovery_cache
    record, _ = load_discovery_case(directory, "case0")
    with pytest.raises(ValueError, match="discovery"):
        load_discovery_case(directory, "confirmation_case")
    tensor = directory / "cases" / "case0.pt"
    tensor.write_bytes(tensor.read_bytes() + b"interrupted replacement")
    with pytest.raises(ValueError, match="checksum"):
        load_discovery_case(directory, "case0")
    protocol = json.loads((directory / "protocol.json").read_text())
    protocol["sae_width"] = 999
    with pytest.raises(ValueError, match="Protocol hash"):
        load_discovery_case(directory, "case1", protocol)


def test_offline_rankings_freeze_and_model_free_viewer(discovery_cache, monkeypatch):
    from medgemma_grounding import inference, sae_reconstruction, feature_viewer
    def forbid(*args, **kwargs):
        raise AssertionError("No model or SAE inference allowed")
    monkeypatch.setattr(inference, "load_model", forbid)
    monkeypatch.setattr(sae_reconstruction, "load_scope_sae", forbid)
    report = run(discovery_cache)
    original = (discovery_cache / "shortlist.json").read_bytes()
    run(discovery_cache)
    assert original == (discovery_cache / "shortlist.json").read_bytes()
    assert report["patients"] == 12 and report["mask_status_counts"] == {"missing": 12}
    app = AppTest.from_file(str(Path(feature_viewer.__file__))).run()
    app.sidebar.text_input[0].set_value(str(discovery_cache)).run()
    assert not app.exception and not app.error
    app.sidebar.number_input[0].set_value(3).run()
    app.selectbox(key="discovery_case_id").select("case11").run()
    assert not app.exception and not app.error
    assert any("Patient patient11" in item.value for item in app.markdown)
    app.sidebar.selectbox[0].select("final").run()
    assert not app.exception and not app.error
    assert any("no spatial overlay" in item.value for item in app.info)
    protocol = json.loads((discovery_cache / "protocol.json").read_text())
    protocol["analysis"]["confirmation_enabled"] = True
    atomic_json(discovery_cache / "protocol.json", protocol)
    with pytest.raises(ValueError, match="Confirmation"):
        run(discovery_cache)


def test_discovery_resume_only_runs_missing_cases_and_rejects_drift(discovery_cache, tmp_path, monkeypatch):
    from medgemma_grounding import inference, sae_reconstruction, chexlocalize, medgemma_features
    source_protocol = json.loads((discovery_cache / "protocol.json").read_text())
    source_case = source_protocol["discovery_cases"][0]
    source_record, source_payload = load_discovery_case(discovery_cache, source_case["case_id"])
    # Two miniature cases exercise the production resume path, with cohort audit
    # mocked separately (the exact production split is covered above and in QA).
    cohort = json.loads((discovery_cache / "cohort.json").read_text())
    cohort["cases"] = cohort["cases"][:2]
    cohort_path = tmp_path / "cohort.json"
    atomic_json(cohort_path, cohort)
    pilot_path = tmp_path / "pilot.json"
    atomic_json(pilot_path, {"model_id": "test-model", "model_revision": "pinned", "prompt": "question",
                             "cases": cohort["cases"]})
    monkeypatch.setattr(chexlocalize, "validate_partition", lambda cohort, pilots: cohort["cases"])
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return {(1, 2): "model\n", (3,): "Yes", (4,): "No"}[tuple(ids)]
        def encode(self, text, **kwargs):
            return {"model\n": [1, 2], "model\nYes": [1, 2, 3], "model\nNo": [1, 2, 4]}[text]
    class Config(SimpleNamespace):
        def to_dict(self):
            return vars(self)
    bundle = SimpleNamespace(model=SimpleNamespace(dtype=torch.float32, generation_config=Config()),
                             processor=SimpleNamespace(tokenizer=Tokenizer()), device="cpu")
    monkeypatch.setattr(inference, "load_model", lambda *a, **k: bundle)
    monkeypatch.setattr(inference, "prepare_inputs", lambda *a: {"input_ids": torch.tensor([[1, 2]])})
    monkeypatch.setattr(sae_reconstruction, "load_scope_sae", lambda *a, **k: (None, {"l0": 120}, None, "pinned"))
    collected = []
    def collect(bundle, sae, case, *args):
        collected.append(case["case_id"])
        record = {**copy.deepcopy(source_record), **case}
        payload = copy.deepcopy(source_payload)
        small = payload["features"].to_dense()
        large = torch.zeros(6, 16384)
        large[:, :4] = small
        payload["features"] = large.to_sparse().coalesce()
        record["feature_shape"] = list(large.shape)
        return record, payload
    monkeypatch.setattr(medgemma_features, "collect_discovery_case", collect)
    output = tmp_path / "resumed"
    medgemma_features.run_discovery(cohort_path, output, pilot_path, limit=1)
    first = output / "cases" / f"{cohort['cases'][0]['case_id']}.pt"
    original = file_sha256(first)
    medgemma_features.run_discovery(cohort_path, output, pilot_path)
    assert collected == [c["case_id"] for c in cohort["cases"]] and file_sha256(first) == original
    monkeypatch.setattr(inference, "load_model", lambda *a, **k: pytest.fail("Completed resume must not load model"))
    medgemma_features.run_discovery(cohort_path, output, pilot_path)
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["source_sha256"]["inference.py"] = "changed"
    atomic_json(protocol_path, protocol)
    with pytest.raises(ValueError, match="Resume protocol"):
        medgemma_features.run_discovery(cohort_path, output, pilot_path)


def test_collection_gates_greedy_argmax_and_preserves_position_groups(tmp_path, monkeypatch):
    from medgemma_grounding import inference, sae_reconstruction, medgemma_features
    path = tmp_path / "image.png"
    Image.new("RGB", (13, 7)).save(path)
    inputs = {"input_ids": torch.tensor([[0, 9, 9, 2]]), "attention_mask": torch.ones(1, 4)}
    baseline = torch.tensor([[[0., 0., 0., 4., 1.]]])
    hidden = torch.arange(8).reshape(1, 4, 2).float()
    class Tokenizer:
        all_special_ids = [0]
        def decode(self, ids, **kwargs):
            if len(ids) == 4:
                return "prefix"
            return {0: "<bos>", 9: "<image>", 2: "\n", 3: "Yes", 4: "No"}[ids[0]]
        def encode(self, text, **kwargs):
            return {"prefix": [0, 9, 9, 2], "prefixYes": [0, 9, 9, 2, 3], "prefixNo": [0, 9, 9, 2, 4], "question": [2]}[text]
        def convert_ids_to_tokens(self, token_id):
            return str(token_id)
    class Model:
        config = SimpleNamespace(image_token_index=9)
        def __call__(self, *a, **k):
            return SimpleNamespace(logits=baseline)
        def generate(self, **kwargs):
            return torch.tensor([[0, 9, 9, 2, self.first]])
        first = 3
    model = Model()
    tokenizer = Tokenizer()
    bundle = SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=tokenizer, decode=lambda *a, **k: "Yes"), device="cpu")
    monkeypatch.setattr(inference, "prepare_inputs", lambda *a: inputs)
    monkeypatch.setattr(sae_reconstruction, "capture_residual", lambda *a: (hidden.clone(), baseline.clone()))
    monkeypatch.setattr(sae_reconstruction, "encode_decode", lambda *a: (hidden[0] + 10, torch.ones(4, 4)))
    replacements = []
    def replace(model, inputs, hook, changed):
        replacements.append(changed)
        return baseline
    monkeypatch.setattr(sae_reconstruction, "replace_residual", replace)
    monkeypatch.setattr(medgemma_features, "spatial_layout", lambda *a: {"verified": False})
    candidates = answer_candidates(tokenizer, inputs["input_ids"])
    record, payload = medgemma_features.collect_discovery_case(bundle, SimpleNamespace(cfg=SimpleNamespace(d_sae=4)),
        {"image_path": str(path)}, "question", candidates, None)
    assert record["parsed_answer"] == "Yes" and record["positions"]["final"] == [3]
    assert record["controls"]["unchanged"]["score_change"] == 0
    assert len(replacements) == 5  # unchanged plus four reconstruction scopes
    torch.testing.assert_close(replacements[1][:, [0, 3]], hidden[:, [0, 3]])
    assert payload["features"].layout == torch.sparse_coo
    model.first = 4
    with pytest.raises(ValueError, match="Greedy first token"):
        medgemma_features.collect_discovery_case(bundle, SimpleNamespace(cfg=SimpleNamespace(d_sae=4)),
            {"image_path": str(path)}, "question", candidates, None)
