from types import SimpleNamespace

import pytest
import torch

from medgemma_grounding.attention_knockout import conditions, token_groups, restrict_mask, knockout
from medgemma_grounding import medgemma_attention as study, attention_analysis as analysis
from medgemma_grounding.medgemma_features import atomic_json


def mask(n=6):
    return torch.where(torch.ones(n, n).tril().bool(), 0., float('-inf'))[None, None]


def test_orientation_restrictions_and_probability():
    original = mask()
    edited, effective, stats = restrict_mask(original, [5], [1, 2])
    assert stats['blocked_edges'] == 2
    assert torch.isneginf(edited[0, 0, 5, 1:3]).all()
    assert torch.equal(edited[0, 0, 1], original[0, 0, 1])
    assert torch.equal(original[~effective], edited[~effective])
    assert not ((edited == 0) & (original != 0)).any()
    probabilities = edited.softmax(-1)
    assert probabilities[0, 0, 5, 1:3].eq(0).all()
    assert probabilities.isfinite().all()


def test_pass_and_nonempty_placebo_exact_identity():
    original = mask()
    for mode in ('pass', 'placebo'):
        edited, _, stats = restrict_mask(original, [0], [4, 5], mode)
        assert torch.equal(original, edited) and stats['blocked_edges'] == 0
    assert stats['requested_edges'] == 2
    with pytest.raises(ValueError, match='entire query row'):
        restrict_mask(original, [0], [0])
    with pytest.raises(ValueError, match='additive'):
        restrict_mask(original.bool(), [5], [1])


def test_token_groups_include_template_and_final_not_answer():
    ids = torch.tensor([[7, 8, 9, 9, 9, 3, 4, 5]])
    x = {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'token_type_ids': (ids == 9).long()}
    g = token_groups(x, 9)
    assert g['image'] == [2, 3, 4] and g['post_image'] == [5, 6, 7] and g['final'] == [7]
    x['token_type_ids'].zero_()
    with pytest.raises(ValueError, match='block types'):
        token_groups(x, 9)


class Attention(torch.nn.Module):
    layer_type = 'full_attention'
    def forward(self, hidden_states, attention_mask):
        probabilities = attention_mask.softmax(-1)
        return probabilities @ hidden_states, probabilities


def toy():
    layers = [SimpleNamespace(self_attn=Attention()) for _ in range(2)]
    model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
    g = {'image': [1, 2], 'final': [5], 'post_image': [3, 4, 5], 'pre_image': [0]}
    return model, layers, g


def test_hooks_probability_counts_and_cleanup_on_error():
    model, layers, g = toy()
    c = {'route': 'B', 'start': 0, 'end': 1}
    with knockout(model, g, c, inspect=True, audit_routes=True) as stats:
        for layer in layers:
            layer.self_attn(hidden_states=torch.ones(1, 1, 6, 3), attention_mask=mask())
    assert len(stats) == 2 and all(r['target_max_probability'] == 0 for r in stats.values())
    assert all(not a.self_attn._forward_pre_hooks and not a.self_attn._forward_hooks for a in layers)
    with pytest.raises(RuntimeError, match='failure'):
        with knockout(model, g, c):
            raise RuntimeError('failure')
    assert all(not a.self_attn._forward_pre_hooks for a in layers)
    with pytest.raises(ValueError, match='one prefill'):
        with knockout(model, g, c):
            pass


def test_full_depth_b_isolates_image_in_multilayer_toy():
    model, layers, g = toy()
    first = torch.arange(18.).reshape(1, 1, 6, 3)
    second = first.clone()
    second[:, :, g['image']] += 100
    def run(x):
        with knockout(model, g, {'route': 'B', 'start': 0, 'end': 1}, audit_routes=True):
            for layer in layers:
                y, _ = layer.self_attn(hidden_states=x, attention_mask=mask())
                x = x + y
        return x[:, :, -1]
    torch.testing.assert_close(run(first), run(second), rtol=0, atol=0)


def test_condition_budget_and_patient_aggregation():
    c = conditions()
    assert len(c) == 18 and len({x['id'] for x in c}) == 18 and len(c) * 142 == 2556
    rows = [{'patient': 'a', 'delta_M': 2}, {'patient': 'a', 'delta_M': 4}, {'patient': 'b', 'delta_M': 1}]
    assert analysis.summary(rows, 'delta_M')['mean'] == 2
    assert analysis.summary(rows, 'delta_M')['patients'] == 2
    assert analysis.summary([], 'delta_M')['mean'] is None


def test_processed_input_hash_and_cohort_record_guards(tmp_path):
    x = {'input_ids': torch.tensor([[1, 2]]), 'pixel_values': torch.ones(1, 3, 2, 2).bfloat16()}
    assert study.tensor_hash(x) == study.tensor_hash({k: v.clone() for k, v in x.items()})
    old = study.tensor_hash(x)
    x['pixel_values'] += 1
    assert old != study.tensor_hash(x)
    p = {'protocol_hash': 'p', 'case_ids': ['allowed']}
    path = tmp_path / 'record.json'
    study.write_record(path, {'protocol_hash': 'p', 'case_id': 'heldout'})
    with pytest.raises(ValueError, match='cohort'):
        study.read_record(path, p)
    study.write_record(path, {'protocol_hash': 'p', 'case_id': 'allowed', 'value': 1})
    row = study.read_record(path, p)
    row['value'] = 2
    atomic_json(path, row)
    with pytest.raises(ValueError, match='checksum'):
        study.read_record(path, p)


def test_margin_direction_and_baseline_outcome_frozen():
    p = {'protocol_hash': 'x', 'scoring': {'ids': {'Yes': 0, 'No': 1}}}
    case = {'case_id': 'c', 'patient': 'p', 'label': 1, 'historical_outcome': 'FN',
            'historical_answer_score': {'score': -2}, 'image_sha256': 'abc'}
    original = torch.tensor([[[-1., 1.]]])
    changed = torch.tensor([[[3., 1.]]])
    row = study.measurement(case, p, original, changed, {'id': 'B_00_33'}, {'seconds': 1}, 'tensor')
    assert row['outcome'] == 'FN' and row['M'] == -2 and row['changed_M'] == 2
    assert row['delta_M'] == 4 and row['corrected'] and not row['introduced']


def test_offline_empty_report_and_viewer_no_model(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(study, 'runtime', lambda *a: pytest.fail('No model loading'))
    p = {'experiment': 'E-009', 'protocol_hash': 'x', 'conditions': conditions(), 'case_ids': ['case']}
    manifest = {'cases': []}
    monkeypatch.setattr(study, 'validate', lambda *a: manifest)
    atomic_json(tmp_path / 'protocol.json', p)
    result = analysis.report(tmp_path, p, manifest)
    assert not result['complete'] and result['knockout_records'] == 0
    app = AppTest.from_string('from pathlib import Path\nfrom medgemma_grounding.attention_analysis import viewer\n'
        f'viewer(Path({str(tmp_path)!r}))').run(timeout=30)
    assert not app.exception and not app.error
    assert len(app.tabs) == 5


def test_case_resume_and_identical_processed_blank_deduplication(tmp_path, monkeypatch):
    p = {'protocol_hash': 'p', 'case_ids': ['a', 'b'], 'conditions': conditions(),
         'scoring': {'ids': {'Yes': 0, 'No': 1}}}
    groups = {'image': [1, 2], 'final': [5], 'post_image': [3, 4, 5], 'pre_image': [0], 'valid': list(range(6))}
    cases = [{'case_id': cid, 'patient': cid, 'label': 1, 'image_sha256': 'image'+cid,
              'historical_outcome': 'TP', 'historical_answer_score': {'score': 2.}} for cid in ('a','b')]
    def inputs(bundle, p, case, data_root, blank=False):
        return {'input_ids': torch.arange(6)[None], 'pixel_values': torch.tensor([0. if blank else 1. if case['case_id']=='a' else 2.])}, groups
    forwards = []
    def forward(bundle, x, groups=None, condition=None, **kwargs):
        forwards.append((float(x['pixel_values'].item()), condition))
        value = x['pixel_values'].item() + (0.25 if condition and kwargs.get('mode') != 'pass' else 0)
        return torch.tensor([[[value, -value]]]), {'seconds': .1, 'blocked_edges': 2 if condition else 0, 'layers': {}}
    monkeypatch.setattr(study, 'image_inputs', inputs)
    monkeypatch.setattr(study, 'timed_forward', forward)
    for case in cases:
        study.execute_case(None, p, case, tmp_path, tmp_path)
        assert study.complete_case(tmp_path, p, case['case_id'])
    assert len(list((tmp_path/'blanks').glob('*.pt'))) == 1
    assert sum(value == 0 for value, _ in forwards) == 1
    before = len(forwards)
    study.execute_case(None, p, cases[0], tmp_path, tmp_path)
    assert len(forwards) == before + 2  # baseline/revalidation, no repeated treatments/gray
    baseline = tmp_path/'cases/a/baseline.pt'
    baseline.write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum'):
        study.complete_case(tmp_path,p,'a')
