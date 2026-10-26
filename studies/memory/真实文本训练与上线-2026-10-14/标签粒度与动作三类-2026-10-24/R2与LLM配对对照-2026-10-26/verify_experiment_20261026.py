from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

import compare_r2_llm_20261026 as D
import dataset_recommender.llm.llm_client as LC


def require_equal(actual, expected, name):
    if actual != expected:
        raise ValueError('Binding mismatch: ' + name)


def contract(calls, data):
    fixed = D.read(D.HERE / 'clean-batch-plan.json')
    extra = [row for batch in fixed['requests'] for row in batch]
    by_key = D.index_unique(data['rows'] + data['smoke'] + extra)
    payloads = []
    for call in calls:
        cfg = call['identity']
        user = D.input_prompt([by_key[key] for key in call['keys']])
        require_equal(D.sha(user), call['prompt_sha256'], 'user prompt')
        payload = {'model': cfg['requested_model'], 'messages': [
            {'role': 'system', 'content': LC.prompts.CURATOR_SYSTEM_PROMPT},
            {'role': 'user', 'content': user}],
            'temperature': cfg['temperature'], 'max_tokens': cfg['max_output_tokens'],
            'thinking': {'type': 'enabled' if cfg['thinking'] else 'disabled'}}
        payloads.append({'request_id': call['request_id'], 'payload_sha256': D.sha(payload)})
    return {'schema': 'post-collection-production-message-contract/1',
            'captured_after_collection': True, 'independent_wire_capture': False,
            'basis': 'Unchanged production client at the recorded repository HEAD; reconstruction, not packet capture.',
            'client_sha256': D.K.sha256_file(LC.__file__),
            'system_prompt': LC.prompts.CURATOR_SYSTEM_PROMPT,
            'system_prompt_sha256': D.sha(LC.prompts.CURATOR_SYSTEM_PROMPT),
            'message_payloads': payloads, 'no_authentication_headers_saved': True}


def verify(capture_contract=False, seal=False):
    data, manifest = D.inputs()
    require_equal(len(data['rows']), 576, 'input denominator')
    for name, digest in manifest['source_files_sha256'].items():
        require_equal(D.K.sha256_file(D.OLD / name), digest, 'old source:' + name)
    dependencies = {'score_conflict': D.S.__file__, 'train_relation_heads': D.K.TRH.__file__,
                    'ocnli_loader': D.O.__file__, 'classification_metrics': D.EM.__file__}
    for name, path in dependencies.items():
        require_equal(D.K.sha256_file(path), manifest['dependencies_sha256'][name], 'dependency:' + name)
    require_equal(D.K.sha256_file(D.HERE / 'collection_driver_v1_20261026.py'),
                  D.read(D.HERE / 'api-run.json')['code_sha256'], 'actual collection source archive')
    calls = []
    for arm, n in (('single', 576), ('batch8', 54), ('smoke', 3), ('batch8_clean', 54)):
        selected = D.collect_calls(arm)
        require_equal(len(selected), n, 'calls:' + arm)
        for call in selected:
            path = D.HERE / 'calls' / arm / (call['request_id'] + '.attempt.json')
            attempt = D.read(path)
            require_equal(attempt['keys'], call['keys'], 'attempt keys')
            require_equal(D.sha(attempt['prompt']), call['prompt_sha256'], 'attempt prompt')
            if arm == 'batch8_clean' and ('cf-' in attempt['prompt'] or 'memory:cf-' in attempt['prompt']):
                raise ValueError('Source identifier still present in repaired batch')
            if not call['provider_success'] or not call['usage']['known']:
                raise ValueError('Failure or missing usage in supposedly complete result')
            require_equal(D.parse_response(call['response_text'], call['keys']), call['predictions'], 'raw response parse')
            for field in ('started_at',):
                stamp = datetime.fromisoformat(attempt[field])
                if stamp.date().isoformat() != '2026-10-26' or not 6 <= stamp.hour < 10:
                    raise ValueError('Published peak price not applicable to all attempts')
        calls.extend(selected)
    fixed = D.read(D.HERE / 'clean-batch-plan.json')
    clean = D.read(D.HERE / 'results/clean-batch-comparison.json')
    require_equal(D.K.sha256_file(D.HERE / 'clean_batch_20261026.py'), fixed['code_sha256'], 'frozen repair code')
    memory = [row for row in data['rows'] if row['domain'] == 'memory']
    flattened = [row for batch in fixed['requests'] for row in batch]
    require_equal(len(flattened), 432, 'repair input denominator')
    for original, anonymous in zip(memory, flattened):
        require_equal(original['visible_text'], anonymous['visible_text'], 'repair preserves text and order')
        require_equal(fixed['mapping'][anonymous['key']], original['key'], 'repair ID mapping')
    opaque = D.prediction_index(D.collect_calls('batch8_clean'))
    mapped = {fixed['mapping'][key]: value for key, value in opaque.items()}
    require_equal(mapped, clean['predictions'], 'clean raw prediction reconstruction')
    correct = sum(mapped[row['key']] == row['gold'] for row in memory)
    tp = sum(mapped[row['key']] == row['gold'] == 'conflict' for row in memory)
    fp = sum(mapped[row['key']] == 'conflict' and row['gold'] != 'conflict' for row in memory)
    fn = sum(mapped[row['key']] != 'conflict' and row['gold'] == 'conflict' for row in memory)
    require_equal(correct, clean['quality']['memory432']['correct'], 'clean independent correct')
    for field, count in (('tp', tp), ('fp', fp), ('fn', fn)):
        require_equal(count, clean['quality']['memory432']['per_class']['conflict'][field], 'clean independent ' + field)
    final_rows = D.read(D.HERE / 'results/final-rows.json')
    require_equal(len(final_rows), 576, 'final export denominator')
    for row in final_rows:
        key = ('memory:' if row['domain'] == 'memory' else 'public:') + row['id']
        require_equal(row['llm_batch8_clean'], mapped.get(key), 'final CSV source clean label')
    expected_contract = contract(calls, data)
    contract_path = D.HERE / 'results/production-message-contract-all-687.json'
    if capture_contract:
        D.publish(contract_path, expected_contract)
    require_equal(D.read(contract_path), expected_contract, 'production message reconstruction')
    for seed in D.SEEDS:
        local = D.read(D.HERE / f'local/r2-seed{seed}-cuda.json')
        require_equal(local['data_sha256'], manifest['data_sha256'], 'local inputs')
        require_equal(local['checkpoint_files'], manifest['locks'][str(seed)]['files'], 'local checkpoint pin')
        actual = D.K.checkpoint_file_shas(D.K.RUNS_DIR / f'r2-seed{seed}/checkpoints/terminal')
        require_equal(actual, local['checkpoint_files'], 'current old checkpoint')
        probe = D.read(D.HERE / f'local/rebatch-probe-seed{seed}.json')
        for name in ('original1502_vs_frozen', 'repeat_subset144_vs_stored'):
            if probe[name]['max_probability_delta'] > 1e-6 or probe[name]['argmax_mismatches']:
                raise ValueError('Same-configuration reproducibility not closed')
        require_equal(probe['original_context_vs_subset_context']['argmax_mismatches'], 0, 'rebatch hard predictions')
        require_equal(probe['original_context_vs_subset_context']['threshold_080_flips'], 0, 'rebatch .80 gate decisions')
    findings = D.read(D.HERE / 'results/findings.json')
    require_equal(findings['request_binding_checks'], 633, 'independent call audit')
    for name, digest in findings['source_sha256'].items():
        require_equal(D.K.sha256_file(D.HERE / name), digest, 'findings source:' + name)
    sources = sorted(D.HERE.glob('*.py'))
    for path in sources:
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    report_path = D.HERE / '报告-2026-10-26.md'
    checked_links = 0
    for path in (report_path, D.HERE / '设计-2026-10-26.md'):
        text = path.read_text(encoding='utf-8')
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
            if target.startswith(('https://', 'http://', '#')):
                continue
            local = path.parent / target.split('#', 1)[0]
            if not local.exists():
                raise ValueError('Broken local report link: ' + target)
            checked_links += 1
    usage = D.total_usage(calls)
    require_equal(usage['total_tokens'], 402466, 'complete provider token accounting')
    estimated = D.cost_estimate(calls, D.read(D.HERE / 'pricing.json'))['known_subtotal']
    if abs(estimated - .098645196) > 1e-12:
        raise ValueError('Complete published-price estimate differs')
    response_ids = {call['raw_response'].get('id') for call in calls}
    require_equal(len(response_ids), 687, 'distinct provider responses')
    receipt = D.read(D.HERE / 'review-receipt.json')
    review_path = D.REPO.parent / '研究' / receipt['review_file_relative_to_research_root']
    files = [path for path in D.HERE.rglob('*') if path.is_file() and path.name != '验收-2026-10-26.json']
    result = {'status': 'verified', 'inputs': len(data['rows']), 'real_calls': len(calls),
              'old_source_pins': len(manifest['source_files_sha256']), 'student_seeds': list(D.SEEDS),
              'source_files_compiled': len(sources), 'report_links_checked': checked_links,
              'all_usage_known': True, 'raw_predictions_reparsed': sum(len(call['keys']) for call in calls),
              'provider_tokens': usage['total_tokens'], 'published_estimated_USD': estimated,
              'review_sha256': D.K.sha256_file(review_path),
              'artifact_files': len(files), 'artifact_bytes': sum(path.stat().st_size for path in files),
              'files_sha256': {path.relative_to(D.HERE).as_posix(): D.K.sha256_file(path) for path in sorted(files)},
              'no_new_training_or_api_calls': True, 'production_gates_not_run': 'No production changes; research-only.'}
    seal_path = D.HERE / '验收-2026-10-26.json'
    if seal:
        if '待最终只读审查与验收' in report_path.read_text(encoding='utf-8'):
            raise ValueError('Report still declares review pending; do not seal')
        D.publish(seal_path, result)
    elif seal_path.exists():
        require_equal(D.read(seal_path), result, 'sealed artifacts have changed')
    print(json.dumps({key: value for key, value in result.items() if key != 'files_sha256'}, ensure_ascii=False), flush=True)


def selftest():
    require_equal('expected', 'expected', 'positive')
    try:
        require_equal('wrong', 'expected', 'negative')
    except ValueError:
        pass
    else:
        raise AssertionError('Binding mismatch accepted')
    D.selftest()
    print(json.dumps({'binding_checks_passed': 2}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-contract', action='store_true')
    parser.add_argument('--seal', action='store_true')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        verify(args.capture_contract, args.seal)
