from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import json

import compare_r2_llm_20261026 as D


SALT = 'opaque-batch-after-id-audit-20261026'


def plan():
    data, manifest = D.inputs()
    originals = [row for row in data['rows'] if row['domain'] == 'memory']
    rows, mapping = [], {}
    for original in originals:
        key = 'q_' + D.sha([SALT, original['key']])[:12]
        if key in mapping:
            raise ValueError('Opaque identifier collision')
        mapping[key] = original['key']
        rows.append({'key': key, 'visible_text': original['visible_text']})
    requests = [rows[i:i + 8] for i in range(0, len(rows), 8)]
    for batch in requests:
        prompt = D.input_prompt(batch)
        if 'memory:cf-' in prompt or 'cf-' in prompt:
            raise ValueError('Source ID leak in clean prompt')
    value = {'schema': 'opaque-id-batch-repair/1', 'data_sha256': manifest['data_sha256'],
             'reason': 'Independent review found source suffixes correlated with gold in legacy batch8.',
             'change': 'Only displayed identifiers change; same texts, order, 8-case grouping, definitions and model settings.',
             'gold_based_tuning': False, 'old_results_preserved': True, 'salt': SALT,
             'mapping': mapping, 'requests': requests,
             'prompt_sha256': D.sha([D.PROMPT, D.SINGLE_SUFFIX, D.BATCH_SUFFIX]),
             'code_sha256': D.K.sha256_file(__file__)}
    D.publish(D.HERE / 'clean-batch-plan.json', value)
    print(json.dumps({'planned_requests': len(requests), 'pairs': len(rows), 'source_ids_in_prompts': 0}), flush=True)


def run():
    fixed = D.read(D.HERE / 'clean-batch-plan.json')
    require = D.sha([D.PROMPT, D.SINGLE_SUFFIX, D.BATCH_SUFFIX])
    if fixed['prompt_sha256'] != require or fixed['code_sha256'] != D.K.sha256_file(__file__):
        raise ValueError('Repair protocol drift')
    cfg = D.configuration(D.MODEL, 768)
    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(D.complete, rows, arm='batch8_clean', cfg=cfg) for rows in fixed['requests']]
        for index, future in enumerate(as_completed(tasks)):
            result = future.result()
            print(json.dumps({'clean_batches_done': index + 1, 'of': len(tasks), 'provider_success': result['provider_success']}), flush=True)


def analyze():
    fixed = D.read(D.HERE / 'clean-batch-plan.json')
    data, _ = D.inputs()
    calls = D.collect_calls('batch8_clean')
    if len(calls) != 54:
        raise ValueError('Repair collection incomplete')
    opaque = D.prediction_index(calls)
    if set(opaque) != set(fixed['mapping']):
        raise ValueError('Opaque response IDs do not match plan')
    predictions = {fixed['mapping'][key]: label for key, label in opaque.items()}
    old = D.read(D.HERE / 'results/summary.json')
    local = {seed: {row['key']: row['label'] for row in D.read(D.HERE / f'local/r2-seed{seed}-cuda.json')['predictions']} for seed in D.SEEDS}
    subsets = {'memory432': [row for row in data['rows'] if row['domain'] == 'memory'],
               'natural288': [row for row in data['rows'] if row['domain'] == 'memory' and row['pool'] != 'counterfactual'],
               'counterfactual144': [row for row in data['rows'] if row['domain'] == 'memory' and row['pool'] == 'counterfactual']}
    result = {'schema': 'clean-batch-supplement/1', 'plan_sha256': D.K.sha256_file(D.HERE / 'clean-batch-plan.json'),
              'quality': {name: D.measure(rows, predictions) for name, rows in subsets.items()},
              'paired': {name + f':seed{seed}': D.paired(rows, local[seed], predictions) for name, rows in subsets.items() for seed in D.SEEDS},
              'cost': D.cost_estimate(calls, D.read(D.HERE / 'pricing.json')),
              'latency': D.latency_stats([call['latency_s'] for call in calls]),
              'legacy_batch_confounded_accuracy': old['quality']['memory432']['llm_batch8']['accuracy'],
              'source_id_leak_removed': True, 'other_unlabelled_batch_context_remains': True,
              'not_independently_reviewed_new_responses': True, 'predictions': predictions}
    D.publish(D.HERE / 'results/clean-batch-comparison.json', result)
    final_rows = []
    for row in D.read(D.HERE / 'results/rows.json'):
        copy = dict(row)
        copy['llm_batch8_legacy_id_leak'] = copy.pop('llm_batch8')
        copy['llm_batch8_clean'] = predictions.get(('memory:' if row['domain'] == 'memory' else 'public:') + row['id'])
        final_rows.append(copy)
    D.publish(D.HERE / 'results/final-rows.json', final_rows)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(final_rows[0]))
    writer.writeheader()
    writer.writerows(final_rows)
    path = D.HERE / 'results/最终逐题对照-2026-10-26.csv'
    text = stream.getvalue()
    if path.exists() and path.read_text(encoding='utf-8-sig') != text.replace('\r\n', '\n'):
        raise ValueError('Existing final CSV differs')
    if not path.exists():
        with path.open('x', encoding='utf-8-sig', newline='') as stream:
            stream.write(text)
    print(json.dumps({key: value for key, value in result.items() if key not in ('predictions', 'paired')}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('plan', 'run', 'analyze'))
    args = parser.parse_args()
    {'plan': plan, 'run': run, 'analyze': analyze}[args.command]()
