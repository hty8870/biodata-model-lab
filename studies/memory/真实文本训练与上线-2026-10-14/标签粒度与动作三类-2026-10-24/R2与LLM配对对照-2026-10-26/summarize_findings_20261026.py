from __future__ import annotations

from collections import Counter
from datetime import datetime
import json

import compare_r2_llm_20261026 as D


def main():
    data, manifest = D.inputs()
    rows = D.read(D.HERE / 'results/rows.json')
    summary = D.read(D.HERE / 'results/summary.json')
    pricing = D.read(D.HERE / 'pricing.json')
    originals = D.index_unique(data['rows'])
    singles, batches = D.collect_calls('single'), D.collect_calls('batch8')
    all_calls = singles + batches + D.collect_calls('smoke')
    assert len(singles) == 576 and len(batches) == 54 and len(all_calls) == 633
    by_key = {call['keys'][0]: call for call in singles}
    request_binding_checks = 0
    peak_checks = []
    for call in all_calls:
        if call['arm'] != 'smoke':
            selected = [originals[key] for key in call['keys']]
        else:
            smoke = D.index_unique(data['smoke'])
            selected = [smoke[key] for key in call['keys']]
        assert D.sha(D.input_prompt(selected)) == call['prompt_sha256']
        assert D.parse_response(call['response_text'], call['keys']) == call['predictions']
        assert call['usage']['known']
        dt = datetime.fromisoformat(call['ended_at'])
        peak_checks.append(dt.date().isoformat() == '2026-10-26' and 6 <= dt.hour < 10)
        request_binding_checks += 1
    independent = {}
    for domain in ('memory', 'public'):
        subset = [row for row in rows if row['domain'] == domain]
        name = 'memory432' if domain == 'memory' else 'public144'
        for arm, report in summary['quality'][name].items():
            correct, tp, fp, fn, failures = 0, 0, 0, 0, 0
            for row in subset:
                predicted = row[arm]
                if predicted is None:
                    failures += 1
                target_conflict = row['gold'] == 'conflict'
                pred_conflict = predicted == 'conflict'
                correct += int(predicted is not None and (predicted == row['gold'] if domain == 'memory' else pred_conflict == target_conflict))
                tp += int(pred_conflict and target_conflict)
                fp += int(pred_conflict and not target_conflict)
                fn += int(not pred_conflict and target_conflict)
            conflict = report['per_class']['conflict']
            assert correct == report['correct'] and failures == report['failures']
            assert tp == conflict['tp'] and fp == conflict['fp'] and fn == conflict['fn']
            independent[domain + ':' + arm] = {'n': len(subset), 'correct': correct, 'tp': tp, 'fp': fp, 'fn': fn, 'failures': failures}
    timings = {}
    serial_keys = data['serial_keys']
    for seed in D.SEEDS:
        value = D.read(D.HERE / f'local/r2-seed{seed}-cuda.json')
        groups = {key: [t['latency_s'] for t in value['batch1_timings'] if t['key'] == key] for key in serial_keys}
        assert all(len(values) == 3 for values in groups.values())
        timings[str(seed)] = {key: float(D.np.median(values)) for key, values in groups.items()}
    paired_latency = {}
    cpu = D.read(D.HERE / 'local/r2-seed13-cpu.json')
    cpu_times = {row['key']: row['latency_s'] for row in cpu['batch1_timings']}
    for domain in ('all', 'memory', 'public'):
        keys = [key for key in serial_keys if domain == 'all' or originals[key]['domain'] == domain]
        llm = D.latency_stats([by_key[key]['latency_s'] for key in keys])
        gpu = D.latency_stats([timings['13'][key] for key in keys])
        cp = D.latency_stats([cpu_times[key] for key in keys])
        paired_latency[domain] = {'llm_serial': llm, 'gpu_seed13_per_input_median_of_3': gpu,
                                 'cpu_seed13_threads2': cp,
                                 'llm_p50_over_gpu_p50': llm['p50_s'] / gpu['p50_s'],
                                 'llm_p50_over_cpu_p50': llm['p50_s'] / cp['p50_s']}
    memory_calls = [call for call in singles if call['keys'][0].startswith('memory:')]
    cost = {}
    for name, calls, n in (('memory_single', memory_calls, 432), ('memory_batch8', batches, 432),
                           ('public_single', [c for c in singles if c['keys'][0].startswith('public:')], 144)):
        estimate = D.cost_estimate(calls, pricing)
        estimate['pairs'] = n
        estimate['estimated_USD_per_10000_pairs'] = estimate['known_subtotal'] * 10000 / n
        cache_unknown = estimate['unknown_cache_assumed_miss_calls']
        subtotal = 0.0
        for call in calls:
            u = call['usage']
            hit = u.get('prompt_cache_hit_tokens', (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0))
            subtotal += (hit * pricing['input_hit_per_million'] + (u['prompt_tokens'] - hit) * pricing['input_miss_per_million']
                         + u['completion_tokens'] * pricing['output_per_million']) / 1e6
        assert abs(subtotal - estimate['known_subtotal']) < 1e-12 and cache_unknown == 0
        cost[name] = estimate
    cpu_seconds = paired_latency['memory']['cpu_seed13_threads2']['mean_s']
    gpu_seconds = paired_latency['memory']['gpu_seed13_per_input_median_of_3']['mean_s']
    for name in ('memory_single', 'memory_batch8'):
        per_pair = cost[name]['known_subtotal'] / 432
        cost[name]['break_even_USD_per_active_gpu_hour'] = per_pair * 3600 / gpu_seconds
        cost[name]['break_even_USD_per_active_cpu_hour'] = per_pair * 3600 / cpu_seconds
    hybrid = {}
    for seed in D.SEEDS:
        local = D.index_unique(D.read(D.HERE / f'local/r2-seed{seed}-cuda.json')['predictions'])
        for domain in ('memory', 'public'):
            subset = [row for row in data['rows'] if row['domain'] == domain]
            accepted = [row for row in subset if max(local[row['key']]['probabilities']) >= D.THRESHOLD]
            fallbacks = [by_key[row['key']] for row in subset if max(local[row['key']]['probabilities']) < D.THRESHOLD]
            total = sum(by_key[row['key']]['usage']['total_tokens'] for row in subset)
            spent = sum(call['usage']['total_tokens'] for call in fallbacks)
            keys = [key for key in serial_keys if originals[key]['domain'] == domain]
            sim_times = [timings[str(seed)][key] + (by_key[key]['latency_s'] if max(local[key]['probabilities']) < .8 else 0) for key in keys]
            hybrid[domain + f':seed{seed}'] = {'n': len(subset), 'local': len(accepted), 'fallbacks': len(fallbacks),
                'call_reduction': len(accepted) / len(subset), 'token_reduction': 1 - spent / total,
                'api_cost': D.cost_estimate(fallbacks, pricing), 'simulated_serial_latency': D.latency_stats(sim_times),
                'not_deployed': True}
    disagreements = {}
    for category, condition in (
        ('memory_r2both_right_llm_wrong', lambda r: r['domain'] == 'memory' and r['r2_seed13'] == r['gold'] == r['r2_seed42'] and r['llm_single'] != r['gold']),
        ('memory_llm_right_r2both_wrong', lambda r: r['domain'] == 'memory' and r['llm_single'] == r['gold'] and r['r2_seed13'] != r['gold'] and r['r2_seed42'] != r['gold']),
        ('memory_batch_right_single_wrong', lambda r: r['domain'] == 'memory' and r['llm_batch8'] == r['gold'] and r['llm_single'] != r['gold']),
        ('public_llm_right_r2both_wrong', lambda r: r['domain'] == 'public' and ((r['llm_single'] == 'conflict') == (r['gold'] == 'conflict'))
            and ((r['r2_seed13'] == 'conflict') != (r['gold'] == 'conflict')) and ((r['r2_seed42'] == 'conflict') != (r['gold'] == 'conflict')))):
        selected = sorted([row for row in rows if condition(row)], key=lambda r: r['id'])
        disagreements[category] = {'n': len(selected), 'first3_by_id': selected[:3]}
    compact = {'data_sha256': manifest['data_sha256'], 'request_binding_checks': request_binding_checks,
               'all_usage_known': all(call['usage']['known'] for call in all_calls),
               'all_provider_success': all(call['provider_success'] for call in all_calls),
               'all_calls_in_verified_peak_window': all(peak_checks),
               'official_deepseek_endpoint_hash_match': D.read(D.HERE / 'smoke.json')['identity']['endpoint_sha256'] in
                    {D.sha('https://api.deepseek.com'), D.sha('https://api.deepseek.com/v1'), D.sha('https://api.deepseek.com/')},
               'returned_models': dict(Counter(call['returned_model'] for call in all_calls)),
               'independent_counts': independent, 'paired_latency': paired_latency, 'cost': cost, 'hybrid': hybrid,
               'disagreements': disagreements,
               'source_sha256': {name: D.K.sha256_file(D.HERE / name) for name in (
                   'results/summary.json', 'results/rows.json', 'results/逐题对照-2026-10-26.csv', 'pricing.json')},
               'code_sha256': D.K.sha256_file(__file__)}
    D.publish(D.HERE / 'results/findings.json', compact)
    view = {key: value for key, value in compact.items() if key not in ('disagreements', 'source_sha256', 'code_sha256')}
    print(json.dumps(view, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
