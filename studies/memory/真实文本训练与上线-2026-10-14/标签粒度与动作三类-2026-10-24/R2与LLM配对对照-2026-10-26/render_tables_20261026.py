from __future__ import annotations

import json
import compare_r2_llm_20261026 as D


def main():
    s = D.read(D.HERE / 'results/summary.json')
    f = D.read(D.HERE / 'results/findings.json')
    parts = ['# 固定结果表（2026-10-26）', '', '由已有逐题结果确定性生成，不调用模型。', '']
    for subset in ('natural288', 'counterfactual144', 'memory432', 'public144'):
        parts.extend(['## ' + subset, '', '| 方案 | 正确/n | macro-F1 | 冲突precision | 冲突recall | 冲突F1 | FP | FN |', '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for arm, result in s['quality'][subset].items():
            c = result['per_class']['conflict']
            parts.append(f"| {arm} | {result['correct']}/{result['n']} | {result['macro_f1']:.6f} | {c['precision']:.6f} | {c['recall']:.6f} | {c['f1']:.6f} | {int(c['fp'])} | {int(c['fn'])} |")
        parts.append('')
    parts.extend(['## 配对差异（左R2减右LLM，语义簇95%区间）', '', '| 子集 | 对照 | accuracy差 | 95%区间 | macro-F1差 | 95%区间 | R2独对 | LLM独对 |', '|---|---|---:|---|---:|---|---:|---:|'])
    for name, result in s['paired'].items():
        acc, f1 = result['intervals']['cluster:accuracy'], result['intervals']['cluster:macro_f1']
        parts.append(f"| {name.split(':')[0]} | {name.split(':')[1]} | {acc['point']:.6f} | {acc['ci']} | {f1['point']:.6f} | {f1['ci']} | {result['left_only_correct']} | {result['right_only_correct']} |")
    parts.extend(['', '## 固定0.80级联（模拟）', '', '| 子集 | seed | 本地接管/n | 回退LLM | 高置信错误 |', '|---|---:|---:|---:|---:|'])
    for name, result in s['cascade'].items():
        parts.append(f"| {name.split(':')[0]} | {name.split(':')[1]} | {result['local_accepted']}/{result['n']} | {result['llm_fallbacks']} | {len(result['high_confidence_wrong'])} |")
    parts.extend(['', '## GPU批量吞吐、加载与LLM批次', ''])
    for key in ('r2_seed13_cuda', 'r2_seed42_cuda', 'llm_batch8_concurrent4'):
        parts.extend(['### ' + key, '', '```json', json.dumps(s['latency'][key], ensure_ascii=False, indent=2), '```', ''])
    cost = {name: {key: value for key, value in result.items() if key != 'pricing'} for name, result in s['costs'].items()}
    parts.extend(['## 真实调用总费用（公开价估算）', '', '```json', json.dumps(cost, ensure_ascii=False, indent=2), '```', ''])
    text = '\n'.join(parts)
    path = D.HERE / 'results/结果表-2026-10-26.md'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError('Existing tables differ')
    if not path.exists():
        path.write_text(text, encoding='utf-8', newline='\n')
    print(text[:15500], flush=True)


if __name__ == '__main__':
    main()
