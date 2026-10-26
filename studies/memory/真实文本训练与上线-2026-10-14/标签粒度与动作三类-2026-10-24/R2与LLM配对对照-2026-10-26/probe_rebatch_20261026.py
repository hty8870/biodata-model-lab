from __future__ import annotations

import argparse
import json
import time

import compare_r2_llm_20261026 as D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=D.SEEDS, default=13)
    args = parser.parse_args()
    data, manifest = D.inputs()
    old = D.read(D.HERE / f'local/r2-seed{args.seed}-cuda.json')
    device = D.torch.device('cuda:0')
    entry = D.S.resolve_run(f'r2-seed{args.seed}', None)
    model, tokenizer, info = D.S.load_model(entry, checkpoint='terminal', device=device)
    encode = D.K.TRH.default_encode(tokenizer, device, D.K.MAX_LENGTH)
    public, accounting = D.O.diagnostic_rows()
    subset = [row for row in data['rows'] if row['domain'] == 'public']
    started = time.perf_counter()
    original = D.K.TRH.forward_probabilities(model, tokenizer, public, 'direct3', device,
        D.K.MAX_LENGTH, True, encode, batch_size=16)
    repeat = D.K.TRH.forward_probabilities(model, tokenizer, subset, 'direct3', device,
        D.K.MAX_LENGTH, True, encode, batch_size=16)
    D.torch.cuda.synchronize(device)
    lookup = {row['id']: i for i, row in enumerate(public)}
    selected_original = D.np.asarray([original[lookup[row['id']]] for row in subset])
    frozen = D.frozen_probabilities(args.seed, 'public')
    all_reference = D.np.asarray([frozen[row['id']] for row in public])
    rebatched = D.index_unique(old['predictions'])
    stored = D.np.asarray([rebatched[row['key']]['probabilities'] for row in subset])
    def diff(left, right):
        return {'n': len(left), 'max_probability_delta': float(D.np.max(D.np.abs(left - right))),
                'argmax_mismatches': int((left.argmax(1) != right.argmax(1)).sum()),
                'threshold_080_flips': int(((left.max(1) >= .8) != (right.max(1) >= .8)).sum())}
    output = {'seed': args.seed, 'dtype': 'BF16', 'padding': 'longest', 'batch_size': 16,
              'original1502_vs_frozen': diff(original, all_reference),
              'repeat_subset144_vs_stored': diff(repeat, stored),
              'original_context_vs_subset_context': diff(selected_original, stored),
              'seconds': time.perf_counter() - started, 'reload': info,
              'same_checkpoint_files': old['checkpoint_files'] == manifest['locks'][str(args.seed)]['files'],
              'input_binding': accounting['selected_input_sha256'],
              'selected_original_probabilities': {row['key']: selected_original[i].tolist() for i, row in enumerate(subset)},
              'interpretation': 'Controlled original-batch replay and reordered-subset replay; no parameter, input text or label changes.',
              'code_sha256': D.K.sha256_file(__file__)}
    D.publish(D.HERE / f'local/rebatch-probe-seed{args.seed}.json', output)
    print(json.dumps({key: value for key, value in output.items() if key not in ('selected_original_probabilities', 'reload')}, ensure_ascii=False), flush=True)
    if output['original1502_vs_frozen']['max_probability_delta'] > 1e-6 or output['repeat_subset144_vs_stored']['max_probability_delta'] > 1e-6:
        raise SystemExit('Original configuration not reproduced; do not resolve parity as mere rebatching')


if __name__ == '__main__':
    main()
