"""Check archived trajectory flags, first passes and spatial checkpoint hashes."""
import csv
import hashlib
import json
from pathlib import Path
P = Path(__file__).resolve().parents[1] / 'revision_results'

def read(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    for name, expected in json.loads((P/'AUDIT_SHA256.json').read_text()).items():
        data = (P/name).read_bytes()
        lf = data.replace(b'\r\n', b'\n')
        variants = (data, lf, lf.replace(b'\n', b'\r\n')) if name.endswith(('.csv', '.json')) else (data,)
        assert any(hashlib.sha256(x).hexdigest() == expected for x in variants), name
    count = 0
    for directory in sorted((P/'audit').glob('*fine*')):
        steps = read(directory/'fine_grid_all_steps.csv')
        first = read(directory/'fine_grid_first_pass.csv')
        assert len(first) == 3
        for row in steps:
            assert int(row['certified_pass']) == int(float(row['certified_relative_adjoint_bound']) < 1e-5)
        for row in first:
            k = min(int(x['budget']) for x in steps if x['method'] == row['method'] and int(x['certified_pass']))
            assert k == int(row['first_certified_iteration'])
            assert k + 1 == int(row['first_certified_charged_actions'])
        count += len(steps)
        if directory.name.startswith('ignn'):
            seed = first[0]['seed']
            model = P/'audit'/f'ignn_arxiv_w32_e200_s{seed}'/'trained_model.pt'
            manifest = json.loads((directory/'manifest.json').read_text())
            assert hashlib.sha256(model.read_bytes()).hexdigest() == manifest['checkpoint_sha256']
            post = P/'audit'/f'ignn_postcheck_e200_s{seed}'
            assert json.loads((post/'postcheck_manifest.json').read_text())['checkpoint_sha256'] == manifest['checkpoint_sha256']
            lookup = {(r['method'], r['budget']): r for r in steps}
            matched = 0
            for row in read(post/'solver_budgets_charged.csv'):
                key = (row['method'], row['budget'])
                if key in lookup:
                    assert row['certified_pass'] == lookup[key]['certified_pass']
                    assert float(row['certified_relative_adjoint_bound']) == float(lookup[key]['certified_relative_adjoint_bound'])
                    matched += 1
            assert matched == 33
        if directory.name.startswith('spatial'):
            seed = first[0]['seed']
            model = P/'audit'/f'spatial_reproduced_s{seed}'/'trained_model.pt'
            manifest = json.loads((directory/'manifest.json').read_text())
            assert hashlib.sha256(model.read_bytes()).hexdigest() == manifest['checkpoint_sha256']
    assert count == 2880
    print('CAT_ARCHIVED_AUDITS_PASS', count)

if __name__ == '__main__':
    main()
