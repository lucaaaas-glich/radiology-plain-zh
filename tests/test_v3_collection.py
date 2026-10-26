import json
import tempfile
import unittest
from pathlib import Path
from scripts.collect_v3_final import collect, digest


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = {}
        self.rows = [{'id': 'one', 'report': 'first'}, {'id': 'two', 'report': 'second'}]
        self.config = {'module': 'frozen', 'kwargs': {'use_ledger': True}}
        for name, field, arm in [('10-final20.json', 'main_request_sha256', 'v3'),
                                 ('11-ablation10.json', 'ablation_request_sha256', 'without_ledger')]:
            source = f'inputs/{arm}.jsonl'
            self.lines(source, self.rows)
            request = {'input': source, 'input_sha256': digest(self.root/source),
                       'out': f'results/{arm}', 'arms': {arm: self.config}}
            self.write('requests/'+name, request)
            self.lock[field] = digest(self.root/'requests'/name)
            self.lines(f'results/{arm}/{arm}/outputs.jsonl',
                       [dict(self.rows[0], error='completed failure is preserved')])
        self.write('final_candidate.lock.json', self.lock)
        self.write('committed-lock.json', self.lock)

    def write(self, name, obj):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj)+'\n')

    def lines(self, name, rows):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def continuation(self, duplicate=False, changed=False):
        source = 'results/resume-1/input-01.jsonl'
        rows = self.rows if duplicate else self.rows[1:]
        self.lines(source, rows)
        req = {'input': source, 'input_sha256': digest(self.root/source),
               'out': 'results/resume-1/part-01',
               'arms': {'v3': {'module':'different'} if changed else self.config}}
        self.write('results/resume-1/request-01.json', req)
        self.write('results/resume-1/plan.json', {'parts': [{
            'original_request': '10-final20.json', 'request': 'results/resume-1/request-01.json',
            'request_sha256': digest(self.root/'results/resume-1/request-01.json'),
            'ids': [r['id'] for r in rows], 'arms': ['v3']}]})
        self.lines(req['out']+'/v3/outputs.jsonl', [dict(r,result={}) for r in rows])

    def run_collect(self):
        return collect(self.root, self.root/'committed-lock.json')

    def test_preserves_failure_and_order_with_explicit_missing(self):
        self.continuation()
        rows, manifest = self.run_collect()
        self.assertIn('error', rows['v3'][0])
        self.assertEqual([r['id'] for r in rows['v3']], ['one','two'])
        self.assertEqual(manifest['missing']['without_ledger'], ['two'])
        self.assertFalse(manifest['complete'])

    def test_duplicate_completed_result_rejected_even_identical(self):
        self.continuation(duplicate=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate completed'):
            self.run_collect()

    def test_changed_generation_settings_rejected(self):
        self.continuation(changed=True)
        with self.assertRaisesRegex(ValueError, 'generation settings'):
            self.run_collect()

    def test_truncated_final_line_never_counts_as_completed(self):
        path = self.root/'results/v3/v3/outputs.jsonl'
        with path.open('ab') as f: f.write(b'{"id":"two"')
        rows, manifest = self.run_collect()
        self.assertEqual(len(rows['v3']), 1)
        self.assertEqual(manifest['missing']['v3'], ['two'])


if __name__ == '__main__':
    unittest.main()
