import json
import tempfile
import unittest
from pathlib import Path

from radplain.common import fingerprint, write_json, write_jsonl
from scripts.build_p0_study_packet import build, output_hash, prepare, render, sha


class StudyPacketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = 'Teaching sample <script>unsafe()</script> size 5 mm.'
        self.row = {'id': 'example', 'report': self.source, 'result': {
            'source_report': self.source,
            'segments': [{'id': 0, 'source': self.source, 'status': 'source_only',
                          'explanation_zh': 'DO_NOT_SHOW_FAILED_DRAFT'}], 'glossary': []}}
        self.inputs = {}
        self.save('sources', 'data/source.jsonl', [{'id': 'example', 'report': self.source}])
        for arm in ('v2', 'v3'):
            self.save(arm, f'data/{arm}.jsonl', [self.row])
            self.save(arm + '_review', f'data/{arm}_review.json', {'cases': [{
                'id': 'example', 'output_sha256': output_hash(self.row),
                'notes_zh': 'Historical AI claim, not a human label.', 'errors': ['omission']}]})
        self.manifest = {'inputs': self.inputs, 'display_seed': 371026,
                         'cases': [{'id': 'example', 'source_sha256': fingerprint(self.source)}]}
        self.manifest_path = self.root / 'manifest.json'
        write_json(self.manifest_path, self.manifest)

    def save(self, name, path, value):
        path = self.root / path
        (write_jsonl if path.suffix == '.jsonl' else write_json)(path, value)
        self.inputs[name] = {'path': str(path.relative_to(self.root)), 'sha256': sha(path)}

    def test_history_does_not_fill_human_labels_and_raw_is_local(self):
        out = self.root / 'data/p0/study'
        result = build(self.root, self.manifest_path, out)
        row = json.loads((out / 'annotations.jsonl').read_text())
        self.assertEqual(result['completed_human_annotations'], 0)
        self.assertFalse(row['human_review_completed'])
        self.assertIsNone(row['reviewer'])
        self.assertEqual(row['must_retain_facts'], [])
        self.assertTrue(all(r['status'] is None for r in row['candidate_reviews']))
        with self.assertRaises(FileExistsError):
            build(self.root, self.manifest_path, out)

    def test_outside_ignored_data_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ignored data/p0'):
            build(self.root, self.manifest_path, self.root / 'docs/study')
        self.assertFalse((self.root / 'docs/study').exists())

    def test_changed_input_rejected_before_output_creation(self):
        path = self.root / self.inputs['v2']['path']
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Input moved or changed'):
            build(self.root, self.manifest_path, self.root / 'data/p0/study')
        self.assertFalse((self.root / 'data/p0/study').exists())

    def test_stale_review_rejected_even_if_file_hash_updated(self):
        changed = json.loads(json.dumps(self.row))
        changed['result']['segments'][0]['explanation_zh'] = 'Changed output'
        self.save('v2', 'data/v2.jsonl', [changed])
        with self.assertRaisesRegex(ValueError, 'Historical review no longer matches'):
            prepare(self.root, self.manifest)

    def test_html_escapes_source_and_never_displays_rejected_draft(self):
        packet, _, _ = prepare(self.root, self.manifest)
        text = render(packet)
        self.assertNotIn('<script>', text)
        self.assertIn('&lt;script&gt;', text)
        self.assertNotIn('DO_NOT_SHOW_FAILED_DRAFT', text)
        self.assertIn('本段中文未完成', text)


if __name__ == '__main__':
    unittest.main()
