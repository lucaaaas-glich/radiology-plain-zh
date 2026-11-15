"""Build a local-only learning packet from already evaluated reports. No inference."""
import argparse
import hashlib
import html
import json
import random
from pathlib import Path

from radplain.common import fingerprint as source_hash, read_jsonl, write_json, write_jsonl
from scripts.summarize_v2 import fingerprint as output_hash


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows):
    result = {r['id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate record IDs')
    return result


def prepare(root, manifest):
    paths = {}
    for name, spec in manifest['inputs'].items():
        path = (root / spec['path']).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != spec['sha256']:
            raise ValueError('Input moved or changed: ' + name)
        paths[name] = path
    sources = indexed(read_jsonl(paths['sources']))
    outputs = {arm: indexed(read_jsonl(paths[arm])) for arm in ('v2', 'v3')}
    reviews = {arm: indexed(json.loads(paths[arm + '_review'].read_text())['cases'])
               for arm in ('v2', 'v3')}
    if len({r['id'] for r in manifest['cases']}) != len(manifest['cases']):
        raise ValueError('Duplicate study IDs')
    rng = random.Random(manifest['display_seed'])
    packet, annotations, mapping = [], [], []
    for case in manifest['cases']:
        key = case['id']
        src = sources[key]
        if source_hash(src['report']) != case['source_sha256']:
            raise ValueError('Source hash mismatch: ' + key)
        order = ['v2', 'v3']
        rng.shuffle(order)
        candidates, candidate_reviews, history = [], [], []
        for alias, arm in zip(('A', 'B'), order):
            row, old = outputs[arm][key], reviews[arm][key]
            if row['report'] != src['report'] or old['output_sha256'] != output_hash(row):
                raise ValueError('Historical review no longer matches source/output: ' + key)
            result = row['result']
            if result['source_report'] != src['report']:
                raise ValueError('Displayed source differs from input: ' + key)
            candidates.append({'alias': alias, 'output_sha256': output_hash(row),
                               'segments': [{k: s.get(k) for k in
                                             ('id', 'status', 'source', 'explanation_zh')}
                                            for s in result['segments']],
                               'glossary': result.get('glossary', [])})
            candidate_reviews.append({'alias': alias, 'output_sha256': output_hash(row),
                                      'status': None, 'fact_checks': [], 'unsupported_claims': [],
                                      'term_definition_issues': [], 'unjudgeable_items': [],
                                      'notes': None})
            history.append({'alias': alias, 'historical_ai_review': old})
            mapping.append({'id': key, 'alias': alias, 'arm': arm})
        packet.append({'id': key, 'source_sha256': case['source_sha256'],
                       'source': src['report'], 'candidates': candidates, 'history': history})
        annotations.append({'id': key, 'source_sha256': case['source_sha256'],
                            'purpose': 'old_report_development_learning_only',
                            'reviewer': None, 'reviewer_background': None,
                            'human_review_completed': False, 'started_at': None,
                            'finished_at': None, 'active_minutes': None,
                            'must_retain_facts': [], 'source_unjudgeable_items': [],
                            'candidate_reviews': candidate_reviews,
                            'second_review_status': 'not_arranged'})
    return packet, annotations, mapping


def render(packet):
    esc = lambda value: html.escape(str(value or ''), quote=True)
    parts = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>P0：10份旧报告学习包</title><style>
body{max-width:960px;margin:32px auto;padding:0 20px;font:17px/1.75 system-ui;color:#202832;background:#fafbfc}
article{background:white;border:1px solid #d8dfe7;border-radius:10px;padding:22px;margin:26px 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.7 ui-monospace,monospace;background:#f1f4f7;padding:16px}
summary{cursor:pointer;font-weight:600;padding:10px 0}li{margin:9px 0}.note{color:#526171}.pending{color:#875009}
</style><h1>先读原文，再判断改写</h1>
<p>这10份是已经使用过的真实报告，专门用于学习和开发，包含旧失败例与对照例。它们不能作为新的最终测试，也不能据此估计总体错误率。</p>
<p><b>目前人工标注完成数：0。</b>此页没有新模型输出。A/B只是打乱旧版本的显示顺序，本学习包不构成独立盲评。</p>
<ol><li>先只读原文，列出必须保留的事实与证据；不懂的内容标为无法判断。</li>
<li>再展开A/B，核对数字所指对象、时间、否定、部位和建议语气。术语释义另行判断。</li>
<li>最后才看历史AI分析，记录不同意见；AI分析不是标准答案。</li>
<li>记录开始、结束及实际审阅分钟数。把自己的判断填入同目录的 annotations.jsonl；本页不会自动保存批注。</li></ol>''']
    for i, case in enumerate(packet, 1):
        parts.append(f'<article><h2>练习 {i} / {len(packet)}</h2><p class="note">{esc(case["id"])}</p>')
        parts.append('<h3>第一步：原文</h3><pre>' + esc(case['source']) + '</pre>')
        parts.append('<p class="note">先记录：原文证据、对象/部位、数值及时间、否定与不确定性、建议条件。记录完成后再展开下面的候选。</p>')
        for candidate in case['candidates']:
            parts.append('<details><summary>第二步：展开候选 ' + esc(candidate['alias']) + '</summary><ol>')
            for segment in candidate['segments']:
                status = segment['status']
                if status == 'generated':
                    text = segment['explanation_zh']
                elif status == 'structure':
                    text = segment['source']
                else:
                    text = '【本段中文未完成／未生成】' + segment['source']
                parts.append('<li>' + esc(text) + '</li>')
            parts.append('</ol>')
            if candidate['glossary']:
                parts.append('<h4>该版本附带的一般术语说明（另行核对）</h4><ul>')
                for item in candidate['glossary']:
                    parts.append('<li>' + esc(item.get('term', item.get('source_term', ''))) + '：' +
                                 esc(item.get('definition_zh', '')) + '</li>')
                parts.append('</ul>')
            parts.append('</details>')
        parts.append('<details><summary>第三步：历史AI分析，仅供讨论</summary><p class="pending">这是旧AI审阅，不是人工金标准；不要直接复制为自己的标签。</p>')
        for history in case['history']:
            parts.append('<p><b>候选 ' + esc(history['alias']) + '</b>：' +
                         esc(history['historical_ai_review']['notes_zh']) + '</p>')
        parts.append('</details></article>')
    parts.append('</html>')
    return ''.join(parts)


def build(root, manifest_path, out):
    root, out = root.resolve(), out.resolve()
    if not out.is_relative_to(root / 'data' / 'p0'):
        raise ValueError('Raw study packets must stay under ignored data/p0/')
    manifest = json.loads(manifest_path.read_text())
    packet, annotations, mapping = prepare(root, manifest)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'packet.json', packet)
    write_jsonl(out / 'annotations.jsonl', annotations)
    write_json(out / 'historical_arm_mapping.json', mapping)
    (out / 'study.html').write_text(render(packet), encoding='utf-8')
    receipt = {'manifest_sha256': sha(manifest_path), 'reports': len(packet),
               'candidate_outputs': 2 * len(packet), 'new_inference_calls': 0,
               'completed_human_annotations': 0,
               'files': {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--manifest', type=Path, default=Path('examples/p0/study_manifest.json'))
    parser.add_argument('--out', type=Path, default=Path('data/p0/study-20261116'))
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.root / args.manifest, args.root / args.out), indent=2))


if __name__ == '__main__':
    main()
