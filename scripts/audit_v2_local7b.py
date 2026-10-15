"""Bounded 7B development comparison on the same unchanged dev7 pipeline.

No holdout use. Run serially after the context ablation. MLX memory limits are
guidelines, not an OS hard limit; actual peak and total process RSS are recorded.
"""
import hashlib
import json
import resource
import signal
import time
from pathlib import Path
import mlx.core as mx
from radplain.common import read_jsonl, write_json
from radplain.explain_v2 import explain
from radplain.mlx_backend import MLXBackend
from radplain.prompts import messages


MODEL = 'artifacts/v2/qwen25-7b'
MEMORY_BUDGET = int(5.5 * 1024**3)
CACHE_LIMIT = 128 * 1024**2
OUTPUT = Path('results/v2/dev_qwen25_7b_01')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def deadline(_signum, _frame):
    raise TimeoutError('Development experiment reached its 45 minute total budget')


def main():
    previous = json.loads(Path('results/v2/real-data/dev_qwen35_contextless_01/runtime.json').read_text())
    if previous['count'] != 16 or previous['source_changed_during_run']:
        raise ValueError('Previous experiment must finish with unchanged source')
    for file, sha in previous['source_files_sha256'].items():
        if digest(file) != sha:
            raise ValueError('Development source changed: '+file)
    lock = json.loads(Path('results/v2/models.lock.json').read_text())
    model = next(x for x in lock if x['local_path'] == MODEL)
    if model['revision'] != 'c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed':
        raise ValueError('Unexpected model revision')
    for name, metadata in model['files'].items():
        if digest(Path(MODEL)/name) != metadata['sha256']:
            raise ValueError('Model digest mismatch: '+name)
    runs = [
        ('synthetic_direct', 'examples/challenge.jsonl', OUTPUT/'synthetic_direct', 'direct', 384),
        ('real_direct', 'data/v2/real/dev.jsonl', Path('results/v2/real-data/dev_qwen25_direct_01'), 'direct', 1024),
        ('synthetic_critic', 'examples/challenge.jsonl', OUTPUT/'synthetic_critic', 'evidence-critic', 384),
        ('real_critic', 'data/v2/real/dev.jsonl', Path('results/v2/real-data/dev_qwen25_critic_01'), 'evidence-critic', 384),
    ]
    if OUTPUT.exists() or any(out.exists() for _, _, out, _, _ in runs):
        raise ValueError('Do not overwrite development outputs')
    OUTPUT.mkdir(parents=True)
    hashes = {str(p): digest(p) for p in [*Path('radplain').glob('*.py'), Path(__file__)]}
    metadata = {'stage': 'development_only', 'model': model,
                'context_policy': 'all_previous_source_sentences',
                'mlx_memory_guideline_bytes': MEMORY_BUDGET,
                'mlx_cache_limit_bytes': CACHE_LIMIT, 'wall_clock_budget_seconds': 2700,
                'source_files_sha256': hashes, 'note': 'Memory guideline is not an OS hard limit.'}
    write_json(OUTPUT/'plan.json', metadata)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(2700)
    start = time.perf_counter()
    try:
        mx.set_memory_limit(MEMORY_BUDGET)
        mx.set_cache_limit(CACHE_LIMIT)
        backend = MLXBackend(MODEL)
        for split, source, out, mode, max_tokens in runs:
            out.mkdir(parents=True)
            count, seconds = 0, 0
            with (out/'outputs.jsonl').open('x') as f:
                for row in read_jsonl(source):
                    if mode == 'direct':
                        result = backend(messages(row['report']), max_tokens=max_tokens)
                    else:
                        result = explain(row['report'], backend, max_tokens=max_tokens, use_critic=True)
                    f.write(json.dumps({'id': row['id'], 'report': row['report'], 'result': result}, ensure_ascii=False)+'\n')
                    f.flush(); count += 1; seconds += result['seconds']
                    print(split, row['id'], round(result['seconds'], 2), flush=True)
                    if mx.get_peak_memory() > MEMORY_BUDGET:
                        raise MemoryError('Observed MLX peak exceeded the experiment budget')
                    mx.clear_cache()
            write_json(out/'runtime.json', dict(metadata, count=count, mode=mode, max_tokens=max_tokens,
                split=split, generation_seconds=seconds, load_seconds=backend.load_seconds,
                input_sha256=digest(source), peak_mlx_allocated_bytes=mx.get_peak_memory(),
                peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                source_changed_during_run=any(digest(p) != sha for p, sha in hashes.items())))
        write_json(OUTPUT/'completion.json', {'status': 'complete', 'seconds': time.perf_counter()-start})
        print('LOCAL7B_DEVELOPMENT_COMPLETE', flush=True)
    except Exception as error:
        # Avoid logging source report text or partial generations in exception messages.
        write_json(OUTPUT/'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
            'seconds': time.perf_counter()-start, 'peak_mlx_allocated_bytes': mx.get_peak_memory(),
            'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'note': 'Partial outputs retained; no complete-result claim. See code for resource limits.'})
        print('LOCAL7B_DEVELOPMENT_FAILED', type(error).__name__, flush=True)
        raise SystemExit(1)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
