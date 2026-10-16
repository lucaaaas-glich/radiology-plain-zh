"""Separate V3 pipeline; never changes or silently substitutes the frozen V2."""
import time
from .common import chinese_ratio
from .glossary_v2 import glossary
from .guard_v2 import review
from .ledger_v3 import (VERSION, LEDGER_SYSTEM, WRITE_SYSTEM, AUDIT_SYSTEM, messages,
                        parse_json, validate_ledger, validate_translations, validate_audit)
from .structure_v3 import units, chunks


def explain(report, generate, progress=None, max_attempts=2, use_ledger=True):
    pieces = units(report)
    if max_attempts not in (1, 2):
        raise ValueError('Only one or two bounded attempts are allowed')
    start = time.perf_counter()
    result = {r['id']: dict(r, status='structure' if r['structure_only'] else 'source_only',
                           explanation_zh=None, ledger=None, attempts=[],
                           reason='该部分中文解释尚未通过文本核对，保留原文。') for r in pieces}
    calls = []

    def call(stage, prompt, max_tokens, target):
        response = generate(prompt, max_tokens=max_tokens)
        calls.append(dict(response, stage=stage, ids=[r['id'] for r in target]))
        return response

    for group in chunks(pieces):
        pending = list(group)
        feedback = []
        for attempt_index in range(max_attempts):
            if not pending:
                break
            attempt = {'attempt': attempt_index+1, 'ids': [r['id'] for r in pending], 'call_start': len(calls)}
            try:
                if use_ledger:
                    raw = call('ledger', messages(LEDGER_SYSTEM, report, pending, 上次检查问题=feedback), 2400, pending)
                    ledger = validate_ledger(parse_json(raw), pending)
                else:
                    ledger = {}
                attempt['ledger'] = list(ledger.values())
                raw = call('write', messages(WRITE_SYSTEM, report, pending, 事实清单=list(ledger.values()), 上次检查问题=feedback), 1600, pending)
                translated = validate_translations(parse_json(raw), pending)
                attempt['translations'] = list(translated.values())
                checks = {r['id']: review(r['source'], translated[r['id']]['text']) for r in pending}
                extra = {} if use_ledger else {'实验说明':'本实验不提取事实清单，只核对原文与中文，不评价不存在的清单。'}
                raw = call('audit', messages(AUDIT_SYSTEM, report, pending, 事实清单=list(ledger.values()),
                           中文解释=list(translated.values()), 自动提示仅供核对=checks, **extra), 1400, pending)
                audit = validate_audit(parse_json(raw), pending)
                attempt['audit'] = audit
                attempt['lexical_checks'] = checks
                feedback = list(audit['issues'])
                # Numerical mismatches and non-Chinese output cannot be waived
                # by a self-critic. Other old lexical flags are evidence for the
                # critic, not a claimed semantic certificate.
                for r in pending:
                    for flag in checks[r['id']]['issues']:
                        if flag['code'] in {'quantity','ambiguous_date','date_changed','numeric_value_missing'}:
                            feedback.append({'id':r['id'],'type':'meaning','reason':flag['detail']})
                    text = translated[r['id']]['text']
                    if chinese_ratio(text) < .45:
                        feedback.append({'id':r['id'],'type':'unclear','reason':'解释未以中文为主'})
                bad = {i['id'] for i in feedback}
                for r in pending:
                    if r['id'] not in bad:
                        result[r['id']].update(status='generated', explanation_zh=translated[r['id']]['text'],
                                               ledger=ledger.get(r['id']), reason=None)
                next_pending = [r for r in pending if r['id'] in bad]
            except (ValueError, TypeError, KeyError) as error:
                attempt['error'] = str(error)
                feedback = [{'id':r['id'],'type':'format','reason':str(error)} for r in pending]
                next_pending = pending
            attempt['call_end'] = len(calls)
            for r in pending:
                result[r['id']]['attempts'].append(attempt)
            pending = next_pending
        if progress:
            progress(len([r for r in result.values() if r['status']=='generated']), len(pieces))
    rows = [result[r['id']] for r in pieces]
    eligible = [r for r in rows if not r['structure_only']]
    count = sum(r['status']=='generated' for r in eligible)
    return {'version':VERSION,'use_ledger':use_ledger,'source_report':report,'segments':rows,'calls':calls,
            'glossary':glossary(report),'seconds':time.perf_counter()-start,
            'coverage':{'generated_segments':count,'eligible_segments':len(eligible),
                        'source_only_segments':len(eligible)-count,
                        'structural_segments':len(rows)-len(eligible),
                        'fully_generated':bool(eligible) and count==len(eligible)},
            'notice':'研究原型。原文证据及模型核对都不能证明医学正确；未完成部分保留原文。'}
