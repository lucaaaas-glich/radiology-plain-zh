"""Separate V3 pipeline; never changes or silently substitutes the frozen V2."""
import time
from .common import chinese_ratio
from .glossary_v3 import glossary
from .guard_v3 import review
from .ledger_v3 import (VERSION, LEDGER_SYSTEM, WRITE_SYSTEM, AUDIT_SYSTEM, messages,
                        parse_json, keyed, validate_ledger, validate_translations, validate_audit)
from .structure_v3 import units, chunks


def explain(report, generate, progress=None, max_attempts=2, use_ledger=True):
    pieces = units(report)
    if max_attempts not in (1, 2):
        raise ValueError('Only one or two bounded attempts are allowed')
    start = time.perf_counter()
    result = {r['id']: dict(r, status='structure' if r['structure_only'] else 'source_only',
                           explanation_zh=None, ledger=None, ledger_status='not_attempted', attempts=[],
                           reason='该部分中文解释尚未通过文本核对，保留原文。') for r in pieces}
    calls = []
    ledger_batches = []

    def call(stage, prompt, max_tokens, target):
        response = generate(prompt, max_tokens=max_tokens)
        calls.append(dict(response, stage=stage, ids=[r['id'] for r in target]))
        return response

    for group in chunks(pieces):
        ledger = {}
        batch = {'ids':[r['id'] for r in group], 'call_index':None, 'errors':{}}
        if use_ledger:
            try:
                batch['call_index']=len(calls)
                raw=call('ledger',messages(LEDGER_SYSTEM,report,group),2400,group)
                value=parse_json(raw)
                if set(value)!={'units'}:raise ValueError('Expected units in ledger')
                mapping=keyed(value['units'],[r['id'] for r in group])
                # One malformed unit must not discard other units' grounded
                # checklists. The immutable original remains available to all.
                for item in group:
                    try:
                        ledger.update(validate_ledger({'units':[mapping[item['id']]]},[item]))
                        result[item['id']]['ledger_status']='quotes_validated'
                    except (ValueError,TypeError,KeyError) as error:
                        batch['errors'][item['id']]=str(error)
                        result[item['id']]['ledger_status']='format_failed'
            except (ValueError,TypeError,KeyError) as error:
                batch['errors']={r['id']:str(error) for r in group}
                for item in group:result[item['id']]['ledger_status']='format_failed'
        else:
            for item in group:result[item['id']]['ledger_status']='disabled'
        ledger_batches.append(batch)
        pending = list(group)
        feedback = []
        for attempt_index in range(max_attempts):
            if not pending:
                break
            attempt = {'attempt': attempt_index+1, 'ids': [r['id'] for r in pending], 'call_start': len(calls)}
            try:
                active_ledger=[ledger[r['id']] for r in pending if r['id'] in ledger]
                missing_ledger=[r['id'] for r in pending if r['id'] not in ledger]
                attempt['ledger'] = active_ledger
                meanings=[{'原文词':g['source_term'],'中文词':g['term'],'词义':g['definition_zh']}
                          for g in glossary(' '.join(r['source'] for r in pending))]
                raw = call('write', messages(WRITE_SYSTEM, report, pending, 事实清单=active_ledger,
                           未提供清单的片段id直接按原文解释=missing_ledger,
                           固定词义参考不是患者新发现=meanings, 上次检查问题=feedback), 1600, pending)
                translated = validate_translations(parse_json(raw), pending)
                attempt['translations'] = list(translated.values())
                checks = {r['id']: review(r['source'], translated[r['id']]['text']) for r in pending}
                # Weak lexical flags confuse word presence with clinical meaning:
                # e.g. a definition of mediastinum mentions both lungs. Keep the
                # full diagnostic trace, but do not prime the bilingual auditor
                # with those unreliable semantic hints.
                hard_codes={'quantity','ambiguous_date','date_changed','numeric_value_missing','projection_direction'}
                hard_checks={key:[flag for flag in check['issues'] if flag['code'] in hard_codes]
                             for key,check in checks.items()}
                extra = {'未提供清单的片段id':missing_ledger,
                         '清单范围说明':'只检查实际提供的清单。未提供清单的片段仍要严格双向核对原文与中文；不要评价不存在的清单。'}
                raw = call('audit', messages(AUDIT_SYSTEM, report, pending, 事实清单=active_ledger,
                           中文解释=list(translated.values()), 数值日期投照核对提示=hard_checks, **extra), 1400, pending)
                audit = validate_audit(parse_json(raw), pending)
                attempt['audit'] = audit
                attempt['lexical_checks'] = checks
                attempt['inapplicable_ledger_issues']=[i for i in audit['issues'] if i['type']=='ledger' and i['id'] not in ledger]
                # A complaint about a nonexistent checklist is inapplicable;
                # every Chinese meaning/omission issue still blocks acceptance.
                feedback = [i for i in audit['issues'] if not (i['type']=='ledger' and i['id'] not in ledger)]
                # Do not trust or display a critic-rejected checklist. Retry its
                # Chinese from the ORIGINAL, then require a fresh source audit.
                for issue in feedback:
                    if issue['type']=='ledger':
                        ledger.pop(issue['id'],None)
                        result[issue['id']]['ledger_status']='critic_rejected'
                # Numerical mismatches and non-Chinese output cannot be waived
                # by a self-critic. Other old lexical flags are evidence for the
                # critic, not a claimed semantic certificate.
                for r in pending:
                    for flag in checks[r['id']]['issues']:
                        if flag['code'] in hard_codes:
                            feedback.append({'id':r['id'],'type':'meaning','reason':flag['detail']})
                    text = translated[r['id']]['text']
                    if chinese_ratio(text) < .45:
                        feedback.append({'id':r['id'],'type':'unclear','reason':'解释未以中文为主'})
                bad = {i['id'] for i in feedback}
                for r in pending:
                    if r['id'] not in bad:
                        result[r['id']].update(status='generated', explanation_zh=translated[r['id']]['text'],
                                               ledger=ledger.get(r['id']), reason=None)
                        if r['id'] in ledger:result[r['id']]['ledger_status']='available'
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
    return {'version':VERSION,'use_ledger':use_ledger,'source_report':report,'segments':rows,'calls':calls,'ledger_batches':ledger_batches,
            'glossary':glossary(report),'seconds':time.perf_counter()-start,
            'coverage':{'generated_segments':count,'eligible_segments':len(eligible),
                        'source_only_segments':len(eligible)-count,
                        'structural_segments':len(rows)-len(eligible),
                        'generated_with_ledger':sum(r['status']=='generated' and r['ledger'] is not None for r in eligible),
                        'generated_from_source_without_ledger':sum(r['status']=='generated' and r['ledger'] is None for r in eligible),
                        'fully_generated':bool(eligible) and count==len(eligible)},
            'notice':'研究原型。原文证据及模型核对都不能证明医学正确；未完成部分保留原文。'}
