import pytest
from scripts.summarize_v3 import summarize, compare, faithful
from scripts.summarize_v2 import fingerprint
from scripts.summarize_v3 import verify_cohort
from radplain.common import fingerprint as source_fingerprint


def data(error=False):
    row={'id':'x','report':'No effusion.','generation_accounting':{'seconds':2,'calls':[]}}
    if error:row['error']='generation failed'
    else:row['result']={'coverage':{'fully_generated':False},'segments':[{'status':'source_only'}]}
    case={'id':'x','output_sha256':fingerprint(row),'errors':[],
          'clinical_coverage_complete':False,'precision_concern':False,'readability':'none',
          'notes_zh':'没有完成解释，保留原句。','ledger_review':'not_available'}
    return [row],{'reviewer':'project AI, not independent clinical review','cases':[case]}


def test_refusal_never_becomes_quality_success():
    rows,review=data()
    s=summarize(rows,review)
    assert s['reports_complete_faithful_chinese']==0
    assert s['reports_with_observed_factual_errors']==0
    assert s['mutually_exclusive_outcomes']=={'incomplete_without_observed_error':1}


def test_changed_trace_invalidates_review():
    rows,review=data();rows[0]['report']='Effusion.'
    with pytest.raises(ValueError,match='changed'):summarize(rows,review)


def test_failed_generation_is_counted_and_cannot_be_complete():
    rows,review=data(True)
    assert summarize(rows,review)['generation_failures']==1
    review['cases'][0]['clinical_coverage_complete']=True
    with pytest.raises(ValueError):summarize(rows,review)


def test_pair_comparison_rejects_same_id_with_different_original():
    a,ra=data();b,rb=data();b[0]['report']='Small effusion.';rb['cases'][0]['output_sha256']=fingerprint(b[0])
    with pytest.raises(ValueError,match='different source'):compare(a,ra,b,rb)


def test_precision_concern_is_not_declared_faithful():
    _,review=data();c=review['cases'][0];c['clinical_coverage_complete']=True;c['precision_concern']=True
    assert not faithful(c)


def test_final_summary_rejects_missing_cases_and_changed_source():
    rows=[{'id':str(i),'report':'Source '+str(i)} for i in range(20)]
    manifest={'rows':[{'id':r['id'],'source_sha256':source_fingerprint(r['report'])} for r in rows]}
    verify_cohort(rows,manifest,'final')
    verify_cohort(rows[::2],manifest,'ablation')
    with pytest.raises(ValueError,match='complete'):verify_cohort(rows[:-1],manifest,'final')
    rows[0]['report']='Different source'
    with pytest.raises(ValueError,match='source report'):verify_cohort(rows,manifest,'final')
