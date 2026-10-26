from radplain.render_v3 import render_html, public_result


def test_display_escapes_source_and_excludes_rejected_candidates():
    result={'source_report':'<script>alert(1)</script>','segments':[{
        'id':1,'source':'<script>alert(1)</script>','status':'source_only',
        'explanation_zh':None,'reason':'未完成','ledger':None,
        'attempts':[{'text':'SECRET_REJECTED_DRAFT'}]}],
        'coverage':{'generated_segments':0,'eligible_segments':1,'source_only_segments':1},
        'calls':[{'text':'SECRET_REJECTED_DRAFT'}],'glossary':[],'notice':'研究用途'}
    html=render_html(result)
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert 'SECRET_REJECTED_DRAFT' not in html
    assert '不是医学准确率' in html
    public=public_result(result)
    assert 'calls' not in public and 'attempts' not in public['segments'][0]
