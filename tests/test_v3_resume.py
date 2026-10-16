import json
import pytest
from scripts.resume_v3_final import completed_rows,pending_groups


def test_completed_failure_is_not_retried_and_pair_order_is_preserved(tmp_path):
    source=[{'id':str(i),'report':'source '+str(i)} for i in range(3)]
    path=tmp_path/'outputs.jsonl'
    path.write_text(json.dumps(dict(source[0],error='OutOfMemoryError'))+'\n')
    completed=completed_rows(path,{r['id']:r for r in source})
    groups=pending_groups(source,['v2','v3'],{'v2':completed,'v3':{}})
    assert groups[0]==(('v3',),source[:1])
    assert groups[1]==(('v2','v3'),source[1:])


def test_only_truncated_final_line_is_ignored(tmp_path):
    row={'id':'x','report':'source'};path=tmp_path/'outputs.jsonl'
    path.write_text(json.dumps(row)+'\n{"id":')
    assert completed_rows(path,{'x':row})=={'x':row}
    path.write_text('{bad}\n'+json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='Corrupt'):completed_rows(path,{'x':row})


def test_duplicate_completed_cases_are_rejected(tmp_path):
    row={'id':'x','report':'source'};path=tmp_path/'outputs.jsonl'
    path.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError,match='Duplicate'):completed_rows(path,{'x':row})
