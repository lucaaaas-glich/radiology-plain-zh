import json
import pytest
from scripts.run_p1_comparison import require_complete_generation

def test_terminal_failure_is_retained_and_allowed(tmp_path):
 (tmp_path/'manifest-000001.json').write_text(json.dumps({'counts':{'pending':0,'terminal':40,'requested':40,'completed':39,'failed':1}}))
 assert require_complete_generation(tmp_path)['counts']['failed']==1

@pytest.mark.parametrize('counts',[
 {'pending':1,'terminal':39,'requested':40,'completed':39,'failed':0},
 {'pending':0,'terminal':39,'requested':40,'completed':39,'failed':0},
 {'pending':0,'terminal':40,'requested':40,'completed':39,'failed':0}])
def test_missing_or_inconsistent_generation_never_marked_finished(tmp_path,counts):
 (tmp_path/'manifest-000001.json').write_text(json.dumps({'counts':counts}))
 with pytest.raises(RuntimeError):require_complete_generation(tmp_path)

def test_manifest_required(tmp_path):
 with pytest.raises(RuntimeError):require_complete_generation(tmp_path)
