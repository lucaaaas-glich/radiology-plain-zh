"""Small V3 checks motivated by observed OLD-report errors."""
import re
from .guard_v2 import review as old_review


def review(source,prediction,finish_reason='stop'):
    result=old_review(source,prediction,finish_reason)
    present={code:bool(re.search(r'\b'+code+r'\s+(?:(?:portable|chest)\s+)*(?:X[- ]?ray|radiograph|view|projection)\b',source,re.I))
             for code in ['PA','AP']}
    if ((present['PA'] and not present['AP'] and '前后位' in prediction)
            or (present['AP'] and not present['PA'] and '后前位' in prediction)):
        result['issues'].append({'code':'projection_direction','detail':'胸片投照方向被对调：PA是后前位，AP是前后位；请忠实保留原文，不能按常见检查习惯猜测。'})
    result['no_flags']=not result['issues']
    return result
