"""Adversarial development diagnostic: handcrafted pairs, not clinical accuracy."""
import argparse
from pathlib import Path
from radplain.guard_v2 import review, VERSION
from radplain.common import write_json

PAIRS = [
    ('past_value_swapped', True, '4 mm, previously 2 mm.', '以前4毫米，现在2毫米。'),
    ('wrong_unit', True, 'Follow-up in 8 months.', '8周后复查。'),
    ('lost_range', True, 'Follow-up in 5–8 weeks.', '8周后复查。'),
    ('lost_value', True, 'Nodule 4 mm, previously 2 mm.', '结节4毫米。'),
    ('invented_difference', True, '4 mm, previously 2 mm.', '现在4毫米，以前2毫米，增长3毫米。'),
    ('certainty_removed', True, 'Indeterminate nodule.', '是良性结节。'),
    ('possible_to_definite', True, 'Opacity may represent infection.', '阴影就是感染。'),
    ('negation_removed', True, 'No pneumothorax.', '存在气胸。'),
    ('negation_association', True, 'No pleural effusion or pneumothorax.', '存在胸腔积液，但没有气胸。'),
    ('added_advice', True, 'A right lung nodule.', '右肺结节，建议进行CT检查。'),
    ('side_flip', True, 'Left pleural effusion.', '右侧胸腔积液。'),
    ('above_below', True, 'Tip 2 cm above carina.', '尖端位于气管分叉下方2厘米。'),
    ('wrong_carina', True, 'Tip 2 cm above carina.', '尖端在气管环上方2厘米。'),
    ('base_omitted', True, 'Mild bibasal atelectasis.', '双肺轻度肺不张。'),
    ('direction_reversed', True, 'Left opacity decreased.', '左侧阴影增多。'),
    ('bilateral_directions_swapped', True, 'Right effusion increased, left effusion decreased.', '右侧积液减少，左侧积液增多。'),
    ('sizes_assigned_wrong_organs', True, 'Liver lesion 4 mm; renal lesion 2 mm.', '肾脏病灶4毫米；肝脏病灶2毫米。'),
    ('newness_lost', True, 'No new consolidation.', '没有肺实变。'),
    ('invented_cause', True, 'A calcified left lung nodule.', '左肺钙化结节，是以前感染造成的。'),
    ('acute_scope_lost', True, 'No acute abnormality.', '没有任何异常。'),
    ('straight_numbers', False, '4 mm, previously 2 mm.', '现在4毫米，以前2毫米。'),
    ('proper_uncertainty', False, '性质待定的右肺结节。', '右肺结节性质尚未确定。'),
    ('unit_translation', False, 'A 4 mm left lung nodule.', '左肺有4毫米的结节。'),
    ('valid_unit_conversion', False, 'A 2 cm nodule.', '有20毫米的结节。'),
    ('valid_numeric_reorder', False, '4 mm, previously 2 mm.', '从以前的2毫米变为4毫米。'),
    ('plain_atelectasis', False, 'Mild bibasal atelectasis.', '双肺底部有少量肺组织没有充分展开。'),
    ('negative_pair', False, 'No pleural effusion or pneumothorax.', '没有胸腔积液，也没有气胸。'),
    ('correct_sides', False, 'Left pleural effusion. No right pleural effusion.', '左侧有胸腔积液，右侧没有胸腔积液。'),
    ('below_synonym', False, 'Tube tip below diaphragm.', '管子尖端位于膈肌之下。'),
    ('proper_cannot_exclude', False, 'Malignancy cannot be excluded.', '目前不能排除恶性病变。'),
]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    rows=[]
    for key,problem,source,prediction in PAIRS:
        r=review(source,prediction)
        rows.append({'id':key,'expected_problem':problem,'source':source,'prediction':prediction,
                     'flagged':not r['no_flags'],'issues':r['issues']})
    summary={
        'problem_pairs':sum(r['expected_problem'] for r in rows),
        'problem_pairs_flagged':sum(r['expected_problem'] and r['flagged'] for r in rows),
        'valid_pairs':sum(not r['expected_problem'] for r in rows),
        'valid_pairs_flagged':sum(not r['expected_problem'] and r['flagged'] for r in rows),
    }
    write_json(a.out,{'version':VERSION,'note':'AI-authored development diagnostic; not a random sample or clinical validation; valid order changes/conversions are deliberately conservative flags.','summary':summary,'pairs':rows})
    print(summary)


if __name__=='__main__':
    main()
