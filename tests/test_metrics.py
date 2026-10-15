from radplain.evaluate import ChineseTokenizer, score_rows


def test_rouge_does_not_drop_all_chinese_tokens():
    assert "气胸" in ChineseTokenizer().tokenize("没有气胸。")
    row = {"reference_zh": "没有气胸。", "prediction": "没有气胸。", "teacher_accepted": True,
           "generation": {"completed": True}}
    scores = score_rows([row])
    assert scores["rouge1"] == 1
    assert scores["rougeL"] == 1
    assert scores["chrf"] == 100


def test_invalid_teacher_targets_are_not_scored_as_gold():
    row = {"reference_zh": "参考", "prediction": "没有气胸。", "teacher_accepted": False,
           "generation": {"completed": True}}
    scores = score_rows([row])
    assert scores["synthetic_reference_scored_rows"] == 0
    assert scores["rouge1"] is None
