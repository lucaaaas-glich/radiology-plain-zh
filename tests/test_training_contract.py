import torch
from radplain.train import Collator, encode_example


class ToyTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        s = "".join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        if add_generation_prompt:
            s += "<assistant>"
        return list(s.encode())


def test_only_answer_and_end_token_are_supervised():
    row = {"report": "No pneumothorax.", "reference_zh": "没有气胸。"}
    e = encode_example(ToyTokenizer(), row, 10000)
    answer = list((row["reference_zh"] + "<end>").encode())
    assert [x for x in e["labels"] if x != -100] == answer
    assert e["labels"][-len(answer):] == e["input_ids"][-len(answer):]


def test_too_long_example_is_rejected_not_silently_truncated():
    assert encode_example(ToyTokenizer(), {"report": "long", "reference_zh": "answer"}, 3) is None


def test_padding_is_masked_in_attention_and_loss():
    batch = Collator(0)([{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
                         {"input_ids": [1, 4], "labels": [-100, 4]}])
    assert batch["input_ids"].shape == (2, 8)
    assert torch.equal(batch["attention_mask"][1], torch.tensor([1, 1, 0, 0, 0, 0, 0, 0]))
    assert (batch["labels"][1, 2:] == -100).all()
