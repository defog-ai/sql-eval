import pytest
from transformers import AutoTokenizer

from eval.hf_runner import dynamic_num_beams


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


def test_dynamic_num_beams_ranges(tokenizer):
    prompt = "word "
    prompt_4 = prompt * 1023
    num_beams_4 = dynamic_num_beams(prompt_4, tokenizer)
    assert num_beams_4 == 4
    prompt_2 = prompt * 1535
    num_beams_2 = dynamic_num_beams(prompt_2, tokenizer)
    assert num_beams_2 == 2
    prompt_1 = prompt * 2048
    num_beams_1 = dynamic_num_beams(prompt_1, tokenizer)
    assert num_beams_1 == 1
