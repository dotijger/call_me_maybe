import pytest
from unittest.mock import Mock
from new.setup import ConstrainedDecoder
from llm_sdk.llm_sdk import Small_LLM_Model
from new.classes import Vocab

TINY_VOCAB = {0: '"', 1: "a", 2: "b", 3: "{", 4: "}"}


@pytest.fixture
def decoder():
    return ConstrainedDecoder.model_construct(
        llm_vocab=Vocab(
            vocab=TINY_VOCAB, inverted={v: k for k, v in TINY_VOCAB.items()}
        )
    )


def test_find_match(decoder):
    assert decoder._find_match('"') == 0


def test_is_prefix(decoder):
    assert decoder._is_prefix("ab", "abc") is True
