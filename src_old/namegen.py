from pydantic import BaseModel, model_validator, ConfigDict
from src.classes import Trie, Vocab
from llm_sdk.llm_sdk import Small_LLM_Model
from src.helpers import replace_space, get_mask
from src.error import VocabError, EncodeError
from src.coder import Coder
from typing import Self
import numpy as np


class NameGenerator(BaseModel):
    """
    ng =
    NameGenerator(function_names=list[str],
    llm_vocab=Vocab(),
    llm_prompt=str,
    coder=Coder)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    function_names: list[str]
    llm_vocab: Vocab
    llm_prompt: list[int]
    coder: Coder
    trie_vocab: Vocab = Vocab(
        vocab={
            0: "_",
            1: "{",
            2: "}",
            3: ",",
            4: ":",
            5: "a",
            6: "b",
            7: "c",
            8: "d",
            9: "e",
            10: "f",
            11: "g",
            12: "h",
            13: "i",
            14: "j",
            15: "k",
            16: "l",
            17: "m",
            18: "n",
            19: "o",
            20: "p",
            21: "q",
            22: "r",
            23: "s",
            24: "t",
            25: "u",
            26: "v",
            27: "w",
            28: "x",
            29: "y",
            30: "z",
        }
    )
    trie_functions: Trie | None = None

    @model_validator(mode="after")
    def load(self) -> Self:
        if self.trie_functions is None:
            self.trie_functions = Trie(
                vocab=self.trie_vocab, entries=self.function_names
            )
        return self

    @property
    def trie_entries(self) -> Trie:
        assert self.trie_functions is not None
        return self.trie_functions

    def generate(self, llm: Small_LLM_Model, prompt: str) -> str:
        """
        namegen.generate(llm_model, "function specific prompt")
        """
        generated = ""
        generating = True
        input_ids = []
        not_allowed: list[int] = []
        prompts = f"Answer this prompt: {prompt}"
        text = replace_space(prompts)
        try:
            input_ids = self.llm_prompt + self.coder.encode(text)
        except EncodeError:
            input_ids = (
                self.llm_prompt + llm.encode(prompts).squeeze(0).tolist()
            )
        while generating is True:
            print(f"Generating: {generated!r}", flush=True)
            logits = np.array(llm.get_logits_from_input_ids(input_ids))
            allowed = self._allowed(generated)
            mask = get_mask(logits, allowed, not_allowed)
            masked = logits + mask
            next_id = np.argmax(masked)
            if self.trie_entries.search(generated):
                generating = False
            else:
                temp = self.llm_vocab.vocab.get(int(next_id))
                if temp is None:
                    raise VocabError(
                        "ID could not be found in the LLM vocab,\
                         generation aborting."
                    )
                temp_gen = generated + temp
                if self.trie_entries.is_prefix(temp_gen):
                    not_allowed = []
                    input_ids.append(int(next_id))
                    generated += temp
                else:
                    not_allowed.append(int(next_id))
        return generated

    def _allowed(self, generated: str) -> list[int]:
        allowed = []
        for value in self.llm_vocab.vocab.values():
            if self.trie_entries.is_prefix(generated + value):
                id = self.llm_vocab.inverted.get(value)
                if id is not None:
                    allowed.append(id)
            elif self.trie_entries.search(generated + value):
                id = self.llm_vocab.inverted.get(value)
                if id is not None:
                    allowed.append(id)
            else:
                continue
        return allowed
